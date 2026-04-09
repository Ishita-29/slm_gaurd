"""
Ablation Study for SLM-Guard
==============================
Systematically removes or changes one component at a time to measure
its contribution to overall performance.

Ablations:
  A1 — Binary-only loss     (no multiclass head training, alpha=1.0)
  A2 — Multiclass-only loss (no binary focal loss, alpha=0.0)
  A3 — Multi-task (default) (alpha=0.7)  ← proposed design
  A4 — No focal loss        (standard BCE instead of focal)
  A5 — No class weighting   (benign_weight=1.0 instead of 11.0)
  A6 — No frozen warm-up    (freeze_epochs=0)

Each ablation trains from scratch and evaluates on the same test set.
Results are saved to ablation_results.json.

Usage:
  python ablation_study.py                      # all ablations
  python ablation_study.py --ablations A1 A3    # specific ones only
  python ablation_study.py --quick              # 2 epochs only (for debugging)
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score, accuracy_score
from transformers import DebertaV2Tokenizer

import sys
sys.path.insert(0, ".")
from config import ALL_LABELS
from train import SLMGuardModel, SLMGuardTrainer, preprocess_function, BENIGN_WEIGHT
from transformers import TrainingArguments, EarlyStoppingCallback
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = "../data/final/slmguard_dataset"
OUT_BASE  = "../checkpoints/ablations"


@dataclass
class AblationConfig:
    name:         str
    description:  str
    alpha:        float = 0.7      # binary loss weight (1.0 = binary-only, 0.0 = mc-only)
    use_focal:    bool  = True     # use focal loss vs standard BCE
    benign_weight:float = 11.0     # class weight for benign samples
    freeze_epochs:int   = 1        # frozen warm-up epochs
    epochs:       int   = 5
    batch_size:   int   = 16
    lr:           float = 2e-5


ABLATIONS = {
    "A1": AblationConfig(
        name="A1 — Binary-only loss",
        description="Only binary head trained (alpha=1.0). Multiclass head receives no gradient. "
                    "This is the 'single-task' baseline — tests whether multi-task learning helps.",
        alpha=1.0,
    ),
    "A2": AblationConfig(
        name="A2 — Multiclass-only loss",
        description="Only 12-class head trained (alpha=0.0). Binary prediction derived from "
                    "P(label != benign). Tests whether multiclass supervision is sufficient alone.",
        alpha=0.0,
    ),
    "A3": AblationConfig(
        name="A3 — Multi-task (proposed)",
        description="Joint binary + multiclass loss (alpha=0.7). Full proposed design. "
                    "This should outperform A1 and A2.",
        alpha=0.7,
    ),
    "A4": AblationConfig(
        name="A4 — Standard BCE (no focal)",
        description="Multi-task loss but binary term uses plain BCE instead of focal loss. "
                    "Tests contribution of focal weighting on hard examples.",
        alpha=0.7,
        use_focal=False,
    ),
    "A5": AblationConfig(
        name="A5 — No class weighting",
        description="Multi-task loss but benign_weight=1.0 (no upweighting). "
                    "Tests contribution of class reweighting for 11:1 imbalance.",
        alpha=0.7,
        benign_weight=1.0,
    ),
    "A6": AblationConfig(
        name="A6 — No frozen warm-up",
        description="Full model trained from epoch 0 without head-only warm-up phase. "
                    "Tests whether frozen warm-up improves convergence stability.",
        alpha=0.7,
        freeze_epochs=0,
    ),
}


class AblationTrainer(SLMGuardTrainer):
    """Extends SLMGuardTrainer with ablation-specific loss configurations."""

    def __init__(self, *args, alpha: float = 0.7, use_focal: bool = True,
                 benign_weight: float = BENIGN_WEIGHT, **kwargs):
        super().__init__(*args, benign_weight=benign_weight, **kwargs)
        self.alpha      = alpha
        self.use_focal  = use_focal

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs  = dict(inputs)
        labels  = inputs.pop("labels")
        is_se   = inputs.pop("is_se")
        forward_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}

        outputs           = model(**forward_inputs)
        binary_logits     = outputs["binary_logit"]
        multiclass_logits = outputs["multiclass_logits"]

        # Binary loss
        bce_raw = nn.functional.binary_cross_entropy_with_logits(
            binary_logits, is_se.float(), reduction="none"
        )
        if self.use_focal:
            p_t          = torch.exp(-bce_raw)
            focal_weight = (1 - p_t) ** 2
        else:
            focal_weight = torch.ones_like(bce_raw)

        sample_weights = torch.where(
            is_se == 0,
            torch.full_like(is_se, self.benign_weight, dtype=torch.float),
            torch.ones_like(is_se, dtype=torch.float),
        )
        binary_loss = (focal_weight * bce_raw * sample_weights).mean()

        # Multiclass loss
        multiclass_loss = nn.functional.cross_entropy(
            multiclass_logits, labels.long(), reduction="mean"
        )

        # Joint loss based on alpha
        if self.alpha >= 1.0:
            loss = binary_loss
        elif self.alpha <= 0.0:
            loss = multiclass_loss
        else:
            loss = self.alpha * binary_loss + (1.0 - self.alpha) * multiclass_loss

        if torch.isnan(loss):
            log.warning("NaN loss — skipping batch")
            loss = torch.tensor(0.0, requires_grad=True, device=binary_logits.device)

        return (loss, outputs) if return_outputs else loss


@torch.inference_mode()
def evaluate_model(model, tokenizer, test_ds, threshold=0.4, alpha=0.7):
    """Run inference and compute metrics on test set."""
    model.eval()
    texts       = list(test_ds["text"])
    true_binary = np.array(test_ds["is_se"])
    true_labels = np.array(test_ds["label_id"])

    all_binary_probs = []
    all_multiclass_probs = []
    BATCH = 32

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tokenizer(
            batch, truncation=True, max_length=256,
            padding=True, return_tensors="pt"
        ).to(DEVICE)
        out = model(**enc)
        bp  = torch.sigmoid(out["binary_logit"]).cpu().float().numpy()
        mp  = torch.softmax(out["multiclass_logits"], dim=-1).cpu().float().numpy()
        all_binary_probs.extend(bp.tolist())
        all_multiclass_probs.extend(mp.tolist())

    binary_probs     = np.array(all_binary_probs)
    multiclass_probs = np.array(all_multiclass_probs)

    # For alpha=0 (multiclass-only), derive binary from P(not benign)
    if alpha <= 0.0:
        pred_binary = (1 - multiclass_probs[:, 0] >= threshold).astype(int)
    else:
        pred_binary = (binary_probs >= threshold).astype(int)

    pred_labels = multiclass_probs.argmax(-1)

    binary_f1   = f1_score(true_binary, pred_binary, zero_division=0)
    binary_acc  = accuracy_score(true_binary, pred_binary)
    macro_f1    = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    tp = int(((pred_binary == 1) & (true_binary == 1)).sum())
    fp = int(((pred_binary == 1) & (true_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (true_binary == 1)).sum())
    tn = int(((pred_binary == 0) & (true_binary == 0)).sum())
    fpr = fp / (fp + tn + 1e-9)
    fnr = fn / (fn + tp + 1e-9)

    return {
        "binary_f1":   float(binary_f1),
        "binary_acc":  float(binary_acc),
        "macro_f1":    float(macro_f1),
        "fpr":         float(fpr),
        "fnr":         float(fnr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def run_ablation(ablation_id: str, cfg: AblationConfig, dataset,
                 tokenizer, quick: bool = False):
    log.info(f"\n{'='*60}")
    log.info(f"Running {cfg.name}")
    log.info(f"  {cfg.description}")
    log.info(f"  alpha={cfg.alpha}  focal={cfg.use_focal}  "
             f"benign_w={cfg.benign_weight}  freeze={cfg.freeze_epochs}")

    output_dir = f"{OUT_BASE}/{ablation_id.lower()}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epochs     = 2 if quick else cfg.epochs
    use_bf16   = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Tokenize
    tokenized = dataset.map(
        lambda x: preprocess_function(x, tokenizer, 256),
        batched=True, num_proc=4, load_from_cache_file=False,
        remove_columns=["text", "source", "novel"],
    )
    tokenized = tokenized.map(
        lambda x: {"labels": x["label_id"], "is_se": x["is_se"]},
        batched=True, num_proc=4, load_from_cache_file=False,
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels", "is_se"])

    model = SLMGuardModel(
        model_name="answerdotai/ModernBERT-large",
        model_key="modernbert",
        use_lora=True,
    ).to(DEVICE)

    # Phase 1: frozen warm-up (if configured)
    if cfg.freeze_epochs > 0:
        model.freeze_encoder()
        args_frozen = TrainingArguments(
            output_dir=f"{output_dir}/warmup",
            num_train_epochs=cfg.freeze_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size * 2,
            learning_rate=1e-3,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_strategy="no",
            eval_strategy="epoch",
            logging_steps=50,
            bf16=use_bf16, fp16=False,
            max_grad_norm=1.0,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            seed=42,
        )
        t_frozen = AblationTrainer(
            model=model, args=args_frozen,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            alpha=cfg.alpha, use_focal=cfg.use_focal,
            benign_weight=cfg.benign_weight,
        )
        t_frozen.train()
        model.unfreeze_encoder()

    # Phase 2: fine-tuning
    remaining = (epochs - cfg.freeze_epochs) if not quick else epochs
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=max(1, remaining),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        seed=42,
        bf16=use_bf16, fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_grad_norm=0.3,
        optim="adamw_torch",
    )
    trainer = AblationTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        alpha=cfg.alpha, use_focal=cfg.use_focal,
        benign_weight=cfg.benign_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # Evaluate
    log.info("Evaluating on test set...")
    model = model.float().to(DEVICE)
    metrics = evaluate_model(model, tokenizer, dataset["test"], alpha=cfg.alpha)

    log.info(f"  Binary F1 : {metrics['binary_f1']:.4f}")
    log.info(f"  Macro  F1 : {metrics['macro_f1']:.4f}")
    log.info(f"  FPR       : {metrics['fpr']:.4f}")
    log.info(f"  FNR       : {metrics['fnr']:.4f}")

    return metrics


def main(ablations_to_run: List[str], quick: bool = False):
    log.info(f"Loading dataset from {DATA_DIR}")
    dataset   = load_from_disk(DATA_DIR)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    all_results = {}

    for ablation_id in ablations_to_run:
        if ablation_id not in ABLATIONS:
            log.warning(f"Unknown ablation: {ablation_id} — skipping")
            continue
        cfg = ABLATIONS[ablation_id]
        try:
            metrics = run_ablation(ablation_id, cfg, dataset, tokenizer, quick=quick)
            all_results[ablation_id] = {"config": cfg.__dict__, "metrics": metrics}
        except Exception as e:
            log.error(f"Ablation {ablation_id} failed: {e}")
            all_results[ablation_id] = {"error": str(e)}

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'ID':<4}  {'Description':<35} {'Bin F1':>7} {'FPR':>6} {'FNR':>6} {'Mac F1':>7}")
    print("─" * 70)

    for ablation_id in ablations_to_run:
        if ablation_id not in all_results:
            continue
        res = all_results[ablation_id]
        if "error" in res:
            print(f"{ablation_id:<4}  {'ERROR: ' + res['error'][:30]:<35}")
            continue
        m = res["metrics"]
        cfg = ABLATIONS[ablation_id]
        name = cfg.name.split("—")[1].strip()[:34]
        star = " ←" if ablation_id == "A3" else ""
        print(f"{ablation_id:<4}  {name:<35} {m['binary_f1']:>7.4f} "
              f"{m['fpr']:>6.4f} {m['fnr']:>6.4f} {m['macro_f1']:>7.4f}{star}")

    print("─" * 70)
    print("  ← = Proposed design (A3)")
    print("  Each row isolates contribution of one design decision.")
    print("="*70)

    # Save
    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    out_path = f"{OUT_BASE}/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for SLM-Guard")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()),
                        choices=list(ABLATIONS.keys()),
                        help="Which ablations to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="2-epoch run for debugging (not for final results)")
    args = parser.parse_args()

    main(ablations_to_run=args.ablations, quick=args.quick)
