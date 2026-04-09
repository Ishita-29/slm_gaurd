"""
Evaluate trained SLM-Guard model.
Reports per-subtype F1, FPR on hard negatives, latency.

Usage: python evaluate.py
       python evaluate.py --checkpoint ../checkpoints/slmguard-v1
"""

import argparse
import json
import time
import logging
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score,
)
from transformers import DebertaV2Tokenizer

import sys
sys.path.insert(0, ".")
from config import ALL_LABELS, LABEL2ID, ID2LABEL
from train import SLMGuardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../data/final/slmguard_dataset"


@torch.inference_mode()
def predict_batch(model, tokenizer, texts, max_length=256):
    """Batch inference"""
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    out = model(**enc)
    binary_probs = torch.sigmoid(out["binary_logit"]).cpu().float().numpy()
    multiclass_probs = torch.softmax(out["multiclass_logits"], dim=-1).cpu().float().numpy()
    
    return binary_probs, multiclass_probs


def evaluate(checkpoint_dir: str = "../checkpoints/slmguard-v1", threshold: float = 0.4):
    """Main evaluation pipeline"""
    
    log.info(f"Loading model from {checkpoint_dir}")

    # Read saved config to get correct backbone / LoRA settings
    import json
    cfg_path = Path(checkpoint_dir) / "slmguard_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved_cfg = json.load(f)
        model_key  = saved_cfg.get("model_key",  "deberta")
        model_name = saved_cfg.get("model_name", "microsoft/deberta-v3-large")
        use_lora   = saved_cfg.get("use_lora",   False)
        log.info(f"Config: model_key={model_key}, use_lora={use_lora}")
    else:
        model_key, model_name, use_lora = "deberta", "microsoft/deberta-v3-large", False

    # Load tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Build model with correct architecture
    model = SLMGuardModel(model_name=model_name, model_key=model_key, use_lora=use_lora)

    # Load weights
    weight_path = Path(checkpoint_dir) / "pytorch_model.bin"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        log.info("Loaded pytorch_model.bin")
    else:
        try:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(Path(checkpoint_dir) / "model.safetensors"))
            log.info("Loaded model.safetensors")
        except Exception as e:
            log.error(f"Failed to load model weights: {e}")
            return 1

    model = model.float().to(DEVICE).eval()  # force float32 throughout
    
    # Load test set
    log.info(f"Loading test set from {DATA_DIR}")
    ds = load_from_disk(DATA_DIR)
    test = ds["test"]
    
    texts = test["text"]
    true_labels = np.array(test["label_id"])
    true_binary = np.array(test["is_se"])
    
    # Inference
    log.info("Running inference...")
    BATCH = 32
    all_binary_probs = []
    all_multiclass_probs = []
    latencies = []
    
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i+BATCH]
        t0 = time.perf_counter()
        bp, mp = predict_batch(model, tokenizer, batch_texts)
        latencies.append((time.perf_counter() - t0) * 1000 / len(batch_texts))
        all_binary_probs.extend(bp.tolist())
        all_multiclass_probs.extend(mp.tolist())
    
    binary_probs = np.array(all_binary_probs)
    multiclass_probs = np.array(all_multiclass_probs)
    
    # Predictions
    pred_labels = multiclass_probs.argmax(-1)
    pred_binary = (binary_probs >= threshold).astype(int)
    
    # Results
    print("\n" + "="*70)
    print("  SLM-Guard Evaluation Results")
    print("="*70)
    
    # Binary metrics
    binary_f1 = f1_score(true_binary, pred_binary, zero_division=0)
    binary_acc = accuracy_score(true_binary, pred_binary)
    try:
        binary_auc = roc_auc_score(true_binary, binary_probs)
    except:
        binary_auc = 0.0
    
    print(f"\nBinary SE Detection:")
    print(f"  F1-score  : {binary_f1:.4f}")
    print(f"  Accuracy  : {binary_acc:.4f}")
    print(f"  AUC-ROC   : {binary_auc:.4f}")
    
    tp = ((pred_binary == 1) & (true_binary == 1)).sum()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum()
    tn = ((pred_binary == 0) & (true_binary == 0)).sum()
    
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  FPR (benign) : {fp/(fp+tn+1e-9):.4f}")
    print(f"  FNR (attacks): {fn/(fn+tp+1e-9):.4f}")
    
    # Multi-class metrics
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    print(f"\nMulti-class Classification (12 tactics):")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Per-class report
    print(f"\nPer-Class Metrics:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=ALL_LABELS,
        zero_division=0,
        digits=4,
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(ALL_LABELS)))
    
    # Hard negatives
    if "source" in test.column_names:
        hard_neg_mask = np.array([
            "hard_negative" in str(s).lower() or "synthetic_hard_negative" in str(s).lower()
            for s in test["source"]
        ])
        if hard_neg_mask.sum() > 0:
            hn_preds = pred_binary[hard_neg_mask]
            hn_fpr = hn_preds.mean()
            print(f"\nHard Negative Analysis:")
            print(f"  FPR on hard negatives: {hn_fpr:.4f}  (target < 0.05)")
            print(f"  Sample count        : {hard_neg_mask.sum()}")
    
    # Latency
    lat = np.array(latencies)
    print(f"\nLatency per sample ({DEVICE}):")
    print(f"  p50: {np.percentile(lat, 50):.1f}ms")
    print(f"  p95: {np.percentile(lat, 95):.1f}ms  (target < 50ms)")
    print(f"  p99: {np.percentile(lat, 99):.1f}ms")
    
    # Save results
    results = {
        "binary": {
            "f1": float(binary_f1),
            "accuracy": float(binary_acc),
            "auc": float(binary_auc),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
        "multiclass": {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
        }
    }
    
    output_path = Path(checkpoint_dir) / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {output_path}")
    
    print("="*70)
    print("✓ Evaluation complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/slmguard-v1",
                        help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Binary classification threshold (default: 0.4, matches thesis results)")
    args = parser.parse_args()

    exit(evaluate(args.checkpoint, threshold=args.threshold))