
import argparse
import json
import logging
import torch
from pathlib import Path
import torchvision.io

if not hasattr(torchvision.io, "VideoReader"):
    class _DummyVideoReader:
        pass
    torchvision.io.VideoReader = _DummyVideoReader

from datasets import load_from_disk
from transformers import (
    DebertaV2Tokenizer, DebertaV2Model,
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    set_seed,
)
from torch import nn
from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import ALL_LABELS, LABEL2ID, ID2LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = "../data/final/slmguard_dataset"
MODEL_DIR = "../checkpoints/slmguard-v2"

MODEL_ALIASES = {
    "deberta":      "microsoft/deberta-v3-large",   # 304M — original backbone
    "deberta-base": "microsoft/deberta-v3-base",    # 86M  — lightweight
    "modernbert":   "answerdotai/ModernBERT-large",  # 395M — modern encoder
    "phi4mini":     "microsoft/Phi-4-mini-instruct", # 3.8B — decoder (requires transformers 4.x)
    "qwen25":       "Qwen/Qwen2.5-1.5B",            # 1.5B — decoder, native transformers 5.x
}


LORA_TARGETS = {
    "deberta":      ["query_proj", "value_proj"],
    "deberta-base": ["query_proj", "value_proj"],
    "modernbert":   ["Wqkv", "Wo"],
    "phi4mini":     ["q_proj", "v_proj"],
    "qwen25":       ["q_proj", "v_proj"],
}

HARM0_WEIGHT = 1.06    
DECODER_FAMILIES = {"phi4mini", "qwen25"}


class SLMGuardModel(nn.Module):
    """
    Multi-task SE-detection model on top of a shared encoder or decoder backbone.

    Two jointly trained heads share the same encoder representation:
      - binary_head     : sigmoid output → P(input is harmful), trained on harm_label
      - multiclass_head : softmax over 12 classes (no_se + 11 SE subtypes), auxiliary

    Joint loss = α·focal_binary_loss + (1-α)·cross_entropy_multiclass
    where α=0.7 by default (binary task weighted more for deployment use).

    Encoder models (DeBERTa, ModernBERT): pool the [CLS] token.
    Decoder models (Phi-4-mini, Qwen2.5): pool the last non-padding token.
    """

    def __init__(
        self,
        model_name:  str  = "answerdotai/ModernBERT-large",
        model_key:   str  = "modernbert",
        use_lora:    bool = False,
        use_int8:    bool = False,
    ):
        super().__init__()
        self.model_name  = model_name
        self.model_key   = model_key
        self.is_decoder  = model_key in DECODER_FAMILIES
        self.use_lora    = use_lora

        if self.is_decoder:
            bnb_cfg = None
            if use_int8:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                log.info("INT8 quantisation enabled via bitsandbytes")
            self.encoder = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if (not use_int8 and torch.cuda.is_available()) else torch.float32,
                device_map="auto" if use_int8 else None,
            )
            hidden_size = self.encoder.config.hidden_size
        elif "deberta" in model_name.lower():
            self.encoder = DebertaV2Model.from_pretrained(model_name)
            hidden_size  = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            hidden_size  = self.encoder.config.hidden_size

        if use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            target_modules = LORA_TARGETS.get(model_key, ["q_proj", "v_proj"])
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            self.encoder.print_trainable_parameters()

        self.binary_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.multiclass_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(ALL_LABELS)),
        )

        for head in [self.binary_head, self.multiclass_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        if self.use_lora:

            for name, p in self.encoder.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
        else:
            for p in self.encoder.parameters():
                p.requires_grad = True

    def _pool(self, outputs, attention_mask):
        """Pool hidden states: CLS for encoders, last non-pad token for decoders."""
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        if self.is_decoder:
            
            seq_lens = attention_mask.sum(dim=1) - 1          
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_idx, seq_lens]          
        else:
            pooled = last_hidden[:, 0]                         
        return pooled.float()                                  

    def forward(self, input_ids, attention_mask, **kwargs):
        if self.is_decoder:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            last_hidden = outputs.hidden_states[-1]

            class _Out:
                pass
            o = _Out()
            o.last_hidden_state = last_hidden
            pooled = self._pool(o, attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled  = self._pool(outputs, attention_mask)

        return {
            "binary_logit":      self.binary_head(pooled).squeeze(-1),
            "multiclass_logits": self.multiclass_head(pooled),
        }


def preprocess_function(examples, tokenizer, max_length=512):
  
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
    )


class SLMGuardTrainer(Trainer):
    """
    Multi-task trainer: jointly optimises binary harm detection + 12-class SE-subtype
    classification (the latter is auxiliary — it never gates the block/allow decision).

    Loss = alpha * L_binary + (1 - alpha) * L_multiclass

    L_binary     : focal BCE with class weighting, target = harm_label (not is_se)
    L_multiclass : standard cross-entropy over 12 SE subtypes + no_se
    alpha        : 0.7 — binary task weighted more because deployment goal is block/pass
    gamma        : 2   — focal loss exponent; downweights easy examples
    harm0_weight : mild upweighting for harm_label=0 (see HARM0_WEIGHT)
    """

    def __init__(self, *args, harm0_weight: float = HARM0_WEIGHT,
                 alpha: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.harm0_weight = harm0_weight
        self.alpha = alpha           

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs     = dict(inputs)
        labels     = inputs.pop("labels")               
        harm_label = inputs.pop("harm_label")            
        forward_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}

        outputs           = model(**forward_inputs)
        binary_logits     = outputs["binary_logit"].float()      
        multiclass_logits = outputs["multiclass_logits"].float() 

        
        binary_logits     = torch.clamp(binary_logits, -20, 20)
        multiclass_logits = torch.clamp(multiclass_logits, -50, 50)

        
        bce_raw = nn.functional.binary_cross_entropy_with_logits(
            binary_logits, harm_label.float(), reduction="none"
        )
        p_t          = torch.exp(-bce_raw)
        focal_weight = (1 - p_t) ** 2                  
        sample_weights = torch.where(
            harm_label == 0,
            torch.full_like(harm_label, self.harm0_weight, dtype=torch.float),
            torch.ones_like(harm_label, dtype=torch.float),
        )
        binary_loss = (focal_weight * bce_raw * sample_weights).mean()

        multiclass_loss = nn.functional.cross_entropy(
            multiclass_logits, labels.long(), reduction="mean"
        )

        loss = self.alpha * binary_loss + (1.0 - self.alpha) * multiclass_loss

        if torch.isnan(loss):
            log.warning("NaN loss — skipping batch")
            loss = torch.tensor(0.0, requires_grad=True, device=binary_logits.device)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        if torch.isnan(loss):
            loss = torch.tensor(float("nan"))
        return (loss.detach(), None, None)

    def _save(self, output_dir, state_dict=None):
        import os
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        # Save as pytorch_model.bin to avoid safetensors shared-memory error
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def train(
    data_dir:      str   = DATA_DIR,
    output_dir:    str   = MODEL_DIR,
    model_key:     str   = "modernbert",
    epochs:        int   = 8,
    batch_size:    int   = 16,
    lr:            float = 2e-5,
    max_length:    int   = 512,
    freeze_epochs: int   = 1,
    warmup_ratio:  float = 0.06,
    weight_decay:  float = 0.01,
    patience:      int   = 3,
    use_lora:      bool  = False,
    use_int8:      bool  = False,
    seed:          int   = 42,
    alpha:         float = 0.7,
):
    
    set_seed(seed)
    model_name = MODEL_ALIASES.get(model_key, model_key)
    log.info(f"Device      : {DEVICE}")
    log.info(f"Backbone    : {model_name}")
    log.info(f"LoRA        : {use_lora}  |  INT8 : {use_int8}")
    log.info(f"Epochs      : {epochs}  (frozen warm-up: {freeze_epochs})")
    log.info(f"Batch size  : {batch_size}  |  LR : {lr}")
    log.info(f"Seed        : {seed}  |  alpha : {alpha}")

   
    log.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)

  
    if "deberta" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Decoder models need a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

   
    log.info("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        num_proc=4,
        load_from_cache_file=False,
        remove_columns=["text", "source"],
    )
    tokenized = tokenized.map(
        lambda x: {"labels": x["label_id"], "harm_label": x["harm_label"]},
        batched=True,
        num_proc=4,
        load_from_cache_file=False,
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels", "harm_label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    log.info("Initialising model...")
    model = SLMGuardModel(
        model_name=model_name,
        model_key=model_key,
        use_lora=use_lora,
        use_int8=use_int8,
    )
    # INT8 models are device-mapped by bitsandbytes — don't call .to(DEVICE)
    if not use_int8:
        model = model.to(DEVICE)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

   
    if freeze_epochs > 0 and not use_int8:
        log.info(f"Phase 1: Frozen encoder warm-up ({freeze_epochs} epoch(s))...")
        model.freeze_encoder()

        args_frozen = TrainingArguments(
            output_dir=f"{output_dir}/warmup",
            num_train_epochs=freeze_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=1e-3,
            warmup_ratio=0.1,
            weight_decay=weight_decay,
            save_strategy="no",
            eval_strategy="epoch",
            logging_steps=50,
            fp16=False,
            bf16=False,
            max_grad_norm=0.5,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            seed=seed,
        )
        trainer_frozen = SLMGuardTrainer(
            model=model,
            args=args_frozen,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
            harm0_weight=HARM0_WEIGHT,
            alpha=alpha,
        )
        trainer_frozen.train()
        log.info("Phase 1 complete — unfreezing encoder")
        model.unfreeze_encoder()
    elif use_int8:
        log.info("Skipping Phase 1 frozen warm-up (INT8 mode — adapters always active)")

    
    remaining_epochs = epochs - freeze_epochs if not use_int8 else epochs
    log.info(f"Phase 2: Fine-tuning ({remaining_epochs} epoch(s))...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=remaining_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=seed,
        fp16=False,
        bf16=use_bf16 and "deberta" not in model_name.lower() and not use_int8,  
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_grad_norm=0.1,                  
        optim="adamw_torch",
    )

    trainer = SLMGuardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        harm0_weight=HARM0_WEIGHT,
        alpha=alpha,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()

    
    log.info(f"Saving model to {output_dir}")
    if use_int8:
        # INT8: save LoRA adapters separately + heads in FP32
        model.encoder.save_pretrained(output_dir)
    else:
        model_fp32 = model.float()
        torch.save(model_fp32.state_dict(), f"{output_dir}/pytorch_model.bin")
        model_fp32.encoder.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    config = {
        "num_labels":    len(ALL_LABELS),
        "labels":        ALL_LABELS,
        "max_length":    max_length,
        "model_name":    model_name,
        "model_key":     model_key,
        "use_lora":      use_lora,
        "use_int8":      use_int8,
        "harm0_weight": HARM0_WEIGHT,
        "alpha":         alpha,
        "is_decoder":    model_key in DECODER_FAMILIES,
    }
    with open(f"{output_dir}/slmguard_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n Training complete")
    print(f"  Model saved → {output_dir}")
    print(f"  Next step  : python evaluate_binary.py --checkpoint {output_dir} --split test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SLM-Guard binary classifier")
    parser.add_argument("--data-dir",      default=DATA_DIR)
    parser.add_argument("--output-dir",    default=MODEL_DIR)
    parser.add_argument("--model",         default="modernbert",
                        choices=list(MODEL_ALIASES.keys()) + ["custom"],
                        help="Backbone: deberta | deberta-base | modernbert | qwen25 | phi4mini")
    parser.add_argument("--epochs",        type=int,   default=8)
    parser.add_argument("--freeze-epochs", type=int,   default=1)
    parser.add_argument("--batch",         type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--max-length",    type=int,   default=512)
    parser.add_argument("--patience",      type=int,   default=3)
    parser.add_argument("--lora",          action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--int8",          action="store_true", help="Enable INT8 quantisation (Phi-4-mini only)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed — vary this and nothing else for seed-averaging runs")
    parser.add_argument("--alpha",         type=float, default=0.7,
                        help="Weight of binary loss in joint loss (1-alpha goes to multiclass)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_length=args.max_length,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        use_lora=args.lora,
        use_int8=args.int8,
        seed=args.seed,
        alpha=args.alpha,
    )
