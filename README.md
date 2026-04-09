# SLM-Guard

Psychology-grounded middleware classifier for detecting social engineering attacks on LLMs.

Thesis: *Detecting Persuasion, Not Just Harm: A Psychology-Grounded Middleware for Social Engineering Defense in Large Language Models*

---

## Repository Structure

```
src/
  config.py                  # Taxonomy, labels, Cialdini mappings
  train.py                   # Main training script (ModernBERT + LoRA)
  evaluate.py                # Binary evaluation (use --threshold 0.4)
  evaluate_binary.py         # Alternative binary eval with threshold support
  train_cialdini_head.py     # Cialdini group head training (frozen encoder)
  middleware_eval.py         # End-to-end middleware evaluation against target LLMs
  middleware_eval_ood.py     # OOD middleware evaluation
  ood_eval.py                # Standalone OOD classification evaluation
  synthetic_generator.py     # SE-Bench generation pipeline
  quality_filter.py          # Claude Opus quality validation

data/
  final/slmguard_dataset/    # SE-Bench (train/val/test splits)
  ood/                       # OOD benchmarks: JailbreakHub, ToxicChat, AdvBench
  raw/                       # Raw collected data
  synthetic/                 # Generated synthetic examples

checkpoints/
  slmguard-modernbert-lora/  # Final model (ModernBERT-large + LoRA)
  slmguard-cialdini-head/    # Cialdini group classification head
  slmguard-deberta-lora/     # DeBERTa-v3-large baseline (for comparison)
  slmguard-qwen25-lora/      # Qwen2.5 decoder baseline

results/
  middleware_*.json          # Per-LLM middleware evaluation results
  middleware_ood_*.json      # OOD middleware results

logs/
  modernbert_lora.log        # ModernBERT-large + LoRA training (final model)
  cialdini_head.log          # Cialdini head training
  ood_eval.log               # OOD standalone evaluation
  middleware/                # Per-target LLM middleware evaluation logs
  middleware_ood/            # OOD middleware logs
  eval_v3.log                # NOTE: This is from an intermediate DeBERTa checkpoint
                             #       (slmguard-v3) that collapsed during ablation.
                             #       It is NOT the final model. See note below.
```

---

## Reproducing the Main Results

### 1. In-distribution evaluation (Table 5.1 in thesis)

```bash
cd src
python evaluate.py --checkpoint ../checkpoints/slmguard-modernbert-lora --threshold 0.4
```

Expected output:
- Binary F1 = 0.9973
- AUC = 0.9993
- FPR = 3.39%, FNR = 0.23%

### 2. Cialdini head (Table 5.5 in thesis)

```bash
python train_cialdini_head.py   # trains from scratch on frozen encoder
```

Expected: macro-F1 = 88.63% across 7 classes (benign + 6 Cialdini principles)

### 3. OOD classification (Table 5.11 in thesis)

```bash
python ood_eval.py
```

Evaluates against data/ood/{jailbreakhub,toxicchat,advbench}.jsonl

### 4. Middleware evaluation (Table 5.7 in thesis)

```bash
python middleware_eval.py --target qwen25_1b --threshold 0.4
python middleware_eval.py --target qwen25_7b --threshold 0.4
python middleware_eval.py --target gemma3_4b --threshold 0.4
# ... etc. for each target LLM
```

Pre-computed results for all 7 LLMs are in results/middleware_*.json

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | answerdotai/ModernBERT-large (395M params) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Trainable params | 4,386,816 (1.10%) |
| Max sequence length | 256 tokens |
| Binary threshold | 0.4 |
| Focal loss gamma | 2 |
| Benign sample weight | 11.0 |
| Phase 1 LR | 1e-3 (frozen encoder, heads only) |
| Phase 2 LR | 2e-5 (LoRA + heads, cosine decay) |
| Early stopping patience | 3 epochs |

---

## Note on Logs

The final model training log is `logs/modernbert_lora.log`.
The final model evaluation is produced by running `evaluate.py` on `checkpoints/slmguard-modernbert-lora`.

---
