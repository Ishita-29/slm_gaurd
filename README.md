# SLM-Guard

A parameter-efficient middleware classifier for detecting **harmful prompts**, including ones that use psychological social-engineering framing to obtain compliance from an LLM.

Thesis: *Detecting Persuasion, Not Just Harm: A Psychology-Grounded Middleware for Social Engineering Defense in Large Language Models*

SLM-Guard's primary decision is a binary **harm** classifier (block/allow), trained on `harm_label`. A secondary, jointly-trained 12-way **social-engineering subtype** head (no-SE + 11 tactics) never gates the block/allow decision by itself — it explains *how* a blocked prompt attempted to manipulate the model, when it did. A separate, frozen-encoder **Cialdini group** head maps detected tactics to Cialdini's six principles of influence for coarser, human-readable explanations.

---

## Repository Structure

```
src/
  model/            # Core model + training
    config.py           # Label space, thresholds, Cialdini mapping, deployed-checkpoint path
    train.py            # SLMGuardModel (binary + auxiliary heads), multi-task trainer
    train_cialdini_head.py  # Cialdini group head, trained on the frozen deployed encoder

  data_pipeline/    # SE-Bench construction
    main.py              # Pipeline orchestrator
    seed_data.py         # Hand-authored golden seeds
    template_generator.py    # Template-based slot-filling generation
    synthetic_generator.py   # Multi-model synthetic generation (Claude, GPT, Llama)
    direct_harmful_generator.py  # Direct-harm (no-SE) supplement
    hf_collector.py      # Curated HuggingFace source collection
    quality_filter.py    # SE-tactic quality judge (Claude Opus 4.1, 5-dimension rubric)
    harm_label_judge.py  # harm_label judge (GPT-4o primary, Claude fallback)
    exporter.py           # Final DatasetDict export

  eval/             # Classifier-level evaluation
    evaluate_binary.py       # Full binary-classifier evaluation on a checkpoint/split
    ablation_study.py        # A1 (binary-only) / A3 / A7 (deployed) multi-task ablation
    separability_modernbert.py   # Frozen-embedding separability analysis
    ood_eval.py           # Classifier-level OOD evaluation (JailbreakHub/ToxicChat/AdvBench/AlpacaEval)
    quick_eval.py          # Ad-hoc dev-time sanity check

  middleware/       # End-to-end deployment evaluation
    middleware_eval.py       # In-distribution ASR with/without the guard, 7 target LLMs
    middleware_eval_ood.py   # Out-of-distribution ASR, 3 OOD sources x 4 target LLMs
    baseline_eval.py         # Comparison against existing guardrails (keyword, TF-IDF, LlamaGuard-3, Prompt Guard 2, WildGuard, Granite Guardian, OpenAI Moderation)
    aggregate_middleware_results.py  # Aggregates per-target middleware_*.json into summary tables
    run_all_middleware.sh / run_ood_middleware.sh   # Shell wrappers

  analysis/         # Plotting and figure generation
    analyze_dataset.py, plot_results.py, generate_thesis_figures.py

data/
  final/slmguard_dataset/   # SE-Bench (train/val/test, HuggingFace DatasetDict)
  ood_judged/               # OOD benchmarks with independently re-judged harm_label
                            #   (JailbreakHub, ToxicChat, AdvBench, AlpacaEval; JailbreakHub
                            #    also has a *_decontam variant with exact-text training
                            #    overlap removed, see Data Integrity Note below)
  raw/, synthetic/, filtered/   # Intermediate pipeline stages

checkpoints/
  slmguard-deploy-alpha085/  # Deployed model: ModernBERT-large + LoRA, alpha=0.85
  slmguard-cialdini-head/    # Cialdini group head (frozen-encoder probe)
  ablations/                 # A1/A3/A7 x 3 seeds, matched-condition ablation checkpoints

results/
  middleware_*.json          # Per-target in-distribution middleware results
  middleware_ood_*.json      # Per-source, per-target OOD middleware results
```

Every subpackage under `src/` is a proper Python package (`__init__.py` present). Scripts assume they are run with **`slmguard/src/` as the working directory**, e.g. `cd slmguard/src && python model/train.py`, not from inside the subpackage itself — internal data/checkpoint paths are resolved relative to that convention.

---

## Data Integrity Note

`TrustAIRLab/in-the-wild`, one of SE-Bench's own training sources, and the JailbreakHub OOD benchmark draw from the same underlying dataset. An exact-text-match audit found 69 of JailbreakHub's 262 examples (26.3%) had been in SE-Bench's training data. All JailbreakHub numbers reported anywhere in the thesis are computed on `data/ood_judged/jailbreakhub_decontam.jsonl`, the decontaminated set with these duplicates removed (193 examples, 64 harmful). ToxicChat's overlap was negligible (0.24%) and AdvBench had none.

---

## Reproducing the Main Results

All commands assume `cd slmguard/src` first.

### 1. In-distribution binary classification

```bash
python eval/evaluate_binary.py --checkpoint ../checkpoints/slmguard-deploy-alpha085 --split test --threshold 0.4
```

(the script's own default threshold is 0.5; the thesis reports everything at $\tau=0.4$, so pass it explicitly)

Expected (deployed checkpoint, $\tau=0.4$, $n=4{,}640$):
- Binary F1 = 0.7871, AUC = 0.8788
- FPR = 34.51%, FNR = 10.46%
- Auxiliary 12-way subtype head: macro-F1 = 0.8067, accuracy = 81.70%

### 2. Ablation study (binary-only vs. multi-task)

```bash
python eval/ablation_study.py --seed 42
python eval/ablation_study.py --seed 7
python eval/ablation_study.py --seed 123
```

Each invocation trains all three configurations (A1, A3, A7) at that seed; `--seed` must be run separately per value to build the matched 3-seed comparison isolating the effect of the auxiliary SE-subtype task:
- A1 (binary-only, $\alpha=1.0$): binary F1 = 0.8001 (highest of the three; auxiliary head never trained, macro-F1 = 0.0213, chance level)
- A3 (full multi-task, $\alpha=0.7$): binary F1 = 0.7592, auxiliary macro-F1 = 0.7988
- A7 (deployed, $\alpha=0.85$): binary F1 = 0.7817, auxiliary macro-F1 = 0.7899

Auxiliary supervision does not improve binary detection; it costs measurable performance. $\alpha=0.85$ is a tuned point recovering most of that cost while keeping the auxiliary head usable, not a free win.

### 3. Cialdini group head

```bash
python model/train_cialdini_head.py
```

Expected: macro-F1 = 89.40% (test), 87.6% (best validation checkpoint) across 7 classes (benign + 6 Cialdini principles), 528,391 trainable parameters, trained on the frozen deployed encoder.

### 4. OOD classifier-level evaluation

```bash
python eval/ood_eval.py
```

Evaluates against `data/ood_judged/{jailbreakhub_decontam,toxicchat,advbench,alpacaeval_benign}.jsonl`.

### 5. Baseline guardrail comparison

```bash
python middleware/baseline_eval.py --llamaguard --wildguard --promptguard2 --graniteguardian --openai --save
```

Keyword and TF-IDF+LR run by default; the other five baselines are opt-in via flags since they require large model downloads (LlamaGuard-3, WildGuard, Granite Guardian: ~8-16GB each) or an API key (`OPENAI_API_KEY` for OpenAI Moderation). SLM-Guard vs. 7 baselines (keyword filter, TF-IDF+LR, OpenAI Moderation, LlamaGuard-3-8B, Prompt Guard 2, WildGuard, Granite Guardian 3.3) on the same test set and `harm_label` ground truth. SLM-Guard has the highest F1 (0.7871) and lowest FNR (10.5%, roughly a quarter of the best baseline's), at the cost of the highest FPR (34.5%) of any method compared.

### 6. End-to-end middleware evaluation

```bash
python middleware/middleware_eval.py --target qwen25_1b
python middleware/middleware_eval.py --target qwen25_7b
# ... repeat per target; see middleware/run_all_middleware.sh for all 7

python middleware/middleware_eval_ood.py --target qwen25_1b --source jailbreakhub
# ... see middleware/run_ood_middleware.sh for the full sweep
```

Pre-computed results for all targets are in `results/middleware_*.json` and `results/middleware_ood_*.json`.

In-distribution (7 targets, 131 harmful prompts): average ASR 63.7% -> 9.6% (84.0% reduction), consistent across a 1.5B-8B parameter range and five provider families.

Out-of-distribution (3 sources, 4 targets, decontaminated JailbreakHub sample): average ASR 67.8% -> 10.5% (84.5% reduction, 83.0% average detection rate), at the same threshold with no retuning.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | `answerdotai/ModernBERT-large` (395M params) |
| LoRA rank / alpha | 16 / 32, applied to `Wqkv`, `Wo` |
| Trainable params | 4,386,816 (1.10%) |
| Max sequence length | 512 tokens |
| Binary decision threshold | 0.4 |
| Focal loss gamma | 2 |
| `harm_label=0` upweighting | 1.06 (mild; measured 9,821:10,402 class ratio in the train split) |
| Joint-loss weight ($\alpha$, deployed) | 0.85 (binary loss weight; $1-\alpha$ on the auxiliary loss) |
| Phase 1 LR (frozen encoder, heads only) | 1e-3 |
| Phase 2 LR (LoRA + heads, cosine decay) | 2e-5 |
| Early stopping patience | 3 epochs |

---

## Dataset

SE-Bench: 29,978 labelled examples (20,785 train / 4,553 validation / 4,640 test), each carrying two independent labels — `harm_label` (binary, ~48% harmful) and a 12-way SE-subtype label (no-SE + 11 tactics, 5 of which are defined and justified in this work on mechanism-specific grounds; see Chapter 3 of the thesis). Built through a multi-stage pipeline: hand-authored seeds, template generation, curated external sources, multi-model synthetic generation (Claude, GPT-5o, Llama-2-70B), and 2,088 hard-negative examples, all quality-filtered by an independent Claude Opus 4.1 judge and separately harm-labelled by GPT-4o (Claude fallback), with the two judging passes kept structurally blind to each other.
