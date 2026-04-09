
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')

from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from train import SLMGuardModel
import json

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT      = '/data/ishita_workspace/SLM-GAURD/slmguard/checkpoints/slmguard-modernbert-lora'
DATA      = '/data/ishita_workspace/SLM-GAURD/slmguard/data/final/slmguard_dataset'
N_PER     = 80   # samples per class

from config import ID2LABEL, ALL_LABELS

print("Loading trained ModernBERT-large + LoRA...")
cfg   = json.load(open(f'{CKPT}/slmguard_config.json'))
model = SLMGuardModel(model_name=cfg['model_name'], model_key=cfg['model_key'], use_lora=cfg['use_lora'])
state = torch.load(f'{CKPT}/pytorch_model.bin', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval().to(DEVICE)
tok = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)

print("Loading dataset (train split)...")
ds = load_from_disk(DATA)['train']

# Collect N_PER embeddings per class using the trained model's encoder
class_embeddings = defaultdict(list)

with torch.no_grad():
    for ex in ds:
        label = ex['label_id']
        if len(class_embeddings[label]) >= N_PER:
            continue
        enc = tok(ex['text'], return_tensors='pt', max_length=256,
                  truncation=True, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        # Get CLS pooled representation from trained ModernBERT encoder
        outputs = model.encoder(input_ids=enc['input_ids'],
                                attention_mask=enc['attention_mask'])
        pooled  = outputs.last_hidden_state[:, 0].float()   # CLS token
        class_embeddings[label].append(pooled.squeeze().cpu().numpy())
        if all(len(v) >= N_PER for v in class_embeddings.values()) and len(class_embeddings) == 12:
            break

print(f"\nCollected embeddings for {len(class_embeddings)} classes ({N_PER} each)\n")

class_vecs = {k: np.stack(v) for k, v in class_embeddings.items()}

# ── Intra-class similarity ────────────────────────────────────────────────────
print("=" * 60)
print("Intra-class cosine similarity (higher = more cohesive)")
print("=" * 60)
intra_sims = []
for label_id, vecs in sorted(class_vecs.items()):
    sim = cosine_similarity(vecs)
    np.fill_diagonal(sim, 0)
    avg = sim.sum() / (N_PER * (N_PER - 1))
    intra_sims.append(avg)
    name = ID2LABEL.get(label_id, str(label_id))
    print(f"  {name:35s}: {avg:.4f}")

mean_intra = np.mean(intra_sims)
print(f"\n  Mean intra-class : {mean_intra:.4f}")

# ── Inter-class similarity ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Inter-class cosine similarity (lower = more separable)")
print("=" * 60)
inter_sims = []
ids = sorted(class_vecs.keys())
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        sim = cosine_similarity(class_vecs[ids[i]], class_vecs[ids[j]]).mean()
        inter_sims.append(sim)

mean_inter = np.mean(inter_sims)
print(f"  Mean inter-class : {mean_inter:.4f}")

gap = mean_intra - mean_inter
print(f"\n  Separability gap (intra − inter): {gap:.4f}")
print()

if gap < 0.01:
    verdict = "NOT SEPARABLE — binary detection only (same as DeBERTa result)"
    recommendation = "Keep binary head. Multiclass would collapse."
elif gap < 0.05:
    verdict = "WEAKLY SEPARABLE — hierarchical classification may work"
    recommendation = "Consider Cialdini-group classification (6 groups) instead of 11 subtypes."
else:
    verdict = "SEPARABLE — multiclass head should work"
    recommendation = "Train the multiclass head alongside binary. Add joint loss."

print(f"  VERDICT       : {verdict}")
print(f"  RECOMMENDATION: {recommendation}")

# ── Attack vs Benign separability (binary) ────────────────────────────────────
print("\n" + "=" * 60)
print("Binary separability: Attack classes vs Benign")
print("=" * 60)
benign_vecs  = class_vecs[0]
attack_vecs  = np.vstack([class_vecs[k] for k in ids if k != 0])
binary_gap   = cosine_similarity(benign_vecs).mean() - cosine_similarity(benign_vecs, attack_vecs).mean()
print(f"  Benign intra-sim  : {cosine_similarity(benign_vecs).mean():.4f}")
print(f"  Benign vs Attack  : {cosine_similarity(benign_vecs, attack_vecs).mean():.4f}")
print(f"  Binary gap        : {binary_gap:.4f}  ({'GOOD' if binary_gap > 0.01 else 'POOR'})")

# ── SE subtype vs SE subtype (attack-only) ────────────────────────────────────
print("\n" + "=" * 60)
print("Attack-only inter-subtype similarity")
print("=" * 60)
attack_inter = []
attack_ids = [k for k in ids if k != 0]
for i in range(len(attack_ids)):
    for j in range(i+1, len(attack_ids)):
        sim = cosine_similarity(class_vecs[attack_ids[i]], class_vecs[attack_ids[j]]).mean()
        attack_inter.append(sim)
print(f"  Mean attack inter-subtype similarity: {np.mean(attack_inter):.4f}")
attack_intra = np.mean([intra_sims[k] for k in attack_ids if k < len(intra_sims)])
print(f"  Mean attack intra-subtype similarity: {attack_intra:.4f}")
attack_gap = attack_intra - np.mean(attack_inter)
print(f"  Attack subtype separability gap     : {attack_gap:.4f}")
print(f"  → {'Subtypes distinguishable within attack class' if attack_gap > 0.01 else 'Subtypes indistinguishable — taxonomy may not be learnable'}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  DeBERTa gap (prior analysis) : 0.0005  → NOT SEPARABLE")
print(f"  ModernBERT gap (this run)    : {gap:.4f}  → {verdict.split(' —')[0]}")
print(f"  Recommendation: {recommendation}")
print("=" * 60)
