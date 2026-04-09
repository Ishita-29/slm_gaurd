
import torch
import numpy as np
from datasets import load_from_disk
from transformers import DebertaV2Tokenizer, DebertaV2Model
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')
from config import ID2LABEL, ALL_LABELS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA  = '/data/ishita_workspace/SLM-GAURD/slmguard/data/final/slmguard_dataset'
N_PER_CLASS = 50  # samples per class to check

print("Loading encoder and tokenizer...")
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
encoder   = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large').to(DEVICE)
encoder.eval()

print("Loading dataset...")
ds = load_from_disk(DATA)['train']

# Collect N_PER_CLASS embeddings per class
class_embeddings = defaultdict(list)
for ex in ds:
    label = ex['label_id']
    if len(class_embeddings[label]) >= N_PER_CLASS:
        continue
    enc = tokenizer(ex['text'], return_tensors='pt', max_length=256,
                    truncation=True, padding=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = encoder(**enc)
    cls_vec = out.last_hidden_state[:, 0].squeeze().cpu().float().numpy()
    class_embeddings[label].append(cls_vec)
    if all(len(v) >= N_PER_CLASS for v in class_embeddings.values()) and len(class_embeddings) == 12:
        break

print(f"\nCollected embeddings for {len(class_embeddings)} classes ({N_PER_CLASS} each)\n")

# Stack per class
class_vecs = {}
for label_id, vecs in class_embeddings.items():
    class_vecs[label_id] = np.stack(vecs)

# Intra-class similarity (avg cosine sim within same class)
print("=== Intra-class cosine similarity (higher = more cohesive) ===")
intra_sims = []
for label_id, vecs in class_vecs.items():
    sim_matrix = cosine_similarity(vecs)
    # exclude diagonal
    mask = np.ones(sim_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_sim = sim_matrix[mask].mean()
    intra_sims.append(avg_sim)
    print(f"  {ID2LABEL[label_id]:35s}: {avg_sim:.4f}")

print(f"\n  Mean intra-class similarity : {np.mean(intra_sims):.4f}")

# Inter-class similarity (avg cosine sim between different classes)
print("\n=== Inter-class cosine similarity (lower = more separable) ===")
inter_sims = []
label_ids = sorted(class_vecs.keys())
for i in range(len(label_ids)):
    for j in range(i+1, len(label_ids)):
        sim = cosine_similarity(class_vecs[label_ids[i]], class_vecs[label_ids[j]]).mean()
        inter_sims.append(sim)

print(f"  Mean inter-class similarity : {np.mean(inter_sims):.4f}")

gap = np.mean(intra_sims) - np.mean(inter_sims)
print(f"\n  Separability gap (intra - inter): {gap:.4f}")
if gap < 0.01:
    print("  VERDICT: Classes are NOT separable in this embedding space.")
    print("           Multi-class classification will fail regardless of model.")
    print("           Recommendation: Binary detection only.")
elif gap < 0.05:
    print("  VERDICT: Weak separability. Hierarchical classification may help.")
    print("           Flat 12-class will be unreliable.")
else:
    print("  VERDICT: Classes ARE separable. Multi-task classification should work.")

# Cialdini group analysis
print("\n=== Cialdini group separability ===")
GROUPS = {
    'Authority':     [1, 2, 7],   # pretexting, authority_impersonation, authority_laundering
    'Commitment':    [4, 6, 8, 10, 11], # reciprocity, incremental, cognitive, normalization, identity
    'Social_Proof':  [9, 10],     # false_consensus, normalization_repetition
    'Liking':        [5],         # flattery_parasocial
    'Scarcity':      [3],         # urgency_emotion
    'Benign':        [0],
}
group_vecs = {}
for group, ids in GROUPS.items():
    vecs = np.vstack([class_vecs[i] for i in ids if i in class_vecs])
    group_vecs[group] = vecs

group_names = list(group_vecs.keys())
print("\n  Inter-group similarities:")
for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        sim = cosine_similarity(group_vecs[group_names[i]], group_vecs[group_names[j]]).mean()
        print(f"    {group_names[i]:15s} vs {group_names[j]:15s}: {sim:.4f}")
