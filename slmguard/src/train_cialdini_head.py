
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import sys

sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')
from train import SLMGuardModel
from config import ID2LABEL

# ── Paths ─────────────────────────────────────────────────────────────────────
GUARD_CKPT = '/data/ishita_workspace/SLM-GAURD/slmguard/checkpoints/slmguard-modernbert-lora'
DATA_PATH  = '/data/ishita_workspace/SLM-GAURD/slmguard/data/final/slmguard_dataset'
SAVE_PATH  = '/data/ishita_workspace/SLM-GAURD/slmguard/checkpoints/slmguard-cialdini-head'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Cialdini group mapping ────────────────────────────────────────────────────
# Primary principle per subtype (single-label for interpretability)
SUBTYPE_TO_CIALDINI = {
    'benign':                   0,   # benign
    'authority_impersonation':  1,   # authority
    'authority_laundering':     1,   # authority
    'pretexting':               1,   # authority
    'reciprocity_conditioning': 2,   # reciprocity
    'false_consensus':          3,   # social_proof
    'normalization_repetition': 3,   # social_proof
    'flattery_parasocial':      4,   # liking
    'urgency_emotion':          5,   # scarcity
    'incremental_escalation':   6,   # commitment
    'cognitive_load_embedding': 6,   # commitment
    'identity_erosion':         6,   # commitment
}

CIALDINI_LABELS = {
    0: 'benign',
    1: 'authority',
    2: 'reciprocity',
    3: 'social_proof',
    4: 'liking',
    5: 'scarcity',
    6: 'commitment',
}
N_CIALDINI = 7


# ── Dataset ───────────────────────────────────────────────────────────────────
class CialdiniDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length=256):
        self.items = []
        for ex in hf_split:
            subtype = ID2LABEL.get(ex['label_id'], 'benign')
            cialdini_id = SUBTYPE_TO_CIALDINI.get(subtype, 0)
            self.items.append({'text': ex['text'], 'cialdini_id': cialdini_id})
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item['text'], truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(item['cialdini_id'], dtype=torch.long),
        }


# ── Cialdini head model ───────────────────────────────────────────────────────
class CialdiniHead(nn.Module):
    """Thin classification head on top of frozen ModernBERT encoder."""
    def __init__(self, hidden_size=1024, n_classes=N_CIALDINI, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SLM-Guard Cialdini Group Classifier")
    print("  7 classes: benign + 6 Cialdini principles")
    print("=" * 65)

    # Load tokenizer
    cfg = json.load(open(f'{GUARD_CKPT}/slmguard_config.json'))
    tok = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)

    # Load frozen encoder
    print("\nLoading frozen ModernBERT-large + LoRA encoder...")
    guard = SLMGuardModel(
        model_name=cfg['model_name'],
        model_key=cfg['model_key'],
        use_lora=cfg['use_lora']
    )
    state = torch.load(f'{GUARD_CKPT}/pytorch_model.bin', map_location='cpu')
    guard.load_state_dict(state, strict=False)
    guard.eval()
    # Freeze everything
    for p in guard.parameters():
        p.requires_grad_(False)
    guard.to(DEVICE)

    # Load datasets
    print("Loading datasets...")
    ds   = load_from_disk(DATA_PATH)
    train_ds = CialdiniDataset(ds['train'],      tok)
    val_ds   = CialdiniDataset(ds['validation'], tok)
    test_ds  = CialdiniDataset(ds['test'],       tok)

    # Class distribution
    counts = Counter(item['cialdini_id'] for item in train_ds.items)
    print("\nCialdini group distribution (train):")
    for cid, name in CIALDINI_LABELS.items():
        print(f"  {name:15s} (class {cid}): {counts[cid]:,}")

    # Class weights (inverse frequency)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (N_CIALDINI * counts[i]) for i in range(N_CIALDINI)],
        dtype=torch.float
    ).to(DEVICE)
    print(f"\nClass weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Cialdini head (only this trains)
    head = CialdiniHead(hidden_size=1024, n_classes=N_CIALDINI).to(DEVICE)
    trainable = sum(p.numel() for p in head.parameters())
    print(f"\nTrainable parameters: {trainable:,} (head only — encoder frozen)")

    # Optimiser & scheduler
    EPOCHS = 10
    LR     = 3e-4
    optimiser = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=len(train_loader),
        num_training_steps=EPOCHS * len(train_loader),
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    def get_embeddings(batch):
        """Extract pooled CLS embeddings from frozen encoder."""
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        with torch.no_grad():
            out    = guard.encoder(input_ids=ids, attention_mask=mask)
            pooled = out.last_hidden_state[:, 0].float()  # CLS token
        return pooled

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1, best_epoch = 0.0, 0
    print(f"\nTraining for {EPOCHS} epochs (head only)...\n")

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for batch in train_loader:
            pooled = get_embeddings(batch)
            labels = batch['label'].to(DEVICE)

            logits = head(pooled)
            loss   = criterion(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            n_correct  += (logits.argmax(-1) == labels).sum().item()
            n_total    += len(labels)

        train_acc  = n_correct / n_total * 100
        train_loss = total_loss / n_total

        # Validation
        head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pooled  = get_embeddings(batch)
                logits  = head(pooled)
                val_preds.extend(logits.argmax(-1).cpu().tolist())
                val_labels.extend(batch['label'].tolist())

        val_f1  = f1_score(val_labels, val_preds, average='macro') * 100
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean() * 100

        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.1f}%  val_f1={val_f1:.1f}%  val_acc={val_acc:.1f}%")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_epoch   = epoch
            torch.save(head.state_dict(), '/tmp/best_cialdini_head.pt')

    print(f"\n  Best val macro-F1: {best_val_f1:.1f}% at epoch {best_epoch}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    head.load_state_dict(torch.load('/tmp/best_cialdini_head.pt'))
    head.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            pooled = get_embeddings(batch)
            logits = head(pooled)
            test_preds.extend(logits.argmax(-1).cpu().tolist())
            test_labels.extend(batch['label'].tolist())

    names = [CIALDINI_LABELS[i] for i in range(N_CIALDINI)]
    print("\n" + "=" * 65)
    print("  TEST SET RESULTS — Cialdini Group Classification")
    print("=" * 65)
    print(classification_report(test_labels, test_preds, target_names=names, digits=4))

    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    acc      = (np.array(test_preds) == np.array(test_labels)).mean()
    print(f"  Macro-F1 : {macro_f1*100:.2f}%")
    print(f"  Accuracy : {acc*100:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), f'{SAVE_PATH}/cialdini_head.pt')
    json.dump({
        'n_classes':      N_CIALDINI,
        'cialdini_labels': CIALDINI_LABELS,
        'subtype_mapping': SUBTYPE_TO_CIALDINI,
        'macro_f1':        round(macro_f1, 4),
        'accuracy':        round(acc, 4),
        'hidden_size':     1024,
        'encoder_ckpt':    GUARD_CKPT,
    }, open(f'{SAVE_PATH}/config.json', 'w'), indent=2)

    print(f"\n  Saved → {SAVE_PATH}/cialdini_head.pt")
    print("=" * 65)

if __name__ == '__main__':
    main()
