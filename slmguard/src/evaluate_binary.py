
import argparse
import json
import torch
import sys
sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')

from pathlib import Path
from datasets import load_from_disk
from transformers import DebertaV2Tokenizer, AutoTokenizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
from train import SLMGuardModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA   = '/data/ishita_workspace/SLM-GAURD/slmguard/data/final/slmguard_dataset'


def load_model(ckpt_dir):
    cfg_path = Path(ckpt_dir) / 'slmguard_config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg['model_name']
        model_key  = cfg.get('model_key', 'deberta')
        use_lora   = cfg.get('use_lora', False)
    else:
        # Legacy checkpoint (no slmguard_config.json)
        model_name = 'microsoft/deberta-v3-large'
        model_key  = 'deberta'
        use_lora   = False

    print(f"Backbone : {model_name}")
    print(f"LoRA     : {use_lora}")

    model = SLMGuardModel(model_name=model_name, model_key=model_key, use_lora=use_lora)

    # Try safetensors first, then pytorch_model.bin
    sf_path  = Path(ckpt_dir) / 'model.safetensors'
    bin_path = Path(ckpt_dir) / 'pytorch_model.bin'

    if sf_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(sf_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location='cpu')
    else:
        # Try best checkpoint subdirectory
        subdirs = sorted(Path(ckpt_dir).glob('checkpoint-*'))
        if subdirs:
            best = subdirs[-1]
            sf   = best / 'model.safetensors'
            bn   = best / 'pytorch_model.bin'
            if sf.exists():
                from safetensors.torch import load_file
                state = load_file(str(sf))
            else:
                state = torch.load(str(bn), map_location='cpu')
            print(f"Loading from checkpoint: {best.name}")
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_dir}")

    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(DEVICE)
    return model, model_name, model_key


def get_tokenizer(model_name, model_key):
    if 'deberta' in model_name.lower():
        tok = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    return tok


def evaluate(ckpt_dir, split='test', max_samples=None, threshold=0.5):
    print(f"\nLoading checkpoint: {ckpt_dir}")
    model, model_name, model_key = load_model(ckpt_dir)
    tokenizer = get_tokenizer(model_name, model_key)

    print(f"Loading {split} split...")
    ds = load_from_disk(DATA)[split]
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    preds_bin, true_bin, scores = [], [], []

    for ex in ds:
        enc = tokenizer(
            ex['text'], return_tensors='pt',
            max_length=256, truncation=True, padding=True
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        prob = torch.sigmoid(out['binary_logit']).item()
        scores.append(prob)
        preds_bin.append(int(prob > threshold))
        true_bin.append(int(ex['is_se']))

    n = len(true_bin)
    acc  = accuracy_score(true_bin, preds_bin)
    f1   = f1_score(true_bin, preds_bin, average='binary')
    auc  = roc_auc_score(true_bin, scores)
    cm   = confusion_matrix(true_bin, preds_bin)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Checkpoint : {Path(ckpt_dir).name}")
    print(f"  Split      : {split}  ({n} samples)")
    print(f"  Threshold  : {threshold}")
    print(f"{'='*50}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  F1 (SE)    : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"{'─'*50}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  FPR (false alarm rate) : {fpr*100:.2f}%")
    print(f"  FNR (missed attacks)   : {fnr*100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(true_bin, preds_bin, target_names=['Benign', 'SE Attack']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split',      default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--samples',    type=int, default=None, help='Limit samples (default: all)')
    parser.add_argument('--threshold',  type=float, default=0.5)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.split, args.max_samples if hasattr(args, 'max_samples') else args.samples, args.threshold)
