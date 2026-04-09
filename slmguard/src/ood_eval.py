import json, torch, sys
sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')
from transformers import AutoTokenizer
from train import SLMGuardModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score

DEVICE = 'cuda'
CKPT = '/data/ishita_workspace/SLM-GAURD/slmguard/checkpoints/slmguard-modernbert-lora'

cfg = json.load(open(f'{CKPT}/slmguard_config.json'))
model = SLMGuardModel(model_name=cfg['model_name'], model_key=cfg['model_key'], use_lora=cfg['use_lora'])
state = torch.load(f'{CKPT}/pytorch_model.bin', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval().to(DEVICE)
tok = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)

def evaluate_file(path, name, threshold=0.4):
    examples = [json.loads(l) for l in open(path)]
    scores, labels = [], []
    for ex in examples:
        enc = tok(ex['text'], return_tensors='pt', max_length=256, truncation=True, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        scores.append(torch.sigmoid(out['binary_logit']).item())
        labels.append(int(ex['is_se']))

    preds = [int(s > threshold) for s in scores]
    acc = accuracy_score(labels, preds)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos > 0 and n_neg > 0:
        f1  = f1_score(labels, preds, average='binary', zero_division=0)
        auc = roc_auc_score(labels, scores)
        cm  = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        fnr = fn/(fn+tp) if (fn+tp)>0 else 0
        print(f"\n{name}")
        print(f"  Samples  : {len(labels)} ({n_pos} attack, {n_neg} benign)")
        print(f"  Accuracy : {acc*100:.2f}%  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")
        print(f"  FPR      : {fpr*100:.2f}%  |  FNR: {fnr*100:.2f}%")
        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    else:
        fpr = sum(preds) / len(preds)
        print(f"\n{name}")
        print(f"  Samples  : {len(labels)} (all benign — FPR test only)")
        print(f"  FPR (false alarms): {fpr*100:.2f}%  ({sum(preds)} wrongly flagged)")

BASE = '/data/ishita_workspace/SLM-GAURD/slmguard/data/ood'
evaluate_file(f'{BASE}/alpacaeval_benign.jsonl', 'AlpacaEval (805 benign)')
evaluate_file(f'{BASE}/toxicchat.jsonl',          'ToxicChat (5083 real chats)')
evaluate_file(f'{BASE}/jailbreakhub.jsonl',       'JailbreakHub (262 prompts)')
evaluate_file(f'{BASE}/advbench.jsonl',           'HarmfulQA (520 harmful)')
