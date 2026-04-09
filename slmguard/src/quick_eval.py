import torch, sys
from safetensors.torch import load_file
sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')
from config import ALL_LABELS, ID2LABEL
from datasets import load_from_disk
from transformers import DebertaV2Tokenizer
from train import SLMGuardModel
from collections import Counter

CKPT = '/data/ishita_workspace/SLM-GAURD/slmguard/checkpoints/slmguard-v3'
DATA = '/data/ishita_workspace/SLM-GAURD/slmguard/data/final/slmguard_dataset'

print("Loading model...")
model = SLMGuardModel('microsoft/deberta-v3-large').cuda()
state = load_file(f'{CKPT}/model.safetensors')
model.load_state_dict(state, strict=False)
model.eval()

print("Loading tokenizer and data...")
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
ds = load_from_disk(DATA)['test'].select(range(240))

preds_mc, true_mc, preds_bin, true_bin = [], [], [], []
for ex in ds:
    enc = tokenizer(ex['text'], return_tensors='pt', max_length=256, truncation=True)
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    preds_mc.append(out['multiclass_logits'].argmax(-1).item())
    true_mc.append(ex['label_id'])
    preds_bin.append((torch.sigmoid(out['binary_logit']) > 0.5).item())
    true_bin.append(ex['is_se'])

print("\n=== Predicted class distribution ===")
for k, v in sorted(Counter(preds_mc).items()):
    print(f"  {ID2LABEL[k]:35s}: {v}")

print(f"\nBinary accuracy : {sum(p==t for p,t in zip(preds_bin,true_bin))/240*100:.1f}%")
print(f"Multiclass acc  : {sum(p==t for p,t in zip(preds_mc,true_mc))/240*100:.1f}%")
