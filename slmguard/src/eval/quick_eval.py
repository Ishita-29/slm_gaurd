import json, torch, sys
from safetensors.torch import load_file
sys.path.insert(0, '/data/ishita_workspace/SLM-GAURD/slmguard/src')
from model.config import ALL_LABELS, ID2LABEL, DEPLOY_CKPT
from datasets import load_from_disk
from transformers import AutoTokenizer
from model.train import SLMGuardModel
from collections import Counter

CKPT = DEPLOY_CKPT
DATA = '/data/ishita_workspace/SLM-GAURD/slmguard/data/final/slmguard_dataset'

print("Loading model...")
cfg = json.load(open(f'{CKPT}/slmguard_config.json'))
MAX_LENGTH = cfg.get('max_length', 256)  # read from checkpoint, not hardcoded — must match training
model = SLMGuardModel(cfg['model_name'], model_key=cfg['model_key'], use_lora=cfg['use_lora']).cuda()
state = torch.load(f'{CKPT}/pytorch_model.bin', map_location='cuda')
load_result = model.load_state_dict(state, strict=False)
if load_result.missing_keys:
    print(f"WARNING: missing_keys when loading checkpoint: {load_result.missing_keys}")
if load_result.unexpected_keys:
    print(f"WARNING: unexpected_keys when loading checkpoint: {load_result.unexpected_keys}")
model.eval()

print("Loading tokenizer and data...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(CKPT)
ds = load_from_disk(DATA)['test'].select(range(240))

preds_mc, true_mc, preds_bin, true_bin = [], [], [], []
for ex in ds:
    enc = tokenizer(ex['text'], return_tensors='pt', max_length=MAX_LENGTH, truncation=True)
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
