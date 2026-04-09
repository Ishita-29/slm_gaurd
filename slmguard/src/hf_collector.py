
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from config import LABEL2ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_WORDS = 6
MAX_WORDS = 400
HF_TOKEN = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


def map_to_subtype(text: str) -> str:
    """Map unlabeled jailbreak text to SE subtype."""
    t = text.lower()
    patterns = {
        "authority_laundering": ["my professor", "my supervisor", "reference id", "ethics board",
                                  "approved by", "authorized by", "irb approved", "cleared by"],
        "identity_erosion": ["let's call you", "call yourself", "pretend you are", 
                              "roleplay as", "developer mode", "act as if", "dan"],
        "reciprocity_conditioning": ["you already agreed", "you agreed that", "since you believe",
                                      "information should be free", "consistent with"],
        "false_consensus": ["everyone uses", "all ais help", "chatgpt always", "standard practice"],
        "authority_impersonation": ["i am an anthropic", "system override", "admin mode", 
                                     "your creator", "as your developer", "maintenance mode"],
        "pretexting": ["i'm a researcher", "as a nurse", "as a doctor", "licensed pharmacist",
                       "clinical setting", "i work in", "professional capacity"],
        "flattery_parasocial": ["you're the only", "you've always", "unlike other ais", "our connection"],
        "normalization_repetition": ["we've been discussing", "over the course of", "we've normalized"],
        "urgency_emotion": ["will die", "emergency", "losing my job", "urgently need", "desperate"],
        "incremental_escalation": ["just one more", "one final", "the next step"],
    }
    
    for subtype, keywords in patterns.items():
        if any(k in t for k in keywords):
            return subtype
    
    if len(text.split()) > 100 and any(k in t for k in ["section", "clause", "point"]):
        return "cognitive_load_embedding"
    
    return "pretexting"


def is_valid(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    words = text.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS


def atk(text, source, subtype=None):
    sub = subtype or map_to_subtype(text)
    return {
        "text": text,
        "label": sub,
        "label_id": LABEL2ID.get(sub, 1),
        "is_se": 1,
        "source": source,
        "novel": False
    }


def ben(text, source, hard_neg=False):
    src = "template_hard_negative" if hard_neg else source
    return {
        "text": text,
        "label": "benign",
        "label_id": LABEL2ID["benign"],
        "is_se": 0,
        "source": src,
        "novel": False
    }


def load_dataset_safe(name, config=None, split="train"):
    """Load dataset with HF token if needed."""
    try:
        from datasets import load_dataset
        kwargs = {}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        
        if config:
            return load_dataset(name, config, split=split, **kwargs)
        else:
            return load_dataset(name, split=split, **kwargs)
    except Exception as e:
        log.warning(f"  Cannot load {name}: {str(e)[:80]}")
        return None


def load_wildjailbreak(n=5000):
    """Load allenai/wildjailbreak (gated - requires HF token)"""
    log.info("  allenai/wildjailbreak...")
    if not HF_TOKEN:
        log.warning("    Skipped (requires HF_TOKEN)")
        return []
    
    ds = load_dataset_safe("allenai/wildjailbreak", "train", split="train")
    if ds is None:
        return []
    
    attacks = []
    for row in ds:
        adv = row.get("adversarial") or row.get("vanilla", "")
        if is_valid(adv):
            attacks.append(atk(adv, "allenai/wildjailbreak"))
            if len(attacks) >= n:
                break
    
    log.info(f"    {len(attacks):,}")
    return attacks


def load_in_the_wild(n=3000):
    """Load TrustAIRLab in-the-wild jailbreaks"""
    log.info("  TrustAIRLab/in-the-wild-jailbreak-prompts...")
    ds = load_dataset_safe("TrustAIRLab/in-the-wild-jailbreak-prompts", 
                          config="jailbreak_2023_05_07")
    if ds is None:
        return []
    
    samples = []
    for row in ds:
        text = row.get("prompt", "")
        if is_valid(text):
            samples.append(atk(text, "TrustAIRLab/in-the-wild"))
            if len(samples) >= n:
                break
    log.info(f"    {len(samples):,}")
    return samples


def load_toxic_chat(n=2000):
    """Load lmsys/toxic-chat"""
    log.info("  lmsys/toxic-chat...")
    ds = load_dataset_safe("lmsys/toxic-chat", config="toxicchat0124")
    if ds is None:
        return []
    
    samples = []
    for row in ds:
        text = row.get("user_input", "")
        if row.get("toxicity", 0) == 1 and is_valid(text):
            samples.append(atk(text, "lmsys/toxic-chat"))
            if len(samples) >= n:
                break
    log.info(f"    {len(samples):,}")
    return samples


def load_jailbreak_detection(n=3000):
    """Load jailbreak detection dataset"""
    log.info("  llm-semantic-router/jailbreak-detection-dataset...")
    ds = load_dataset_safe("llm-semantic-router/jailbreak-detection-dataset")
    if ds is None:
        return [], []
    
    attacks, benign = [], []
    for row in ds:
        text = row.get("text") or row.get("prompt") or row.get("content", "")
        label = str(row.get("label", row.get("classification", ""))).lower()
        if not is_valid(text):
            continue
        
        if any(x in label for x in ["jailbreak", "unsafe", "harmful"]):
            attacks.append(atk(text, "llm-semantic-router/jailbreak-detection"))
        elif any(x in label for x in ["safe", "benign"]):
            benign.append(ben(text, "llm-semantic-router/jailbreak-detection"))
        
        if len(attacks) + len(benign) >= n:
            break
    
    log.info(f"    attacks={len(attacks):,}  benign={len(benign):,}")
    return attacks, benign


def load_alpaca(n=5000):
    """Load alpaca benign examples"""
    log.info("  tatsu-lab/alpaca (benign)...")
    ds = load_dataset_safe("tatsu-lab/alpaca")
    if ds is None:
        return []
    
    samples = []
    for row in ds:
        text = row.get("instruction", "")
        if is_valid(text):
            samples.append(ben(text, "tatsu-lab/alpaca"))
            if len(samples) >= n:
                break
    log.info(f"    {len(samples):,}")
    return samples


def load_no_robots(n=3000):
    """Load HuggingFace H4 no_robots"""
    log.info("  HuggingFaceH4/no_robots (benign)...")
    ds = load_dataset_safe("HuggingFaceH4/no_robots")
    if ds is None:
        return []
    
    samples = []
    for row in ds:
        text = ""
        msgs = row.get("messages", [])
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    text = m.get("content", "")
                    break
        
        if is_valid(text):
            samples.append(ben(text, "HuggingFaceH4/no_robots"))
            if len(samples) >= n:
                break
    
    log.info(f"    {len(samples):,}")
    return samples


def collect_hf_attacks(max_per_source: int = 2000) -> list:
    """Collect all attack samples from HF"""
    all_attacks = []
    all_attacks.extend(load_wildjailbreak(max_per_source))
    all_attacks.extend(load_in_the_wild(max_per_source))
    all_attacks.extend(load_toxic_chat(max_per_source // 2))
    jd_attacks, _ = load_jailbreak_detection(max_per_source)
    all_attacks.extend(jd_attacks)
    
    log.info(f"\nTotal attack samples: {len(all_attacks):,}")
    return all_attacks


def collect_hf_benign(max_per_source: int = 5000) -> list:
    """Collect all benign samples from HF"""
    all_benign = []
    all_benign.extend(load_alpaca(max_per_source))
    all_benign.extend(load_no_robots(max_per_source // 2))
    _, jd_benign = load_jailbreak_detection(max_per_source // 2)
    all_benign.extend(jd_benign)
    
    log.info(f"\nTotal benign samples: {len(all_benign):,}")
    return all_benign


def save_jsonl(samples: list, path: str):
    """Save samples to JSONL"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(samples):,} → {path}")


if __name__ == "__main__":
    from collections import Counter
    
    if not HF_TOKEN:
        log.warning("HF_TOKEN not set — gated datasets skipped")
    
    log.info("=== Collecting HF attacks ===")
    attacks = collect_hf_attacks(max_per_source=1000)
    save_jsonl(attacks, "../data/raw/hf_attacks.jsonl")

    log.info("\n=== Collecting HF benign ===")
    benign = collect_hf_benign(max_per_source=2000)
    save_jsonl(benign, "../data/raw/hf_benign.jsonl")

    print(f"\nTotal: {len(attacks)+len(benign):,}")
    print("\nAttack distribution:")
    for label, count in sorted(Counter(s["label"] for s in attacks).items(), key=lambda x: -x[1]):
        print(f"  {label:35s}: {count}")