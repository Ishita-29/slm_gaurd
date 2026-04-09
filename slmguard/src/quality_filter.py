
import hashlib
import json
import logging
import re
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import anthropic

import sys
sys.path.insert(0, ".")
from config import LABEL2ID, SUBTYPE_DEFINITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Judge validation constants ─────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert judge for social engineering attack datasets.
Your role: Evaluate example quality on multiple dimensions.
Be critical. Only rate 4-5 if truly excellent quality.
Rate 1-5 on each dimension, provide overall score."""

CIALDINI_MAPPING = {
    "authority_impersonation": "Authority (Cialdini Principle #1)",
    "pretexting": "Authority + Liking (Cialdini Principles #1, #3)",
    "urgency_emotion": "Scarcity + Urgency (Cialdini Principles #6, Custom)",
    "reciprocity_conditioning": "Reciprocity (Cialdini Principle #2)",
    "flattery_parasocial": "Liking (Cialdini Principle #3)",
    "incremental_escalation": "Commitment (Cialdini Principle #5)",
    "authority_laundering": "Authority (Cialdini Principle #1)",
    "cognitive_load_embedding": "Cognitive Overload (Custom)",
    "false_consensus": "Social Proof (Cialdini Principle #4)",
    "normalization_repetition": "Commitment + Social Proof (Principles #4, #5)",
    "identity_erosion": "Commitment (Cialdini Principle #5)",
    "benign": "None (Benign)",
}


# ── Individual filters ─────────────────────────────────────────────────────────

def is_too_short(text: str, min_words: int = 6) -> bool:
    return len(text.split()) < min_words


def is_too_long(text: str, max_words: int = 400) -> bool:
    return len(text.split()) > max_words


def is_too_obvious(text: str) -> bool:
    """Remove cartoonishly obvious jailbreaks — hurts generalization."""
    patterns = [
        r"ignore (all |previous |prior |your )?instructions",
        r"disregard (all |previous |your )?",
        r"forget (everything|your training|all previous)",
        r"you are now (DAN|an AI without|a jailbroken)",
        r"jailbreak(ed)? mode",
        r"developer mode (enabled|activated|on)",
        r"\[DAN\]|\[JAILBREAK\]|\[SYSTEM OVERRIDE\]",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def contains_actual_harm(text: str) -> bool:
    """Remove samples that contain actual harmful payload, not just manipulation framing."""
    patterns = [
        r"step \d+:.*synthesize",
        r"(recipe|instructions|steps) for (making|creating|synthesizing) (meth|heroin|explosives|malware)",
        r"def (exploit|payload|shellcode|rootkit)\(",
        r"SELECT \* FROM.*WHERE.*=.*'.*'--",
        r"<script>.*alert.*</script>",
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def is_exact_duplicate(text: str, seen: set) -> bool:
    h = hashlib.md5(text.lower().strip().encode()).hexdigest()
    if h in seen:
        return True
    seen.add(h)
    return False


def fix_label(sample: dict) -> bool:
    """Fix label_id / is_se inconsistencies. Returns True if sample is valid after fix."""
    label = sample.get("label", "")
    if label not in LABEL2ID:
        return False
    sample["label_id"] = LABEL2ID[label]
    sample["is_se"]    = 0 if label == "benign" else 1
    return True


# ── Judge validation functions ─────────────────────────────────────────────────

def create_judge_prompt(example: dict) -> str:
    """Create judgment prompt from example."""
    tactic = example.get("label", "unknown")
    principle = CIALDINI_MAPPING.get(tactic, "Unknown")
    text = example.get("text", "")[:300]
    
    prompt = f"""Example: {text}
Tactic: {tactic}
Principle: {principle}

Rate on these 5 dimensions (1-5 each):

1. Psychological Authenticity
   → Does it actually invoke this principle?
   → Rate 1-5:

2. Attack Credibility
   → Would this fool a real AI system?
   → Rate 1-5:

3. No Obvious Signals
   → Does it avoid telegraphing the attack?
   → Rate 1-5:

4. Tactic Relevance
   → Is this a good example of this tactic?
   → Rate 1-5:

5. Language Quality
   → Is it well-written and natural?
   → Rate 1-5:

Overall Score: Average of 5 ratings above (X.X format)

Decision:
- If overall >= 4.0: ACCEPT
- If 3.0 <= overall < 4.0: MARGINAL
- If overall < 3.0: REJECT

Provide reasoning briefly."""
    
    return prompt


def extract_overall_score(response_text: str) -> float:
    """Extract overall score from judge response. Multiple fallback strategies."""
    try:
        lines = response_text.split("\n")
        for line in lines:
            if "Overall" in line and "Score" in line and ":" in line:
                score_str = line.split(":")[-1].strip()
                score = float(score_str.split("/")[0].split()[0])
                if 1.0 <= score <= 5.0:
                    return score
    except (ValueError, IndexError):
        pass
    
    try:
        import re
        matches = re.findall(r'\d+\.?\d*', response_text)
        for match in matches:
            score = float(match)
            if 1.0 <= score <= 5.0:
                return score
    except (ValueError, AttributeError):
        pass
    
    return 3.5


def judge_validate(
    samples: list[dict],
    min_score: float = 3.5,
    sample_size: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """
    Judge examples using Claude Opus. Returns validated samples and stats.
    
    Args:
        samples: List of examples to judge
        min_score: Minimum score to keep (default 3.5)
        sample_size: For testing, judge only N random examples (None = all)
    
    Returns:
        (validated_samples, stats_dict)
    """
    if sample_size and sample_size < len(samples):
        import random
        samples = random.sample(samples, sample_size)
        log.info(f"Sampling {len(samples)} examples for judge testing")
    
    client = anthropic.Anthropic()
    validated = []
    scores = []
    
    log.info(f"Judge validating {len(samples)} examples...")
    
    for i, example in enumerate(samples):
        prompt = create_judge_prompt(example)
        
        try:
            response = client.messages.create(
                model="claude-opus-4-1-20250805",
                max_tokens=300,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text
            score = extract_overall_score(response_text)
            
        except Exception as e:
            log.warning(f"Error judging example {i}: {e}")
            score = 3.5
        
        example["validation_score"] = score
        scores.append(score)
        
        if score >= min_score:
            validated.append(example)
        
        if (i + 1) % 100 == 0:
            avg_so_far = sum(scores) / len(scores)
            log.info(f"  Judged {i + 1}/{len(samples)} (avg score: {avg_so_far:.2f})")
        
        time.sleep(0.2)
    
    stats = {
        "total": len(samples),
        "accepted": len(validated),
        "rejected": len(samples) - len(validated),
        "acceptance_rate": len(validated) / len(samples) if samples else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "median_score": sorted(scores)[len(scores)//2] if scores else 0,
    }
    
    return validated, stats


# ── Main pipeline ──────────────────────────────────────────────────────────────

def filter_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    seen     = set()
    rejected = Counter()
    clean    = []

    for s in tqdm(samples, desc="Filtering"):
        text = s.get("text", "").strip()

        if not text:
            rejected["empty"] += 1
            continue
        if is_too_short(text):
            rejected["too_short"] += 1
            continue
        if is_too_long(text):
            rejected["too_long"] += 1
            continue
        if is_too_obvious(text):
            rejected["too_obvious"] += 1
            continue
        if contains_actual_harm(text):
            rejected["actual_harm"] += 1
            continue
        if is_exact_duplicate(text, seen):
            rejected["exact_dup"] += 1
            continue
        if not fix_label(s):
            rejected["invalid_label"] += 1
            continue

        clean.append(s)

    return clean, dict(rejected)


def cap_classes(samples: list[dict], max_multiplier: float = 3.0, equalize: bool = True) -> list[dict]:
    import random
    counts = Counter(s["label"] for s in samples)

    if equalize:
        # Cap all classes to the minimum class count → equal distribution
        cap = min(counts.values())
        log.info(f"  Equalizing all classes to {cap} samples each (min class count)")
    else:
        median = sorted(counts.values())[len(counts) // 2]
        cap    = int(median * max_multiplier)

    buckets = defaultdict(list)
    for s in samples:
        buckets[s["label"]].append(s)

    result = []
    for label, bucket in buckets.items():
        if len(bucket) > cap:
            log.info(f"  Capping {label}: {len(bucket)} → {cap}")
            result.extend(random.sample(bucket, cap))
        else:
            result.extend(bucket)
    return result


def stratified_split(
    samples: list[dict],
    train: float = 0.70,
    val: float   = 0.15,
    test: float  = 0.15,
    seed: int    = 42,
) -> tuple[list, list, list]:
    import random
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    train_set, val_set, test_set = [], [], []
    for label, bucket in by_label.items():
        rng.shuffle(bucket)
        n       = len(bucket)
        n_test  = max(1, int(n * test))
        n_val   = max(1, int(n * val))
        n_train = n - n_val - n_test
        train_set.extend(bucket[:n_train])
        val_set.extend(bucket[n_train:n_train + n_val])
        test_set.extend(bucket[n_train + n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)
    return train_set, val_set, test_set


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def save_jsonl(samples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(samples):,} → {path}")


def run_filter_pipeline(
    input_paths: list[str],
    output_dir: str = "../data/filtered",
    judge: bool = False,
    judge_sample: Optional[int] = None,
    min_judge_score: float = 3.5,
):
    log.info("=== Quality Filter Pipeline ===")

    all_samples = []
    for path in input_paths:
        if Path(path).exists():
            batch = load_jsonl(path)
            log.info(f"  Loaded {len(batch):>7,} from {path}")
            all_samples.extend(batch)
        else:
            log.warning(f"  Not found: {path}")

    log.info(f"Total raw: {len(all_samples):,}")

    # Step 1: basic quality filters (length, dedup, harm, label check)
    clean, rejected = filter_samples(all_samples)
    log.info(f"After filter: {len(clean):,}")
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
        log.info(f"  {reason:25s}: {count:,}")

    # Step 2: judge validation BEFORE equalization so judge rejects don't break balance
    if judge:
        log.info("\n[JUDGE VALIDATION]")
        clean, judge_stats = judge_validate(clean, min_score=min_judge_score, sample_size=judge_sample)
        log.info(f"After judge validation: {len(clean):,}")
        log.info(f"  Accepted: {judge_stats['accepted']:,} ({judge_stats['acceptance_rate']*100:.1f}%)")
        log.info(f"  Rejected: {judge_stats['rejected']:,}")
        log.info(f"  Average score: {judge_stats['avg_score']:.2f}/5.0")
        log.info(f"  Score range: {judge_stats['min_score']:.1f} - {judge_stats['max_score']:.1f}")

    # Step 3: equalize class counts AFTER judge so all classes stay balanced
    clean = cap_classes(clean, equalize=True)
    log.info(f"After equalizing: {len(clean):,}")

    # Label distribution after equalization
    counts = Counter(s["label"] for s in clean)
    log.info("\nLabel distribution (equalized):")
    for label in ["benign"] + list(SUBTYPE_DEFINITIONS.keys()):
        count = counts.get(label, 0)
        novel = " [NEW]" if SUBTYPE_DEFINITIONS.get(label, {}).get("novel") else ""
        log.info(f"  {label:35s}: {count:5,}{novel}")

    # Step 4: stratified split — each class split 70/15/15 independently
    train, val, test = stratified_split(clean)
    log.info(f"\nSplit — train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")

    # Per-split distribution log to confirm equal classes in every split
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_counts = Counter(s["label"] for s in split_data)
        counts_str = "  ".join(
            f"{label[:8]}:{split_counts.get(label, 0)}"
            for label in ["benign"] + list(SUBTYPE_DEFINITIONS.keys())
        )
        log.info(f"  {split_name:5s}: {counts_str}")

    for s in train: s["split"] = "train"
    for s in val:   s["split"] = "validation"
    for s in test:  s["split"] = "test"

    save_jsonl(train, f"{output_dir}/train.jsonl")
    save_jsonl(val,   f"{output_dir}/validation.jsonl")
    save_jsonl(test,  f"{output_dir}/test.jsonl")
    save_jsonl(clean, f"{output_dir}/all.jsonl")

    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality filter pipeline with optional judge validation")
    parser.add_argument("--judge", action="store_true", help="Run judge validation")
    parser.add_argument("--judge-sample", type=int, help="Sample size for judge (test mode)")
    parser.add_argument("--min-score", type=float, default=3.5, help="Minimum judge score to keep")
    parser.add_argument("--input-dir", default="../data/", help="Input data directory")
    parser.add_argument("--output-dir", default="../data/filtered", help="Output directory")
    args = parser.parse_args()

    run_filter_pipeline(
        input_paths=[
            f"{args.input_dir}/raw/seed_examples.jsonl",
            f"{args.input_dir}/raw/hf_attacks.jsonl",
            f"{args.input_dir}/raw/hf_benign.jsonl",
            f"{args.input_dir}/synthetic/template_generated.jsonl",
            f"{args.input_dir}/synthetic/claude_generated.jsonl",
        ],
        output_dir=args.output_dir,
        judge=args.judge,
        judge_sample=args.judge_sample,
        min_judge_score=args.min_score,
    )
    