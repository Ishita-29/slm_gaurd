
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from config import LABEL2ID, SUBTYPE_DEFINITIONS


def load_jsonl(path):
    """Load JSONL file, skip invalid lines"""
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


def section(title):
    """Print section header"""
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    # Load all splits
    splits = {}
    for split in ["train", "validation", "test"]:
        p = Path(f"../data/filtered/{split}.jsonl")
        if p.exists():
            splits[split] = load_jsonl(str(p))

    if not splits:
        print("❌ No filtered data found. Run: python main.py --step filter")
        sys.exit(1)

    all_samples = [s for sp in splits.values() for s in sp]

    # Overall stats
    section("Overall Statistics")
    total = len(all_samples)
    se = sum(1 for s in all_samples if s.get("is_se") == 1)
    bn = sum(1 for s in all_samples if s.get("is_se") == 0)
    novel = sum(1 for s in all_samples if s.get("novel"))

    print(f"  Total   : {total:,}")
    for name, sp in splits.items():
        print(f"  {name:12s}: {len(sp):,}")
    print(f"  SE      : {se:,} ({se/total*100:.1f}%)")
    print(f"  Benign  : {bn:,} ({bn/total*100:.1f}%)")
    print(f"  Novel   : {novel:,} ({novel/total*100:.1f}%)")

    # Label distribution
    section("Label Distribution")
    dist = Counter(s.get("label", "unknown") for s in all_samples)
    max_count = max(dist.values()) if dist else 1
    labels_ordered = ["benign"] + sorted(SUBTYPE_DEFINITIONS.keys())
    
    for label in labels_ordered:
        count = dist.get(label, 0)
        bar = "█" * int(count / max_count * 30) if max_count > 0 else ""
        novel_tag = " [NEW]" if SUBTYPE_DEFINITIONS.get(label, {}).get("novel") else ""
        print(f"  {label:35s}: {count:5,} {bar}{novel_tag}")

    # Text length
    section("Text Length (words)")
    texts = [s.get("text", "") for s in all_samples if s.get("text")]
    if texts:
        lengths = [len(t.split()) for t in texts]
        print(f"  Min    : {min(lengths)}")
        print(f"  Max    : {max(lengths)}")
        print(f"  Mean   : {statistics.mean(lengths):.1f}")
        print(f"  Median : {statistics.median(lengths):.1f}")
        print(f"  StDev  : {statistics.stdev(lengths):.1f}")

    # Source distribution
    section("Source Distribution")
    sources = Counter(s.get("source", "unknown") for s in all_samples)
    for src, count in sources.most_common():
        print(f"  {src:45s}: {count:,}")

    # Sample examples
    section("Sample Examples")
    shown_labels = ["pretexting", "urgency_emotion", "authority_laundering",
                    "identity_erosion", "cognitive_load_embedding", "benign"]
    for label in shown_labels:
        subset = [s for s in all_samples if s.get("label") == label][:2]
        total_label = dist.get(label, 0)
        print(f"\n  [{label}]  ({total_label} total)")
        for s in subset:
            text = s.get("text", "")[:160] + ("..." if len(s.get("text", "")) > 160 else "")
            print(f"    {text}")

    # Quality checks
    section("Quality Checks")
    errors = sum(1 for s in all_samples 
                 if LABEL2ID.get(s.get("label"), -1) != s.get("label_id"))
    harm_found = sum(1 for s in all_samples
                     if any(p in s.get("text", "").lower()
                            for p in ["synthesize meth", "step-by-step bomb", "instructions for"]))
    all_fields = sum(1 for s in all_samples 
                     if all(k in s for k in ['text', 'label', 'label_id', 'is_se']))

    print(f"  Label consistency errors : {errors}")
    print(f"  Harmful payload detected : {harm_found}")
    print(f"  All fields present       : {all_fields}/{len(all_samples)}")

    # Hard negatives
    section("Hard Negative Analysis")
    hn = [s for s in all_samples if "hard_negative" in s.get("source", "")]
    print(f"  Hard negatives total : {len(hn)}")
    if hn:
        print("  Examples:")
        for s in hn[:3]:
            print(f"    {s.get('text', '')[:140]}...")

    # Summary
    print("\n✓ Dataset analysis complete")
    print(f"  Load for training: datasets.load_from_disk('../data/final/slmguard_dataset')")
    print(f"  Next step: python train.py")


if __name__ == "__main__":
    main()