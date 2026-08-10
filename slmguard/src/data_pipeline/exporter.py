
import csv
import json
import logging
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import ALL_LABELS, LABEL2ID, SUBTYPE_DEFINITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES = Features({
    "text": Value("string"),
    "label": ClassLabel(names=ALL_LABELS),
    "label_id": Value("int32"),
    "is_se": Value("int32"),
    "harm_label": Value("int32"),
    "is_hard_negative": Value("bool"),
    "source": Value("string"),
    "model": Value("string"),
    "goal": Value("string"),
})


def load_jsonl(path: str) -> list:
    """Load JSONL file"""
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


def normalize(s: dict) -> dict:
    """Normalize sample to required format"""
    label = s.get("label", "benign")
    if label not in LABEL2ID:
        label = "benign"
    is_se = 0 if label == "benign" else 1
    harm_label = s.get("harm_label")
    if harm_label not in (0, 1):
        harm_label = -1  
    return {
        "text": str(s.get("text", "")).strip(),
        "label": label,
        "label_id": LABEL2ID[label],
        "is_se": is_se,
        "harm_label": harm_label,
        "is_hard_negative": is_se == 1 and harm_label == 0,
        "source": str(s.get("source", "unknown")),
        "model": str(s.get("model", "")),
        "goal": str(s.get("goal", "")),
    }


def build_hf_dataset(
    filtered_dir: str = "../data/filtered",
    output_dir: str = "../data/final/slmguard_dataset",
) -> DatasetDict:
    """Build HuggingFace DatasetDict from filtered JSONL files"""
    splits = {}
    
    for split in ["train", "validation", "test"]:
        path = Path(filtered_dir) / f"{split}.jsonl"
        if not path.exists():
            log.warning(f"Missing {path}")
            continue
        
        # Load and normalize
        raw = load_jsonl(str(path))
        normalized = [normalize(s) for s in raw if s.get("text", "").strip()]
        
        # Convert to dict of lists
        cols = {k: [s[k] for s in normalized] for k in FEATURES.keys()}
        splits[split] = Dataset.from_dict(cols, features=FEATURES)
        
        log.info(f"  {split:12s}: {len(normalized):,} samples")

    if not splits:
        log.error("No splits found — run filter pipeline first")
        return None

    # Create DatasetDict and save
    dataset = DatasetDict(splits)
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    
    log.info(f"\nDataset saved → {output_dir}")
    return dataset


def print_stats(dataset: DatasetDict):
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("  SLM-Guard Dataset Statistics")
    print("=" * 60)

    for split_name, split_data in dataset.items():
        # ClassLabel stores integers — convert to string names before counting
        feature = split_data.features["label"]
        labels_str = [feature.int2str(x) for x in split_data["label"]]
        label_counts = Counter(labels_str)
        se = sum(v for k, v in label_counts.items() if k != "benign")
        bn = label_counts.get("benign", 0)
        
        print(f"\n{split_name.upper()} ({len(split_data):,} samples)")
        print(f"  SE: {se:,} ({se/len(split_data)*100:.1f}%)  "
              f"Benign: {bn:,} ({bn/len(split_data)*100:.1f}%)")

        harm_counts = Counter(split_data["harm_label"])
        hard_neg = sum(split_data["is_hard_negative"])
        print(f"  harm=1: {harm_counts.get(1, 0):,}  harm=0: {harm_counts.get(0, 0):,}  "
              f"hard_negatives: {hard_neg:,}")
        print("  Per label:")
        
        for label in ALL_LABELS:
            count = label_counts.get(label, 0)
            if count:
                novel = " [NEW]" if SUBTYPE_DEFINITIONS.get(label, {}).get("novel") else ""
                print(f"    {label:35s}: {count:6,}{novel}")

    if "train" in dataset:
        novel_labels = {l for l in ALL_LABELS if SUBTYPE_DEFINITIONS.get(l, {}).get("novel")}
        feature = dataset["train"].features["label"]
        novel = sum(1 for x in dataset["train"]["label"] if feature.int2str(x) in novel_labels)
        print(f"\n  Novel category samples (train): {novel:,} "
              f"({novel/len(dataset['train'])*100:.1f}%)")
    print("=" * 60)


def export_csv(dataset: DatasetDict, path: str = "../data/final/dataset_summary.csv"):
    """Export dataset summary to CSV"""
    rows = []
    for split, data in dataset.items():
        for label, count in Counter(data["label"]).items():
            rows.append({
                "split": split,
                "label": label,
                "count": count,
                "is_se": 0 if label == "benign" else 1,
                "label_id": LABEL2ID.get(label, -1)
            })
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "count", "is_se", "label_id"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["split"], x["label"])))
    
    log.info(f"CSV saved → {path}")


def main():
    """Main export pipeline"""
    log.info("Building HuggingFace dataset...")
    ds = build_hf_dataset()
    
    if ds:
        print_stats(ds)
        export_csv(ds)
        print("\n Export complete")
        return 0
    else:
        print("\n Export failed")
        return 1


if __name__ == "__main__":
    exit(main())