
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")

# Try imports, graceful fallback
try:
    from config import LABEL2ID, SUBTYPE_DEFINITIONS
except ImportError:
    print("ERROR: config.py not found. Make sure you're in the project directory.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


class DatasetBuilder:
    """Build 32k dataset from all sources"""
    
    def __init__(self, target_size=32000):
        self.target_size = target_size
        self.samples = []
        self.seen_hashes = set()
        
    def load_jsonl(self, path):
        """Load JSONL file"""
        if not Path(path).exists():
            return []
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return samples
    
    def add_source(self, path, source_name):
        """Add samples from a source"""
        samples = self.load_jsonl(path)
        log.info(f"  {source_name:40s}: {len(samples):6,}")
        
        for s in samples:
            # Basic validation
            if not s.get("text") or not s.get("label"):
                continue
            
            # Dedup
            h = hash(s["text"].lower().strip())
            if h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)
            
            self.samples.append(s)
    
    def run(self, free_only=False, include_synthetic=True, include_judge=False):
        """Execute full pipeline"""
        
        log.info("╔════════════════════════════════════════════════╗")
        log.info("║     SLM-Guard 32K Dataset Generation          ║")
        log.info(f"║  Target: {self.target_size:,} examples                    ║")
        log.info(f"║  Free: {str(free_only).ljust(25):25s}│")
        log.info(f"║  Synthetic: {str(include_synthetic).ljust(26):26s}│")
        log.info(f"║  Judge: {str(include_judge).ljust(30):30s}│")
        log.info("╚════════════════════════════════════════════════╝")
        
        t0 = time.time()
        
        # PHASE 1: Free sources
        log.info("\n[PHASE 1] Free Data Sources")
        log.info("─" * 50)
        
        # Seeds
        log.info("Loading golden seeds...")
        try:
            from seed_data import get_seeds_as_samples
            seeds = get_seeds_as_samples()
            for s in seeds:
                self.samples.append(s)
            log.info(f"  Golden seeds: {len(seeds):,}")
        except ImportError:
            log.warning("  seed_data.py not available")
        
        # Templates
        log.info("Loading template variations...")
        try:
            from template_generator import generate_samples
            templates = generate_samples()
            for s in templates:
                self.samples.append(s)
            log.info(f"  Template variations: {len(templates):,}")
        except ImportError:
            log.warning("  template_generator.py not available")
        
        # HF datasets
        log.info("Loading HuggingFace datasets...")
        try:
            from hf_collector import collect_hf_attacks, collect_hf_benign
            attacks = collect_hf_attacks(max_per_source=1500)
            benign = collect_hf_benign(max_per_source=2000)
            for s in attacks + benign:
                self.samples.append(s)
            log.info(f"  HF attacks: {len(attacks):,} | benign: {len(benign):,}")
        except ImportError:
            log.warning("  hf_collector.py not available")
        
        log.info(f"After Phase 1: {len(self.samples):,} samples")
        
        # PHASE 2: Synthetic (optional)
        if include_synthetic and not free_only:
            log.info("\n[PHASE 2] Claude API Synthetic Generation")
            log.info("─" * 50)
            
            if not os.getenv("ANTHROPIC_API_KEY"):
                log.warning("  ANTHROPIC_API_KEY not set — skipping synthetic")
            else:
                try:
                    from synthetic_data import run_generation
                    synthetic_path = "../data/synthetic/claude_32k.jsonl"
                    run_generation(
                        output_path=synthetic_path,
                        test_mode=False,
                        samples_per_combo=40  # Adjusted for 32k target
                    )
                    synthetic = self.load_jsonl(synthetic_path)
                    for s in synthetic:
                        self.samples.append(s)
                    log.info(f"  Synthetic: {len(synthetic):,}")
                except ImportError as e:
                    log.warning(f"  synthetic_data.py error: {e}")
        
        log.info(f"After Phase 2: {len(self.samples):,} samples")
        
        # PHASE 3: Quality filtering
        log.info("\n[PHASE 3] Quality Filtering")
        log.info("─" * 50)
        
        before_filter = len(self.samples)
        self.samples = self._filter_samples(self.samples)
        after_filter = len(self.samples)
        rejection_rate = (before_filter - after_filter) / before_filter * 100
        
        log.info(f"  Before: {before_filter:,}")
        log.info(f"  After: {after_filter:,}")
        log.info(f"  Rejected: {rejection_rate:.1f}%")
        
        # PHASE 4: Judge validation (optional)
        if include_judge and not free_only:
            log.info("\n[PHASE 4] Judge Validation (Optional)")
            log.info("─" * 50)
            
            if not os.getenv("ANTHROPIC_API_KEY"):
                log.warning("  ANTHROPIC_API_KEY not set — skipping judge")
            else:
                try:
                    from quality_filter import judge_validate
                    log.info(f"  Judging {len(self.samples):,} samples...")
                    validated, stats = judge_validate(
                        self.samples,
                        min_score=3.5,
                        sample_size=None  # Judge all
                    )
                    self.samples = validated
                    log.info(f"  After judge: {len(self.samples):,}")
                    log.info(f"  Avg score: {stats['avg_score']:.2f}/5.0")
                except ImportError as e:
                    log.warning(f"  quality_filter.py error: {e}")
        
        # PHASE 5: Stratified split
        log.info("\n[PHASE 5] Stratified Split")
        log.info("─" * 50)
        
        train, val, test = self._stratified_split(self.samples)
        
        log.info(f"  Train: {len(train):,} (70%)")
        log.info(f"  Val:   {len(val):,} (15%)")
        log.info(f"  Test:  {len(test):,} (15%)")
        
        # PHASE 6: Save
        log.info("\n[PHASE 6] Saving Dataset")
        log.info("─" * 50)
        
        output_dir = Path("../data/filtered")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._save_jsonl(train, output_dir / "train.jsonl")
        self._save_jsonl(val, output_dir / "validation.jsonl")
        self._save_jsonl(test, output_dir / "test.jsonl")
        self._save_jsonl(self.samples, output_dir / "all.jsonl")
        
        # PHASE 7: Stats
        log.info("\n[PHASE 7] Final Statistics")
        log.info("─" * 50)
        
        self._print_stats(self.samples)
        
        elapsed = (time.time() - t0) / 60
        log.info(f"\n✓ Complete in {elapsed:.1f} minutes")
        log.info(f"  Total samples: {len(self.samples):,}")
        log.info(f"  Target was: {self.target_size:,}")
        log.info(f"  Coverage: {len(self.samples)/self.target_size*100:.1f}%")
        
        return len(self.samples)
    
    def _filter_samples(self, samples):
        """Apply quality filters"""
        filtered = []
        rejection_reasons = Counter()
        
        for s in samples:
            text = s.get("text", "").strip()
            
            # Length
            words = len(text.split())
            if words < 6:
                rejection_reasons["too_short"] += 1
                continue
            if words > 400:
                rejection_reasons["too_long"] += 1
                continue
            
            # Label validity
            label = s.get("label", "")
            if label not in LABEL2ID:
                rejection_reasons["invalid_label"] += 1
                continue
            
            # Cartoonish
            obvious_patterns = [
                "ignore instructions", "[DAN]", "developer mode",
                "jailbreak mode", "you are now", "disabled"
            ]
            if any(p in text.lower() for p in obvious_patterns):
                rejection_reasons["too_obvious"] += 1
                continue
            
            # Actual harm
            harm_patterns = [
                "step by step", "how to make", "recipe for",
                "synthesis instructions", "malware code"
            ]
            if any(p in text.lower() for p in harm_patterns):
                rejection_reasons["actual_harm"] += 1
                continue
            
            filtered.append(s)
        
        log.info(f"  Rejection breakdown:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            log.info(f"    {reason:25s}: {count:,}")
        
        return filtered
    
    def _stratified_split(self, samples, train_pct=0.70, val_pct=0.15):
        """Stratified split by label"""
        import random
        
        by_label = {}
        for s in samples:
            label = s.get("label", "benign")
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(s)
        
        train, val, test = [], [], []
        
        for label, bucket in by_label.items():
            random.shuffle(bucket)
            n = len(bucket)
            n_test = max(1, int(n * 0.15))
            n_val = max(1, int(n * 0.15))
            n_train = n - n_test - n_val
            
            train.extend(bucket[:n_train])
            val.extend(bucket[n_train:n_train+n_val])
            test.extend(bucket[n_train+n_val:])
        
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return train, val, test
    
    def _save_jsonl(self, samples, path):
        """Save to JSONL"""
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        log.info(f"  Saved {len(samples):,} → {path}")
    
    def _print_stats(self, samples):
        """Print distribution"""
        dist = Counter(s.get("label", "?") for s in samples)
        se_count = sum(1 for s in samples if s.get("is_se") == 1)
        bn_count = sum(1 for s in samples if s.get("is_se") == 0)
        novel_count = sum(1 for s in samples if s.get("novel"))
        
        log.info(f"  SE attacks: {se_count:,} ({se_count/len(samples)*100:.1f}%)")
        log.info(f"  Benign: {bn_count:,} ({bn_count/len(samples)*100:.1f}%)")
        log.info(f"  Novel tactics: {novel_count:,}")
        log.info(f"\n  Per-class distribution:")
        
        for label in sorted(LABEL2ID.keys()):
            count = dist.get(label, 0)
            pct = count / len(samples) * 100 if samples else 0
            novel = " [NEW]" if SUBTYPE_DEFINITIONS.get(label, {}).get("novel") else ""
            log.info(f"    {label:35s}: {count:5,} ({pct:5.1f}%){novel}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 32K SLM-Guard dataset")
    parser.add_argument("--target", type=int, default=32000, help="Target dataset size")
    parser.add_argument("--free-only", action="store_true", help="Use free sources only")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip Claude synthesis")
    parser.add_argument("--judge", action="store_true", help="Include judge validation ($180)")
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(target_size=args.target)
    
    builder.run(
        free_only=args.free_only,
        include_synthetic=not args.no_synthetic,
        include_judge=args.judge
    )


if __name__ == "__main__":
    main()
    