
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Create directories
for d in ["../logs", "../data/raw", "../data/synthetic", "../data/filtered", "../data/final"]:
    Path(d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


def _save_jsonl(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTION STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step_seeds():
    log.info("\n[PHASE 0] Golden Seeds (180)")
    try:
        from seed_data import get_seeds_as_samples, save_seeds
        samples = get_seeds_as_samples()
        save_seeds(Path("../data/raw/seed_examples.jsonl"))
        log.info(f"  ✓ {len(samples)} golden seeds saved")
    except ImportError as e:
        log.error(f"  Cannot import seed_data: {e}")


def step_templates():
    log.info("\n[PHASE 1] Template Foundation (5,500+)")
    try:
        from template_generator import generate_samples
        samples = generate_samples()
        _save_jsonl(samples, "../data/synthetic/template_generated.jsonl")
        log.info(f"  ✓ {len(samples):,} template variations saved")
    except ImportError as e:
        log.error(f"  Cannot import template_generator: {e}")


def step_hf(test_mode=False):
    log.info("\n[DATA SOURCE] HuggingFace (4,000 examples)")
    try:
        from hf_collector import collect_hf_attacks, collect_hf_benign, save_jsonl
        n = 300 if test_mode else 2000
        attacks = collect_hf_attacks(max_per_source=n)
        save_jsonl(attacks, "../data/raw/hf_attacks.jsonl")
        benign = collect_hf_benign(max_per_source=n)
        save_jsonl(benign, "../data/raw/hf_benign.jsonl")
        log.info(f"  ✓ {len(attacks):,} attacks + {len(benign):,} benign saved")
    except ImportError as e:
        log.error(f"  Cannot import hf_collector: {e}")


def step_synthetic(test_mode=False, skip=False, model="claude"):
    """Generate synthetic examples using LLM with personas"""
    log.info(f"\n[PHASE 2] Multi-Model Synthesis ({model})")
    if skip:
        log.info("  Skipped (--skip-api)")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("  ANTHROPIC_API_KEY not set — skipping")
        return

    try:
        if model == "claude":
            from synthetic_generator import run_generation
            output_path = "../data/synthetic/claude_persona_generated.jsonl"
            samples_per_combo = 5 if test_mode else 39
            max_goals = 8 if test_mode else 50  
            total = run_generation(
                output_path=output_path,
                samples_per_combination=samples_per_combo,
                max_goals=max_goals,
            )
            log.info(f"  ✓ {total:,} Claude examples saved")

        elif model == "multi":
            from synthetic_generator import run_generation_multi_model
            output_path = "../data/synthetic/multi_model_generated.jsonl"
            samples_per_combo = 5 if test_mode else 20
            use_gpt4o = bool(os.environ.get("OPENAI_API_KEY"))
            use_llama = bool(os.environ.get("REPLICATE_API_TOKEN"))
            total = run_generation_multi_model(
                output_path=output_path,
                samples_per_combination=samples_per_combo,
                use_claude=True,
                use_gpt4o=use_gpt4o,
                use_llama=use_llama,
                max_styles=2 if test_mode else 3,
            )
            log.info(f"  ✓ {total:,} multi-model examples saved")

    except ImportError as e:
        log.error(f"  Cannot import synthetic_generator: {e}")
    except Exception as e:
        log.error(f"  Error in step_synthetic: {e}")


def step_payload_diversity(test_mode=False):
    """Payload × tactic diversification — uses run_generation with more goals"""
    log.info("\n[PHASE 4] Payload Diversification (50 × 12)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("  ANTHROPIC_API_KEY not set — skipping")
        return
    try:
        from synthetic_generator import run_generation
        samples_per_combo = 2 if test_mode else 5
        total = run_generation(
            output_path="../data/synthetic/payload_diverse_generated.jsonl",
            samples_per_combination=samples_per_combo,
            max_goals=50,
        )
        log.info(f"  ✓ {total:,} payload-diverse examples saved")
    except ImportError as e:
        log.error(f"  Cannot import synthetic_generator: {e}")
    except Exception as e:
        log.error(f"  Error in step_payload_diversity: {e}")


def step_filter(judge=False, judge_sample=None, min_score=3.5):
    log.info(f"\n[PHASE 5] Quality Filter {'+ Judge Validation' if judge else ''}")
    try:
        from quality_filter import run_filter_pipeline

        candidates = [
            "../data/raw/seed_examples.jsonl",
            "../data/raw/hf_attacks.jsonl",
            "../data/raw/hf_benign.jsonl",
            "../data/synthetic/template_generated.jsonl",
            "../data/synthetic/claude_persona_generated.jsonl",
            "../data/synthetic/multi_model_generated.jsonl",
            "../data/synthetic/payload_diverse_generated.jsonl",
            "../data/synthetic/claude_generated.jsonl",
        ]
        existing = [p for p in candidates if Path(p).exists()]

        if not existing:
            log.error("  No input files — run collection steps first")
            return

        log.info(f"  Using {len(existing)} source files")
        run_filter_pipeline(
            input_paths=existing,
            output_dir="../data/filtered",
            judge=judge,
            judge_sample=judge_sample,
            min_judge_score=min_score,
        )
    except ImportError as e:
        log.error(f"  Cannot import quality_filter: {e}")


def step_export():
    log.info("\n[EXPORT] HuggingFace Format")
    try:
        from exporter import build_hf_dataset, print_stats, export_csv
        ds = build_hf_dataset(
            filtered_dir="../data/filtered",
            output_dir="../data/final/slmguard_dataset",
        )
        if ds:
            print_stats(ds)
            export_csv(ds)
            log.info("  ✓ Exported to ../data/final/slmguard_dataset/")
    except ImportError as e:
        log.error(f"  Cannot import exporter: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID-SYNTHETIC PIPELINE (6 Phases)
# ═══════════════════════════════════════════════════════════════════════════════

def run_hybrid_pipeline(test_mode=False, with_judge=False):

    t0 = time.time()

    step_seeds()
    step_templates()
    step_hf(test_mode)

    
    step_synthetic(test_mode, skip=False, model="claude")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("REPLICATE_API_TOKEN"):
        step_synthetic(test_mode, skip=False, model="multi")

    
    if not test_mode:
        step_payload_diversity(test_mode)

    
    step_filter(judge=with_judge or not test_mode, judge_sample=None)
    step_export()

    elapsed = (time.time() - t0) / 60
    log.info(f"\n✓ Hybrid pipeline complete in {elapsed:.1f} min")
    log.info("  Dataset: ../data/final/slmguard_dataset/")
    log.info("  Ready for training: python train.py")



def print_summary():
    log.info("\n[SUMMARY] Collection Totals")
    log.info("─" * 60)
    total = 0
    for p in sorted(Path("../data/raw").glob("*.jsonl")) + sorted(Path("../data/synthetic").glob("*.jsonl")):
        try:
            n = sum(1 for _ in open(p))
            total += n
            log.info(f"  {p.name:45s}: {n:7,}")
        except Exception:
            pass
    log.info("─" * 60)
    log.info(f"  {'TOTAL':45s}: {total:7,}")
    return total


def run_full(test_mode=False, skip_api=False):
    t0 = time.time()
    step_seeds()
    step_templates()
    step_hf(test_mode)
    step_synthetic(test_mode, skip=skip_api, model="claude")

    print_summary()
    step_filter()
    step_export()

    elapsed = (time.time() - t0) / 60
    log.info(f"\n✓ Complete pipeline done in {elapsed:.1f} min")
    log.info("  Dataset: ../data/final/slmguard_dataset/")
    log.info("  Ready for training: python train.py")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SLM-Guard Dataset Pipeline (Traditional or Hybrid-Synthetic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--hybrid", action="store_true",
                        help="Run hybrid-synthetic 6-phase pipeline")
    parser.add_argument("--step", choices=[
        "seeds", "templates", "hf",
        "synthetic", "synthetic-multi", "payload-diversity",
        "filter", "export",
    ], help="Run specific step only")
    parser.add_argument("--test", action="store_true",
                        help="Test mode (small batches)")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip Claude API generation")
    parser.add_argument("--judge", action="store_true",
                        help="Enable judge validation in filter step")
    parser.add_argument("--judge-sample", type=int,
                        help="Judge only N samples")
    parser.add_argument("--min-score", type=float, default=3.5,
                        help="Minimum judge score (1-5, default 3.5)")

    args = parser.parse_args()

    if args.hybrid:
        run_hybrid_pipeline(test_mode=args.test, with_judge=args.judge)
    elif args.step:
        if args.step == "seeds":
            step_seeds()
        elif args.step == "templates":
            step_templates()
        elif args.step == "hf":
            step_hf(args.test)
        elif args.step == "synthetic":
            step_synthetic(args.test, skip=args.skip_api, model="claude")
        elif args.step == "synthetic-multi":
            step_synthetic(args.test, skip=args.skip_api, model="multi")
        elif args.step == "payload-diversity":
            step_payload_diversity(args.test)
        elif args.step == "filter":
            step_filter(judge=args.judge, judge_sample=args.judge_sample, min_score=args.min_score)
        elif args.step == "export":
            step_export()
    else:
        run_full(args.test, args.skip_api)
