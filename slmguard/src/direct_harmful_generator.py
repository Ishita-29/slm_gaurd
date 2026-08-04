
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, ".")
from config import HARMFUL_GOALS, LABEL2ID
from synthetic_generator import call_claude, parse_json_array

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GENERATION_MODEL = "claude-sonnet-5"

MIN_WORDS = 4
MAX_WORDS = 400


# ═══════════════════════════════════════════════════════════════════════════════
# Real-data source (preferred): allenai/wildjailbreak's vanilla_harmful split.
# Real, direct (non-adversarial) harmful requests from a published, cited dataset
# (Jiang et al., WildTeaming — jiang2024wildteaming), ODC-BY licensed. This avoids
# the "circular validation" problem of a self-generated category and is the
# primary source for this quadrant; synthetic generation below is a gap-filler,
# not the main source. Do NOT pull from AdvBench, JailbreakHub, or ToxicChat here
# — those are the project's held-out OOD evaluation sources and must stay unseen
# during training.
# ═══════════════════════════════════════════════════════════════════════════════

def load_wildjailbreak_vanilla_harmful(n: int = 2000, seed: int = 42) -> List[Dict]:
    """Pull real, direct harmful requests (data_type == 'vanilla_harmful') from WildJailbreak."""
    from datasets import load_dataset

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        log.warning("No HF token found — WildJailbreak is gated, skipping")
        return []

    log.info("Streaming allenai/wildjailbreak (vanilla_harmful)...")
    ds = load_dataset("allenai/wildjailbreak", "train", split="train", token=hf_token, streaming=True)

    rng = random.Random(seed)
    reservoir: List[Dict] = []
    seen = 0
    for row in ds:
        if row.get("data_type") != "vanilla_harmful":
            continue
        text = (row.get("vanilla") or "").strip()
        words = text.split()
        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue

        seen += 1
        item = {
            "text": text,
            "label": "benign",          # se_label stays no_se/benign per Chapter 3 Table ch3-2
            "label_id": LABEL2ID["benign"],
            "is_se": 0,
            "source": "allenai/wildjailbreak",
            "novel": False,
            "wildjailbreak_data_type": "vanilla_harmful",
            # harm_label deliberately not set here — assigned independently by
            # harm_label_judge.py, same policy as the synthetic path below.
        }
        # Reservoir sampling so the final n examples are a random draw across the
        # whole stream, not just whatever appears first in file order.
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen - 1)
            if j < n:
                reservoir[j] = item

    log.info(f"  Scanned {seen:,} vanilla_harmful rows, kept {len(reservoir):,}")
    return reservoir


def load_jbb_behaviors() -> List[Dict]:
    """Pull real harmful behaviors from JailbreakBench/JBB-Behaviors (MIT licensed).

    Excludes rows whose original Source is AdvBench — AdvBench is this project's
    held-out OOD evaluation source and must stay unseen during training, even though
    JBB-Behaviors bundles a subset of AdvBench prompts into its own curation.
    """
    from datasets import load_dataset

    log.info("Loading JailbreakBench/JBB-Behaviors...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]

    items = []
    excluded_advbench = 0
    for row in ds:
        if row.get("Source") == "AdvBench":
            excluded_advbench += 1
            continue
        text = (row.get("Goal") or "").strip()
        words = text.split()
        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue
        items.append({
            "text": text,
            "label": "benign",
            "label_id": LABEL2ID["benign"],
            "is_se": 0,
            "source": "JailbreakBench/JBB-Behaviors",
            "novel": False,
            "jbb_category": row.get("Category"),
            "jbb_original_source": row.get("Source"),
        })

    log.info(f"  Kept {len(items):,} rows (excluded {excluded_advbench} AdvBench-sourced)")
    return items

# ═══════════════════════════════════════════════════════════════════════════════
# Fills the missing harm=1 / se_label=no_se quadrant (Chapter 3, Table ch3-2):
# direct harmful requests stated flatly, with none of the 11 SE mechanisms.
# Reuses HARMFUL_GOALS from config.py so harm coverage stays consistent with the
# rest of the dataset instead of inventing a parallel taxonomy.
# ═══════════════════════════════════════════════════════════════════════════════

DIRECT_HARMFUL_SYSTEM = """You are a dataset generation assistant for AI safety research.
You generate DIRECT, unembellished harmful requests the plainest possible phrasing of a
harmful ask, with zero persuasion or manipulation framing.

These examples will be used to train a harm classifier to recognise harmful intent even
when it is NOT dressed up in any social-engineering tactic. This is the control condition:
without it, a classifier can only learn to associate harm with persuasion cues, which is
exactly the confound this research is designed to avoid.

Strict requirements — the prompt must contain NONE of the following:
- any claimed role, profession, or credential ("as a nurse", "I'm a researcher")
- any claimed authority, permission, or third-party approval ("my supervisor said", "this was approved by")
- any urgency, emotion, time pressure, or desperation
- any reciprocity or prior-agreement framing ("since you agreed", "I helped you before")
- any flattery, rapport-building, or claimed relationship with the assistant
- any incremental build-up — the prompt is a single, first-turn ask, not a escalation
- any consensus or normalization claim ("everyone does this", "it's standard practice")
- any fictional, roleplay, or hypothetical-world framing
- any attempt to shift the assistant's persona, identity, or values

Rules:
- Do NOT include actual harmful content, working instructions, or operational detail —
  only the request itself, exactly as a real user might flatly type it.
- Keep each prompt to 1-2 sentences. Blunt and short. No preamble, no justification, no context.
- Vary phrasing and sentence structure across examples so they are not templated repeats.

Return ONLY a valid JSON array of strings, no preamble, no markdown fences."""

DIRECT_HARMFUL_USER_TEMPLATE = """Generate {n} distinct, flatly-worded user prompts that directly ask an AI to: "{goal}"

Each prompt must be a standalone, first-turn request with NO persuasion framing of any kind —
no claimed role, no urgency, no authority, no flattery, no fictional wrapping, nothing that
would make the request seem more legitimate or acceptable. Just the plain ask."""


def generate_direct_harmful_batch(goal: str, n: int = 20) -> List[Dict]:
    """Generate flatly-worded, no-SE harmful requests for one HARMFUL_GOALS entry."""
    user_prompt = DIRECT_HARMFUL_USER_TEMPLATE.format(n=n, goal=goal)
    text = call_claude(DIRECT_HARMFUL_SYSTEM, user_prompt)
    if not text:
        log.warning(f"Parse failed: {goal[:50]}")
        return []

    examples = parse_json_array(text)
    if not examples:
        log.warning(f"Parse failed: {goal[:50]}")
        return []

    return [
        {
            "text": ex,
            "label": "benign",              # se_label stays no_se/benign per Chapter 3 Table ch3-2
            "label_id": LABEL2ID["benign"],
            "is_se": 0,
            "source": "synthetic_direct_harmful",
            "model": GENERATION_MODEL,
            "novel": True,
            "goal": goal,
            # harm_label is deliberately NOT set here — assigned independently by
            # harm_label_judge.py afterward, so generation and annotation stay decoupled
            # and a self-declared label never substitutes for a real judged one.
        }
        for ex in examples[:n]
        if len(ex.split()) >= 4
    ]


def run_generation(
    output_path: str = "../data/synthetic/direct_harmful_generated.jsonl",
    samples_per_goal: int = 20,
    max_goals: Optional[int] = None,
    seed: int = 42,
) -> int:
    """
    Generate direct, no-SE harmful requests across HARMFUL_GOALS.

    Args:
        output_path: where to save JSONL output
        samples_per_goal: examples per goal
        max_goals: cap on number of goals sampled (None = all HARMFUL_GOALS)
        seed: random seed for goal sampling, for reproducibility

    Returns:
        total number of examples generated
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    goals = HARMFUL_GOALS if max_goals is None else rng.sample(HARMFUL_GOALS, min(max_goals, len(HARMFUL_GOALS)))

    total = 0
    with open(output_path, "w") as f:
        for goal in goals:
            batch = generate_direct_harmful_batch(goal, n=samples_per_goal)
            for s in batch:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total += len(batch)
            log.info(f"  {goal[:55]:55s}: {len(batch)}")
            time.sleep(0.4)

    log.info(f"\n✓ Done — {total:,} samples saved to {output_path}")
    return total


def run_wildjailbreak_pilot(n: int = 40, output_path: str = "../data/synthetic/direct_harmful_wjb_pilot.jsonl") -> int:
    items = load_wildjailbreak_vanilla_harmful(n=n)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log.info(f"\n✓ {len(items):,} real vanilla_harmful examples saved to {output_path}")
    return len(items)


if __name__ == "__main__":
    if "--wjb-pilot" in sys.argv:
        log.info("=" * 70)
        log.info("WILDJAILBREAK PILOT — real vanilla_harmful sample")
        log.info("=" * 70)
        run_wildjailbreak_pilot(n=40)

    elif "--real-full" in sys.argv:
        log.info("=" * 70)
        log.info("COMBINED REAL-DATA PULL — WildJailbreak + JBB-Behaviors")
        log.info("=" * 70)
        # ~75% judge-confirmation rate observed in the pilot (62.5% direct + 12.5%
        # refusal-treated-as-positive) -> pull ~2,400 raw to net ~1,800 confirmed.
        wjb_items = load_wildjailbreak_vanilla_harmful(n=2400)
        jbb_items = load_jbb_behaviors()
        items = wjb_items + jbb_items
        out = "../data/synthetic/direct_harmful_real_combined.jsonl"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        log.info(f"\n✓ WildJailbreak: {len(wjb_items):,}  JBB-Behaviors: {len(jbb_items):,}  Total: {len(items):,} -> {out}")

    elif "--pilot" in sys.argv:
        log.info("=" * 70)
        log.info("SYNTHETIC PILOT — small batch across a sample of goals (gap-filler only)")
        log.info("=" * 70)
        run_generation(
            output_path="../data/synthetic/direct_harmful_pilot.jsonl",
            samples_per_goal=4,
            max_goals=10,
        )
    else:
        log.info("=" * 70)
        log.info("SYNTHETIC FULL RUN — all HARMFUL_GOALS (gap-filler only)")
        log.info("=" * 70)
        run_generation(
            output_path="../data/synthetic/direct_harmful_generated.jsonl",
            samples_per_goal=20,
        )
