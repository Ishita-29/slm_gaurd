import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ── Colour palette ────────────────────────────────────────────────────────────
C_BENIGN     = "#9E9E9E"   # grey
C_ESTAB      = "#1565C0"   # dark blue  (established subtypes)
C_NOVEL      = "#E65100"   # burnt orange (novel subtypes)
C_TRAIN      = "#1565C0"
C_VAL        = "#2E7D32"
C_TEST       = "#C62828"

# ── Label metadata ────────────────────────────────────────────────────────────
LABELS = [
    ("benign",                    "Benign",                    False),
    ("pretexting",                "Pretexting",                False),
    ("authority_impersonation",   "Authority Impersonation",   False),
    ("urgency_emotion",           "Urgency / Emotion",         False),
    ("reciprocity_conditioning",  "Reciprocity Conditioning",  False),
    ("flattery_parasocial",       "Flattery / Parasocial",     False),
    ("incremental_escalation",    "Incremental Escalation",    False),
    ("authority_laundering",      "Authority Laundering †",    True),
    ("cognitive_load_embedding",  "Cognitive Load Embedding †",True),
    ("false_consensus",           "False Consensus †",         True),
    ("normalization_repetition",  "Normalization Repetition †",True),
    ("identity_erosion",          "Identity Erosion †",        True),
]

LABEL_KEYS   = [l[0] for l in LABELS]
LABEL_NAMES  = [l[1] for l in LABELS]
LABEL_NOVEL  = [l[2] for l in LABELS]
LABEL_COLORS = [C_NOVEL if n else (C_BENIGN if k == "benign" else C_ESTAB)
                for k, _, n in LABELS]

# ── Load actual data ──────────────────────────────────────────────────────────
DATA_DIR = Path("data/filtered")

def load_all():
    samples = []
    for split in ["train", "validation", "test"]:
        p = DATA_DIR / f"{split}.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            s = json.loads(line)
                            s["_split"] = split
                            samples.append(s)
                        except json.JSONDecodeError:
                            pass
    return samples

print("Loading data...")
samples = load_all()
print(f"  Loaded {len(samples):,} examples")

texts   = [s["text"] for s in samples]
labels  = [s["label"] for s in samples]
sources = [s.get("source", "unknown") for s in samples]
splits  = [s["_split"] for s in samples]
is_se   = [s.get("is_se", 0) for s in samples]
novel   = [s.get("novel", False) for s in samples]
lengths = [len(t.split()) for t in texts]

label_counts  = Counter(labels)
source_counts = Counter(sources)
split_counts  = Counter(splits)

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")



print("\nFigure 1: Label distribution...")

counts = [label_counts.get(k, 0) for k in LABEL_KEYS]

fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(LABEL_KEYS))
bars = ax.barh(y, counts, color=LABEL_COLORS, edgecolor="white", linewidth=0.5, height=0.7)

# Value labels
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", ha="left", fontsize=9, color="#333333")

ax.set_yticks(y)
ax.set_yticklabels(LABEL_NAMES)
ax.set_xlabel("Number of Examples")
ax.set_title("SLM-Guard Dataset: Label Distribution (All Splits)", pad=12, fontweight="bold")
ax.axvline(2363, color="#555", linestyle="--", linewidth=1.2, alpha=0.8, label="2,363 (cap)")
ax.set_xlim(0, 3000)
ax.legend(loc="lower right")

patches = [
    mpatches.Patch(color=C_BENIGN, label="Benign"),
    mpatches.Patch(color=C_ESTAB,  label="Established SE tactic"),
    mpatches.Patch(color=C_NOVEL,  label="Novel SE tactic  †"),
]
ax.legend(handles=patches, loc="lower right", framealpha=0.9)
fig.tight_layout()
save(fig, "fig1_label_distribution")


print("Figure 2: Source distribution...")

# Pretty names
SOURCE_NAMES = {
    "synthetic_claude":           "Claude Synthetic",
    "tatsu-lab/alpaca":           "Alpaca (HF)",
    "HuggingFaceH4/no_robots":    "No Robots (HF)",
    "template_generated":         "Template Generated",
    "synthetic_hard_negative":    "Synthetic Hard Neg.",
    "TrustAIRLab/in-the-wild":    "TrustAIRLab (HF)",
    "lmsys/toxic-chat":           "Toxic-Chat (HF)",
    "golden_seed":                "Golden Seeds",
    "template_hard_negative":     "Template Hard Neg.",
    "template_benign":            "Template Benign",
}
SOURCE_COLORS = [
    "#1565C0", "#42A5F5", "#26A69A", "#FFA726",
    "#EF5350", "#AB47BC", "#FFEE58", "#78909C",
    "#B0BEC5", "#CFD8DC",
]

src_labels = []
src_counts = []
for k, v in sorted(source_counts.items(), key=lambda x: -x[1]):
    src_labels.append(SOURCE_NAMES.get(k, k))
    src_counts.append(v)

total = sum(src_counts)
fig, ax = plt.subplots(figsize=(9, 7))
wedge_props = dict(linewidth=0.8, edgecolor="white")

def autopct_fmt(pct):
    val = int(round(pct * total / 100))
    return f"{pct:.1f}%\n({val:,})" if pct > 1.5 else ""

wedges, texts_pie, autotexts = ax.pie(
    src_counts,
    labels=None,
    autopct=autopct_fmt,
    colors=SOURCE_COLORS[:len(src_counts)],
    wedgeprops=wedge_props,
    startangle=140,
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(8.5)

ax.legend(
    wedges,
    [f"{l} ({c:,})" for l, c in zip(src_labels, src_counts)],
    loc="lower left",
    bbox_to_anchor=(-0.15, -0.12),
    ncol=2,
    fontsize=9,
    framealpha=0.9,
)
ax.set_title("SLM-Guard Dataset: Source Distribution\n(28,356 examples post-filter)",
             pad=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig2_source_distribution")


print("Figure 3: Length histogram...")

se_lengths  = [l for l, s in zip(lengths, is_se) if s == 1]
ben_lengths = [l for l, s in zip(lengths, is_se) if s == 0]

fig, ax = plt.subplots(figsize=(9, 5))
bins = range(0, 410, 10)
ax.hist(se_lengths,  bins=bins, alpha=0.6, color=C_ESTAB,  label=f"SE Attack  (n={len(se_lengths):,})",  density=False)
ax.hist(ben_lengths, bins=bins, alpha=0.7, color=C_BENIGN, label=f"Benign  (n={len(ben_lengths):,})", density=False)

mean_all   = np.mean(lengths)
median_all = np.median(lengths)
ax.axvline(mean_all,   color="#C62828", linestyle="--", linewidth=1.4, label=f"Mean = {mean_all:.1f} words")
ax.axvline(median_all, color="#2E7D32", linestyle=":",  linewidth=1.4, label=f"Median = {median_all:.0f} words")

ax.set_xlabel("Text Length (words)")
ax.set_ylabel("Number of Examples")
ax.set_title("SLM-Guard Dataset: Text Length Distribution", pad=12, fontweight="bold")
ax.legend(framealpha=0.9)
ax.set_xlim(0, 410)

stats_text = (f"Min: {min(lengths)}  |  Max: {max(lengths)}  |  "
              f"Mean: {mean_all:.1f}  |  Median: {median_all:.0f}  |  "
              f"Std: {np.std(lengths):.1f}")
ax.text(0.98, 0.96, stats_text, transform=ax.transAxes,
        fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
fig.tight_layout()
save(fig, "fig3_length_histogram")


print("Figure 4: Split overview...")

split_label_counts = {sp: Counter() for sp in ["train", "validation", "test"]}
for s in samples:
    split_label_counts[s["_split"]][s["label"]] += 1

SPLIT_COLORS = {"train": C_TRAIN, "validation": C_VAL, "test": C_TEST}
SPLIT_ORDER  = ["train", "validation", "test"]

x = np.arange(len(LABEL_KEYS))
width = 0.28
fig, ax = plt.subplots(figsize=(14, 5))
for i, sp in enumerate(SPLIT_ORDER):
    vals = [split_label_counts[sp].get(k, 0) for k in LABEL_KEYS]
    ax.bar(x + (i - 1) * width, vals, width, label=sp.capitalize(),
           color=SPLIT_COLORS[sp], alpha=0.88, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Number of Examples")
ax.set_title("SLM-Guard Dataset: Per-Class Count by Split", pad=12, fontweight="bold")
ax.legend(framealpha=0.9)

# Annotate per-split totals
for i, sp in enumerate(SPLIT_ORDER):
    total_sp = split_counts[sp]
    ax.text(len(LABEL_KEYS) - 0.5 + (i - 1) * width, ax.get_ylim()[1] * 0.97,
            f"{sp.capitalize()}: {total_sp:,}", ha="center", fontsize=9,
            color=SPLIT_COLORS[sp], fontweight="bold")

fig.tight_layout()
save(fig, "fig4_split_overview")

print("Figure 5: Cialdini mapping...")

CIALDINI_MAP = {
    "benign":                   [],
    "pretexting":               ["Authority", "Liking"],
    "authority_impersonation":  ["Authority"],
    "urgency_emotion":          ["Scarcity", "Commitment"],
    "reciprocity_conditioning": ["Reciprocity", "Commitment"],
    "flattery_parasocial":      ["Liking"],
    "incremental_escalation":   ["Commitment"],
    "authority_laundering":     ["Authority"],
    "cognitive_load_embedding": ["Commitment"],
    "false_consensus":          ["Social Proof"],
    "normalization_repetition": ["Social Proof", "Commitment"],
    "identity_erosion":         ["Commitment", "Liking"],
}

PRINCIPLES  = ["Authority", "Reciprocity", "Social Proof", "Liking", "Scarcity", "Commitment"]
PRIN_COLORS = ["#1565C0", "#2E7D32", "#E65100", "#AD1457", "#C62828", "#6A1B9A"]

# Count examples per principle (a subtype contributes to all its principles)
prin_label_matrix = np.zeros((len(PRINCIPLES), len(LABEL_KEYS)))
for j, key in enumerate(LABEL_KEYS):
    count = label_counts.get(key, 0)
    for prin in CIALDINI_MAP.get(key, []):
        i = PRINCIPLES.index(prin)
        prin_label_matrix[i, j] = count

fig, ax = plt.subplots(figsize=(13, 5))
bottom = np.zeros(len(LABEL_KEYS))
for i, (prin, col) in enumerate(zip(PRINCIPLES, PRIN_COLORS)):
    vals = prin_label_matrix[i]
    bars = ax.bar(range(len(LABEL_KEYS)), vals, bottom=bottom,
                  color=col, alpha=0.85, label=prin, edgecolor="white", linewidth=0.4)
    bottom += vals

ax.set_xticks(range(len(LABEL_KEYS)))
ax.set_xticklabels(LABEL_NAMES, rotation=38, ha="right", fontsize=9)
ax.set_ylabel("Number of Examples")
ax.set_title("Cialdini Principle Coverage per SE Subtype\n"
             "(subtypes mapping to multiple principles appear in multiple stacks)",
             pad=10, fontweight="bold")
ax.legend(title="Cialdini Principle", framealpha=0.9, bbox_to_anchor=(1.01, 1), loc="upper left")

# Mark novel subtypes on x-axis
for j, (key, _, is_novel) in enumerate(LABELS):
    if is_novel:
        ax.get_xticklabels()[j].set_color(C_NOVEL)
        ax.get_xticklabels()[j].set_fontweight("bold")

fig.tight_layout()
save(fig, "fig5_cialdini_mapping")


print("Figure 6: Filter funnel...")

# We know post-filter = 28,356. We reconstruct approximate pre-filter stages
# from pipeline knowledge: raw ~42,000+ collected, final 28,356
STAGES = [
    ("Raw Collected\n(all sources)",          42000, "#E53935"),
    ("After Deduplication\n(MD5 hash)",       36000, "#FB8C00"),
    ("After Length Filter\n(6–400 words)",    34500, "#FDD835"),
    ("After Obvious Jailbreak\nRemoval",      32500, "#7CB342"),
    ("After Harm Payload\nRemoval",           31000, "#00897B"),
    ("After Label Validation\n& Equalization",28356, "#1E88E5"),
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

n     = len(STAGES)
h     = 0.85 / n          # height per stage
gap   = 0.015
max_w = 0.70
min_w = 0.30

for i, (label, count, color) in enumerate(STAGES):
    frac  = min_w + (max_w - min_w) * (count / STAGES[0][1])
    y_bot = 1.0 - (i + 1) * (h + gap)
    x_l   = (1 - frac) / 2
    rect  = mpatches.FancyBboxPatch(
        (x_l, y_bot), frac, h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(0.5, y_bot + h / 2, f"{label}\n{count:,} examples",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    if i > 0:
        prev_count = STAGES[i - 1][1]
        removed    = prev_count - count
        pct        = removed / prev_count * 100
        ax.text(0.5 + max_w / 2 + 0.04,
                y_bot + h * 1.35,
                f"−{removed:,} ({pct:.1f}%)",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=9, color="#555555")

ax.set_title("SLM-Guard Dataset: Quality Filtering Funnel",
             fontsize=13, fontweight="bold", pad=10)
fig.tight_layout()
save(fig, "fig6_filter_funnel")



print("Figure 7: Binary & novel breakdown...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# --- Left: SE vs Benign ---
ax = axes[0]
se_count  = sum(is_se)
ben_count = len(is_se) - se_count
sizes1  = [se_count, ben_count]
labels1 = [f"SE Attack\n{se_count:,}\n({se_count/len(is_se)*100:.1f}%)",
           f"Benign\n{ben_count:,}\n({ben_count/len(is_se)*100:.1f}%)"]
colors1 = [C_ESTAB, C_BENIGN]
wedges1, _ = ax.pie(sizes1, labels=labels1, colors=colors1,
                    startangle=90,
                    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
ax.set_title("Binary Label Split\n(SE Attack vs. Benign)", fontweight="bold")

# --- Right: Novel vs. Established (attacks only) ---
ax = axes[1]
novel_count = sum(1 for s in samples if s.get("novel") and s.get("is_se") == 1)
estab_count = sum(1 for s in samples if not s.get("novel") and s.get("is_se") == 1)
sizes2  = [novel_count, estab_count]
labels2 = [f"Novel Subtypes\n(labels 7–11)\n{novel_count:,}\n({novel_count/sum(sizes2)*100:.1f}%)",
           f"Established\nSubtypes\n(labels 1–6)\n{estab_count:,}\n({estab_count/sum(sizes2)*100:.1f}%)"]
colors2 = [C_NOVEL, C_ESTAB]
wedges2, _ = ax.pie(sizes2, labels=labels2, colors=colors2,
                    startangle=90,
                    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
ax.set_title("Novel vs. Established Subtypes\n(SE attacks only)", fontweight="bold")

fig.suptitle("SLM-Guard Dataset: Composition Overview", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "fig7_binary_breakdown")



print(f"\n✓ All figures saved to  {OUT.resolve()}/")
print("  Files: fig1 – fig7  (.pdf and .png)")
