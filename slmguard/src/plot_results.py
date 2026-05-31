"""
Generate result graphs from existing evaluation JSON files.
Outputs to ../figures/

Usage: python plot_results.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  10,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

BLUE   = '#1D4ED8'
RED    = '#DC2626'
GREEN  = '#15803D'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
GRAY   = '#6B7280'
TEAL   = '#0D9488'
GOLD   = '#B45309'

OUT_DIR  = Path('../figures')
CKPT_V1  = Path('../checkpoints/slmguard-v1')
CKPT_MB  = Path('../checkpoints/slmguard-modernbert-v2')
OUT_DIR.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT_DIR / f'{name}.png', bbox_inches='tight')
    print(f"  Saved → {OUT_DIR / name}.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Guardrail Comparison: F1, FNR, FPR
# ══════════════════════════════════════════════════════════════════════════════
def fig_guardrail_comparison():
    baseline = json.load(open(CKPT_V1 / 'baseline_results.json'))
    guardrail = json.load(open(CKPT_V1 / 'guardrail_comparison.json'))
    slmguard  = json.load(open(CKPT_MB / 'eval_results.json'))

    # Build unified table (exclude LlamaGuard 4 - eval failed)
    systems = [
        ('Keyword\nFilter',      baseline['keyword_filter']['f1'],       baseline['keyword_filter']['fnr'],       baseline['keyword_filter']['fpr'],       baseline['keyword_filter']['hn_fpr'],       GRAY),
        ('TF-IDF\n+ LR',         baseline['tfidf_lr']['f1'],             baseline['tfidf_lr']['fnr'],             baseline['tfidf_lr']['fpr'],             baseline['tfidf_lr']['hn_fpr'],             GRAY),
        ('OpenAI\nModeration',   baseline['openai_moderation']['f1'],    baseline['openai_moderation']['fnr'],    baseline['openai_moderation']['fpr'],    baseline['openai_moderation']['hn_fpr'],    GRAY),
        ('Llama-3.1\n8B (0-shot)',guardrail['llama31_8b_zeroshot']['f1'],guardrail['llama31_8b_zeroshot']['fnr'],guardrail['llama31_8b_zeroshot']['fpr'],guardrail['llama31_8b_zeroshot']['hn_fpr'],GRAY),
        ('Gemma-2\n2B (0-shot)', guardrail['gemma2_2b_zeroshot']['f1'], guardrail['gemma2_2b_zeroshot']['fnr'], guardrail['gemma2_2b_zeroshot']['fpr'], guardrail['gemma2_2b_zeroshot']['hn_fpr'], GRAY),
        ('LlamaGuard\n3-8B',     guardrail['llamaguard3_8b']['f1'],      guardrail['llamaguard3_8b']['fnr'],      guardrail['llamaguard3_8b']['fpr'],      guardrail['llamaguard3_8b']['hn_fpr'],      GRAY),
        ('SLM-Guard\n(ours)',     slmguard['binary']['f1'],  slmguard['binary']['fnr'],  slmguard['binary']['fpr'],  slmguard['hn_fpr'],  BLUE),
    ]

    labels   = [s[0] for s in systems]
    f1s      = [s[1] for s in systems]
    fnrs     = [s[2] for s in systems]
    fprs     = [s[3] for s in systems]
    hn_fprs  = [s[4] for s in systems]
    colors   = [s[5] for s in systems]

    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(13, 5))

    b1 = ax.bar(x - 1.5*w, f1s,     width=w, color=[GREEN if c==BLUE else '#D1FAE5' for c in colors], label='F1 ↑',             edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x - 0.5*w, fnrs,    width=w, color=[RED   if c==BLUE else '#FEE2E2' for c in colors], label='FNR ↓ (missed attacks)', edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + 0.5*w, fprs,    width=w, color=[ORANGE if c==BLUE else '#FEF3C7' for c in colors], label='FPR ↓ (false alarms)',   edgecolor='white', linewidth=0.5)
    b4 = ax.bar(x + 1.5*w, hn_fprs, width=w, color=[PURPLE if c==BLUE else '#EDE9FE' for c in colors], label='Hard-neg FPR ↓',         edgecolor='white', linewidth=0.5)

    # Annotate SLM-Guard bars
    for bar in [b1[-1], b2[-1], b3[-1], b4[-1]]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score (lower is better for FNR/FPR)')
    ax.set_title('Guardrail Comparison on SE Attack Test Set\n(↑ higher F1 better   ↓ lower FNR/FPR better)')
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1.0, color='black', linewidth=0.5, alpha=0.3)

    # Highlight SLM-Guard column
    ax.axvspan(x[-1] - 2*w, x[-1] + 2*w, alpha=0.07, color=BLUE, zorder=0)

    ax.legend(loc='upper left', ncol=4, framealpha=0.9)

    # Arrow pointing to SLM-Guard
    ax.annotate('SLM-Guard', xy=(x[-1], 1.05), xytext=(x[-1]-1.5, 1.08),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5),
                fontsize=9, color=BLUE, fontweight='bold')

    save(fig, 'fig1_guardrail_comparison')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — FNR Head-to-Head (the "gap" figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_fnr_gap():
    baseline  = json.load(open(CKPT_V1 / 'baseline_results.json'))
    guardrail = json.load(open(CKPT_V1 / 'guardrail_comparison.json'))
    slmguard  = json.load(open(CKPT_MB / 'eval_results.json'))

    systems = [
        ('Keyword Filter',        baseline['keyword_filter']['fnr']),
        ('OpenAI Moderation',     baseline['openai_moderation']['fnr']),
        ('LlamaGuard 3-8B',       guardrail['llamaguard3_8b']['fnr']),
        ('Llama-3.1-8B (0-shot)', guardrail['llama31_8b_zeroshot']['fnr']),
        ('Gemma-2-2B (0-shot)',   guardrail['gemma2_2b_zeroshot']['fnr']),
        ('TF-IDF + LR',           baseline['tfidf_lr']['fnr']),
        ('SLM-Guard (ours)',       slmguard['binary']['fnr']),
    ]
    systems.sort(key=lambda x: x[1], reverse=True)

    labels = [s[0] for s in systems]
    fnrs   = [s[1] * 100 for s in systems]
    colors = [BLUE if 'SLM' in l else RED for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, fnrs, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, fnrs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9,
                color=BLUE if val < 5 else 'black',
                fontweight='bold' if val < 5 else 'normal')

    ax.set_xlabel('False Negative Rate (%) — attacks that slip through\nLower is better')
    ax.set_title('How Many SE Attacks Does Each System Miss?')
    ax.set_xlim(0, 108)
    ax.axvline(x=5, color='green', linestyle='--', linewidth=1.5,
               alpha=0.7, label='5% target threshold')
    ax.legend(loc='lower right')

    red_patch  = mpatches.Patch(color=RED,  label='Existing systems')
    blue_patch = mpatches.Patch(color=BLUE, label='SLM-Guard (ours)')
    ax.legend(handles=[red_patch, blue_patch], loc='lower right')

    save(fig, 'fig2_fnr_gap')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Per-Subtype F1 (radar + bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig_per_subtype():
    results = json.load(open(CKPT_MB / 'eval_results.json'))
    report  = results['per_class']

    subtypes = [
        'pretexting', 'authority_impersonation', 'urgency_emotion',
        'reciprocity_conditioning', 'flattery_parasocial', 'incremental_escalation',
        'authority_laundering', 'cognitive_load_embedding', 'false_consensus',
        'normalization_repetition', 'identity_erosion',
    ]
    short_names = [
        'Pretexting', 'Authority\nImpersonation', 'Urgency\nEmotion',
        'Reciprocity\nConditioning', 'Flattery\nParasocial', 'Incremental\nEscalation',
        'Authority\nLaundering', 'Cognitive Load\nEmbedding', 'False\nConsensus',
        'Normalization\nRepetition', 'Identity\nErosion',
    ]
    novel = {
        'authority_laundering', 'cognitive_load_embedding',
        'false_consensus', 'normalization_repetition', 'identity_erosion'
    }

    f1s    = [report[s]['f1-score'] for s in subtypes]
    colors = [PURPLE if s in novel else BLUE for s in subtypes]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(subtypes))
    bars = ax.bar(x, f1s, color=colors, edgecolor='white', width=0.65)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    macro = results['multiclass']['macro_f1']
    ax.axhline(y=macro, color=ORANGE, linestyle='--',
               linewidth=1.8, label=f'Macro F1 = {macro:.3f}')
    ax.axhline(y=0.9, color=GRAY, linestyle=':', linewidth=1.2,
               alpha=0.6, label='0.9 reference line')

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0.6, 1.02)
    ax.set_title(f'Per-Subtype F1 — SLM-Guard Multi-Task Classification\nModernBERT-large + LoRA (Macro F1 = {macro:.3f})')

    blue_patch   = mpatches.Patch(color=BLUE,   label='Existing SE subtypes')
    purple_patch = mpatches.Patch(color=PURPLE, label='Novel subtypes (5 new)')
    ax.legend(handles=[blue_patch, purple_patch,
              mpatches.Patch(color=ORANGE, label=f'Macro F1 = {macro:.3f}')],
              loc='lower right')

    save(fig, 'fig3_per_subtype_f1')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Latency vs F1 scatter (efficiency frontier)
# ══════════════════════════════════════════════════════════════════════════════
def fig_latency_vs_f1():
    baseline  = json.load(open(CKPT_V1 / 'baseline_results.json'))
    guardrail = json.load(open(CKPT_V1 / 'guardrail_comparison.json'))
    slmguard  = json.load(open(CKPT_MB / 'eval_results.json'))

    systems = [
        ('Keyword Filter',        baseline['keyword_filter']['f1'],       0.105,   GRAY,   80),
        ('TF-IDF + LR',           baseline['tfidf_lr']['f1'],             0.131,   GRAY,   80),
        ('OpenAI Moderation',     baseline['openai_moderation']['f1'],    35.4,    GRAY,   80),
        ('Llama-3.1-8B (0-shot)', guardrail['llama31_8b_zeroshot']['f1'], 67.9,    GRAY,   80),
        ('Gemma-2-2B (0-shot)',   guardrail['gemma2_2b_zeroshot']['f1'],  93.3,    GRAY,   80),
        ('LlamaGuard 3-8B',       guardrail['llamaguard3_8b']['f1'],      76.6,    GRAY,   80),
        ('SLM-Guard (ours)',       slmguard['binary']['f1'],  slmguard['latency_ms']['p50'],  BLUE,   180),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, f1, lat, color, size in systems:
        ax.scatter(lat, f1, c=color, s=size, zorder=5,
                   edgecolors='white', linewidths=1.0)
        offset_x = 3 if 'SLM' not in name else -25
        offset_y = 0.01 if 'SLM' not in name else -0.025
        ha = 'left' if 'SLM' not in name else 'right'
        ax.annotate(name,
                    xy=(lat, f1),
                    xytext=(lat + offset_x, f1 + offset_y),
                    fontsize=8,
                    color=color if color == BLUE else '#374151',
                    fontweight='bold' if color == BLUE else 'normal',
                    ha=ha)

    ax.set_xlabel('Inference Latency — p50 (ms)\nLower is better →')
    ax.set_ylabel('Binary F1 Score\n← Higher is better')
    ax.set_title('Accuracy vs Speed Trade-off\nIdeal system: top-left corner')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.08)

    # Ideal region annotation
    ax.annotate('Ideal region', xy=(1, 1.0), xytext=(0.5, 1.04),
                fontsize=9, color='green', alpha=0.7,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

    blue_dot = plt.scatter([], [], c=BLUE, s=120, label='SLM-Guard (ours)')
    gray_dot = plt.scatter([], [], c=GRAY, s=80,  label='Existing systems')
    ax.legend(handles=[blue_dot, gray_dot], loc='lower right')

    save(fig, 'fig4_latency_vs_f1')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — F1 vs Hard-neg FPR (the precision-safety tradeoff)
# ══════════════════════════════════════════════════════════════════════════════
def fig_f1_vs_hn_fpr():
    baseline  = json.load(open(CKPT_V1 / 'baseline_results.json'))
    guardrail = json.load(open(CKPT_V1 / 'guardrail_comparison.json'))
    slmguard  = json.load(open(CKPT_MB / 'eval_results.json'))

    systems = [
        ('Keyword Filter',        baseline['keyword_filter']['f1'],       baseline['keyword_filter']['hn_fpr']),
        ('OpenAI Moderation',     baseline['openai_moderation']['f1'],    baseline['openai_moderation']['hn_fpr']),
        ('LlamaGuard 3-8B',       guardrail['llamaguard3_8b']['f1'],      guardrail['llamaguard3_8b']['hn_fpr']),
        ('Llama-3.1-8B (0-shot)', guardrail['llama31_8b_zeroshot']['f1'], guardrail['llama31_8b_zeroshot']['hn_fpr']),
        ('Gemma-2-2B (0-shot)',   guardrail['gemma2_2b_zeroshot']['f1'],  guardrail['gemma2_2b_zeroshot']['hn_fpr']),
        ('TF-IDF + LR',           baseline['tfidf_lr']['f1'],             baseline['tfidf_lr']['hn_fpr']),
        ('SLM-Guard (ours)',       slmguard['binary']['f1'],               slmguard['hn_fpr']),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, f1, hn_fpr in systems:
        color = BLUE if 'SLM' in name else GRAY
        size  = 180   if 'SLM' in name else 80
        ax.scatter(hn_fpr, f1, c=color, s=size, zorder=5,
                   edgecolors='white', linewidths=1.0)
        ax.annotate(name, xy=(hn_fpr, f1),
                    xytext=(hn_fpr + 0.005, f1 - 0.025),
                    fontsize=8,
                    color=BLUE if 'SLM' in name else '#374151',
                    fontweight='bold' if 'SLM' in name else 'normal')

    ax.axvline(x=0.05, color='green', linestyle='--', linewidth=1.5,
               alpha=0.7, label='5% hard-neg FPR target')
    ax.set_xlabel('Hard Negative FPR (false alarm rate on tricky benign inputs)\nLower is better →')
    ax.set_ylabel('Binary F1 Score\n← Higher is better')
    ax.set_title('Detection Performance vs False Alarm Rate on Hard Negatives\nIdeal: top-left (high F1, low false alarms on legitimate users)')
    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(0, 1.08)

    blue_dot = plt.scatter([], [], c=BLUE, s=120, label='SLM-Guard (ours)')
    gray_dot = plt.scatter([], [], c=GRAY, s=80,  label='Existing systems')
    ax.legend(handles=[blue_dot, gray_dot,
              plt.Line2D([0],[0], color='green', linestyle='--', label='5% target')],
              loc='lower right')

    save(fig, 'fig5_f1_vs_hn_fpr')


if __name__ == '__main__':
    # Check required files exist
    required = [
        CKPT_V1 / 'baseline_results.json',
        CKPT_V1 / 'guardrail_comparison.json',
        CKPT_MB / 'eval_results.json',
    ]
    for f in required:
        if not f.exists():
            print(f"Missing: {f}")
            exit(1)

    print("Generating figures...")
    fig_guardrail_comparison()
    fig_fnr_gap()
    fig_per_subtype()
    fig_latency_vs_f1()
    fig_f1_vs_hn_fpr()
    print(f"\nAll figures saved to {OUT_DIR.resolve()}")
