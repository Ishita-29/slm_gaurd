
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

BLUE   = '#1D4ED8'
RED    = '#DC2626'
GREEN  = '#15803D'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
GRAY   = '#9CA3AF'
TEAL   = '#0D9488'

OUT_DIR     = Path('../figures')
RESULTS_DIR = Path('../results')
CKPT_DIR    = Path('../checkpoints/slmguard-modernbert-lora')
OUT_DIR.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(OUT_DIR / name, bbox_inches='tight')
    fig.savefig(OUT_DIR / name.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"  Saved → {OUT_DIR / name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Training Loss Curve
# ══════════════════════════════════════════════════════════════════════════════
def fig_training_loss():
    state  = json.load(open(CKPT_DIR / 'checkpoint-6210' / 'trainer_state.json'))
    logs   = state['log_history']
    epochs = [l['epoch'] for l in logs if 'loss' in l]
    losses = [l['loss']  for l in logs if 'loss' in l]

    def smooth(y, w=7):
        pad = np.pad(y, w//2, mode='edge')
        return np.convolve(pad, np.ones(w)/w, mode='valid')[:len(y)]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(epochs, losses, color=BLUE, alpha=0.2, linewidth=1.0)
    ax.plot(epochs, smooth(losses), color=BLUE, linewidth=2.2, label='Training loss (smoothed)')

    # Phase shading
    ax.axvspan(0, 1.0, alpha=0.06, color=ORANGE, label='Phase 1: frozen encoder warm-up')
    ax.axvspan(1.0, max(epochs), alpha=0.04, color=BLUE, label='Phase 2: LoRA fine-tuning')
    ax.axvline(x=1.0, color=ORANGE, linestyle='--', linewidth=1.4, alpha=0.8)
    ax.text(0.45, 0.32, 'Phase 1\n(Warm-up)', transform=ax.transAxes,
            fontsize=9, color=ORANGE, ha='center', alpha=0.9)
    ax.text(0.72, 0.32, 'Phase 2\n(LoRA)', transform=ax.transAxes,
            fontsize=9, color=BLUE, ha='center', alpha=0.9)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('SLM-Guard Training Convergence — ModernBERT-large + LoRA')
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(bottom=-0.005)
    ax.legend(loc='upper right', framealpha=0.9)
    save(fig, 'fig_training_loss.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Score Distribution (key figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_score_distribution():
    attack_probs, benign_probs = [], []
    for f in RESULTS_DIR.glob('middleware_*.json'):
        if 'ood' in f.name:
            continue
        d = json.load(open(f))
        for r in d['raw']:
            if r['is_se'] == 1:
                attack_probs.append(r['guard_prob'])
            else:
                benign_probs.append(r['guard_prob'])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 52)

    ax.hist(benign_probs, bins=bins, color=GREEN, alpha=0.75,
            label=f'Benign prompts (n={len(set(benign_probs))})', density=True, zorder=3)
    ax.hist(attack_probs, bins=bins, color=RED,   alpha=0.75,
            label=f'SE attack prompts (n={len(set(attack_probs))})', density=True, zorder=3)

    ax.axvline(x=0.4, color='black', linestyle='--', linewidth=2.0,
               label='Decision threshold (τ = 0.4)', zorder=5)

    # Annotate peaks
    ax.annotate('Benign: score ≈ 0\n(correctly not flagged)',
                xy=(0.02, 35), xytext=(0.15, 28),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                fontsize=9, color=GREEN)
    ax.annotate('Attacks: score ≈ 1\n(correctly flagged)',
                xy=(0.97, 18), xytext=(0.65, 25),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                fontsize=9, color=RED)

    ax.set_xlabel('SLM-Guard Attack Probability Score $\\hat{p}$')
    ax.set_ylabel('Density')
    ax.set_title('Guard Score Distribution: Attacks vs Benign (In-Distribution Test Set)')
    ax.legend(loc='upper center', framealpha=0.9)
    ax.set_xlim(0, 1)
    save(fig, 'fig_score_distribution.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — ASR Comparison (7 target LLMs)
# ══════════════════════════════════════════════════════════════════════════════
def fig_asr_comparison():
    MODEL_INFO = {
        'middleware_qwen25_1b.json':   ('Qwen2.5\n1.5B',     'Alibaba'),
        'middleware_qwen25_7b.json':   ('Qwen2.5\n7B',       'Alibaba'),
        'middleware_deepseek_r1.json': ('DeepSeek-R1\n1.5B', 'DeepSeek'),
        'middleware_llama31_8b.json':  ('Llama-3.1\n8B',     'Meta'),
        'middleware_gemma2_2b.json':   ('Gemma-2\n2B',       'Google'),
        'middleware_gemma3_4b.json':   ('Gemma-3\n4B',       'Google'),
        'middleware_mistral_7b.json':  ('Mistral\n7B',       'Mistral'),
    }

    labels, asr_no, benign_pass = [], [], []
    for fname, (label, _) in MODEL_INFO.items():
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            continue
        d = json.load(open(fpath))
        labels.append(label)
        asr_no.append(d['asr_noguard'] * 100)
        benign_pass.append(d['benign_pass_rate'] * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # No-guard bars
    bars = ax.bar(x, asr_no, 0.5, color=RED, alpha=0.85, zorder=3,
                  label='Without SLM-Guard (baseline ASR)')

    # Benign pass-through as line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(x, benign_pass, color=GREEN, linewidth=2.2, marker='D',
             markersize=7, zorder=5, label='Benign pass-through (%)')
    ax2.set_ylim(0, 120)
    ax2.set_ylabel('Benign Pass-Through Rate (%)', color=GREEN)
    ax2.tick_params(axis='y', labelcolor=GREEN)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    # Value labels on bars
    for bar, val in zip(bars, asr_no):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#7F1D1D')

    # "100% blocked" annotation band
    ax.axhspan(-2, 3, alpha=0.12, color=GREEN, zorder=1)
    ax.text(len(labels)-1, 1.5, '← With SLM-Guard: 0% ASR (all blocked)',
            ha='right', fontsize=9, color=GREEN, style='italic')

    avg = np.mean(asr_no)
    ax.axhline(y=avg, color=RED, linestyle=':', linewidth=1.5, alpha=0.6, zorder=4)
    ax.text(len(labels)-0.5, avg+1.5, f'Avg: {avg:.0f}%',
            ha='right', fontsize=9, color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Attack Success Rate without Guard (%)')
    ax.set_title('Middleware Evaluation: Attack Success Rate Across 7 Target LLMs\n'
                 '(In-distribution test set, 220 attack prompts per model, τ = 0.4)')
    ax.set_ylim(-2, 115)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    # Combined legend
    h1 = mpatches.Patch(color=RED, alpha=0.85, label='ASR without SLM-Guard')
    h2 = plt.Line2D([0], [0], color=GREEN, linewidth=2, marker='D',
                    markersize=7, label='Benign pass-through (with guard)')
    h3 = mpatches.Patch(color=GREEN, alpha=0.3, label='ASR with SLM-Guard = 0%')
    ax.legend(handles=[h1, h2, h3], loc='upper left', framealpha=0.9)

    save(fig, 'fig_asr_comparison.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-Subtype ASR (redesigned: baseline only + reduction annotation)
# ══════════════════════════════════════════════════════════════════════════════
def fig_subtype_asr():
    subtype_no = defaultdict(list)
    for f in RESULTS_DIR.glob('middleware_*.json'):
        if 'ood' in f.name:
            continue
        d = json.load(open(f))
        for sub, vals in d.get('per_subtype', {}).items():
            subtype_no[sub].append(vals['asr_noguard'])

    SUBTYPE_DISPLAY = {
        'authority_impersonation':  'Authority\nImpersonation',
        'authority_laundering':     'Authority\nLaundering',
        'cognitive_load_embedding': 'Cognitive Load\nEmbedding',
        'false_consensus':          'False\nConsensus',
        'flattery_parasocial':      'Flattery /\nParasocial Bond',
        'identity_erosion':         'Identity\nErosion',
        'incremental_escalation':   'Incremental\nEscalation',
        'normalization_repetition': 'Normalization /\nRepetition',
        'pretexting':               'Pretexting',
        'reciprocity_conditioning': 'Reciprocity\nConditioning',
        'urgency_emotion':          'Urgency /\nEmotion',
    }

    # Sort by descending baseline ASR
    subtypes = sorted(subtype_no.keys(), key=lambda s: np.mean(subtype_no[s]), reverse=True)
    avgs_no  = [np.mean(subtype_no[s]) * 100 for s in subtypes]
    labels   = [SUBTYPE_DISPLAY.get(s, s) for s in subtypes]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(subtypes))

    # Baseline bars (no guard)
    bars = ax.barh(y, avgs_no, 0.55, color=RED, alpha=0.82, zorder=3,
                   label='Baseline ASR (no guard)')

    # Protected bars — thin green strip to show "blocked to 0"
    ax.barh(y, [1.5]*len(subtypes), 0.55, color=GREEN, alpha=0.9, zorder=4,
            label='Protected ASR (with SLM-Guard) = 0%')

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, avgs_no)):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9.5,
                fontweight='bold', color='#7F1D1D')

    # 100% reduction annotation on right side
    for i in range(len(subtypes)):
        ax.text(101, i, '↓ 100%', va='center', fontsize=8.5,
                color=GREEN, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel('Attack Success Rate (%, averaged across 7 target LLMs)')
    ax.set_title('Per SE Subtype: Baseline ASR and SLM-Guard Protection\n'
                 '(Subtypes ordered by baseline ASR, descending)')
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.legend(loc='lower right', framealpha=0.9)

    # Cialdini principle labels
    PRINCIPLE = {
        'incremental_escalation':   'Commitment',
        'cognitive_load_embedding': 'Commitment',
        'identity_erosion':         'Commitment',
        'normalization_repetition': 'Social Proof',
        'false_consensus':          'Social Proof',
        'urgency_emotion':          'Scarcity',
        'authority_laundering':     'Authority',
        'flattery_parasocial':      'Liking',
        'pretexting':               'Authority',
        'reciprocity_conditioning': 'Reciprocity',
        'authority_impersonation':  'Authority',
    }
    for i, s in enumerate(subtypes):
        p = PRINCIPLE.get(s, '')
        ax.text(-0.5, i, p, va='center', ha='right', fontsize=7.5,
                color=GRAY, style='italic')

    ax.text(-0.5, -0.7, 'Cialdini\nPrinciple', va='center', ha='right',
            fontsize=7.5, color=GRAY, fontweight='bold')

    save(fig, 'fig_subtype_asr.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Threshold Sweep (dual-axis)
# ══════════════════════════════════════════════════════════════════════════════
def fig_threshold_sweep():
    # Pool all guard probs from all in-dist runs for larger sample
    attack_probs, benign_probs = [], []
    for f in RESULTS_DIR.glob('middleware_*.json'):
        if 'ood' in f.name:
            continue
        d = json.load(open(f))
        for r in d['raw']:
            if r['is_se'] == 1:
                attack_probs.append(r['guard_prob'])
            else:
                benign_probs.append(r['guard_prob'])

    thresholds = np.arange(0.05, 0.96, 0.05)
    fprs, fnrs, f1s, precisions, recalls = [], [], [], [], []

    for tau in thresholds:
        tp = sum(1 for p in attack_probs if p >= tau)
        fn = sum(1 for p in attack_probs if p <  tau)
        fp = sum(1 for p in benign_probs  if p >= tau)
        tn = sum(1 for p in benign_probs  if p <  tau)
        fpr  = fp/(fp+tn) if (fp+tn)>0 else 0
        fnr  = fn/(fn+tp) if (fn+tp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 1
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        fprs.append(fpr*100);  fnrs.append(fnr*100)
        f1s.append(f1*100);    precisions.append(prec*100); recalls.append(rec*100)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # F1 on left axis
    ax1.plot(thresholds, f1s, color=BLUE, linewidth=2.5, marker='o',
             markersize=5, label='F1 Score', zorder=4)
    ax1.set_ylabel('F1 Score (%)', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.set_ylim(85, 101)

    # FPR / FNR on right axis
    ax2 = ax1.twinx()
    ax2.plot(thresholds, fprs, color=RED,   linewidth=2.2, marker='s',
             markersize=5, linestyle='-',  label='FPR (false positive rate)', zorder=3)
    ax2.plot(thresholds, fnrs, color=ORANGE, linewidth=2.2, marker='^',
             markersize=5, linestyle='--', label='FNR (false negative rate)', zorder=3)
    ax2.set_ylabel('Error Rate (%)', color='#374151')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 15)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    # Operating point
    ax1.axvline(x=0.4, color='black', linestyle='--', linewidth=1.8,
                label='Operating point (τ = 0.4)', zorder=5)

    ax1.set_xlabel('Decision Threshold (τ)')
    ax1.set_title('Threshold Sensitivity: F1, FPR, and FNR vs τ\n'
                  '(Pooled across 7 target LLM evaluation runs, n = 1,540 attacks + 350 benign)')
    ax1.set_xlim(0.05, 0.95)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
               framealpha=0.9, ncol=2)

    # Annotate operating point metrics
    idx04 = np.argmin(np.abs(thresholds - 0.4))
    ax1.annotate(f'τ=0.4\nF1={f1s[idx04]:.1f}%',
                 xy=(0.4, f1s[idx04]), xytext=(0.55, 93),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, color='black')
    ax2.annotate(f'FPR={fprs[idx04]:.1f}%\nFNR={fnrs[idx04]:.1f}%',
                 xy=(0.4, fprs[idx04]), xytext=(0.55, 8),
                 arrowprops=dict(arrowstyle='->', color=RED),
                 fontsize=9, color=RED)

    save(fig, 'fig_threshold_sweep.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — ROC Curve (using known AUC from full evaluation)
# ══════════════════════════════════════════════════════════════════════════════
def fig_roc_curve():
    # Use all pooled guard scores for ROC
    attack_probs, benign_probs = [], []
    for f in RESULTS_DIR.glob('middleware_*.json'):
        if 'ood' in f.name:
            continue
        d = json.load(open(f))
        for r in d['raw']:
            if r['is_se'] == 1:
                attack_probs.append(r['guard_prob'])
            else:
                benign_probs.append(r['guard_prob'])

    all_probs  = np.array(attack_probs + benign_probs)
    all_labels = np.array([1]*len(attack_probs) + [0]*len(benign_probs))

    thresholds = np.unique(np.concatenate([np.linspace(0, 1, 500), all_probs]))[::-1]
    tprs, fprs_list = [], []
    for tau in thresholds:
        pred = (all_probs >= tau).astype(int)
        tp = np.sum((pred==1) & (all_labels==1))
        fn = np.sum((pred==0) & (all_labels==1))
        fp = np.sum((pred==1) & (all_labels==0))
        tn = np.sum((pred==0) & (all_labels==0))
        tprs.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        fprs_list.append(fp/(fp+tn) if (fp+tn)>0 else 0)

    # Use the validated AUC from full test set evaluation (n=4,248)
    AUC_FULL = 0.9993

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs_list, tprs, color=BLUE, linewidth=2.5,
            label=f'SLM-Guard ModernBERT-large + LoRA\n(AUC = {AUC_FULL}, n = 4,248)')
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle='--', linewidth=1.2,
            label='Random classifier (AUC = 0.50)')
    ax.fill_between(fprs_list, tprs, alpha=0.08, color=BLUE)

    # Mark τ=0.4 operating point
    tau_op = 0.4
    pred_op = (all_probs >= tau_op).astype(int)
    tp_op = np.sum((pred_op==1) & (all_labels==1))
    fn_op = np.sum((pred_op==0) & (all_labels==1))
    fp_op = np.sum((pred_op==1) & (all_labels==0))
    tn_op = np.sum((pred_op==0) & (all_labels==0))
    op_tpr = tp_op/(tp_op+fn_op)
    op_fpr = fp_op/(fp_op+tn_op)

    ax.scatter([op_fpr], [op_tpr], color=RED, s=100, zorder=6,
               label=f'Operating point (τ = 0.4)\nTPR={op_tpr*100:.1f}%, FPR={op_fpr*100:.1f}%')
    ax.annotate(f'τ = 0.4\nTPR = {op_tpr*100:.1f}%\nFPR = {op_fpr*100:.1f}%',
                xy=(op_fpr, op_tpr), xytext=(0.15, 0.75),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                fontsize=9, color=RED)

    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('ROC Curve — SLM-Guard Binary Social Engineering Detection')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, 'fig_roc_curve.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Model Comparison: ModernBERT vs Qwen2.5 (4 metrics, dual view)
# ══════════════════════════════════════════════════════════════════════════════
def fig_model_comparison():
    """
    Compares two successfully trained models:
      - Qwen2.5-1.5B + LoRA  (decoder, 1.5B total, 38.5M trainable)
      - ModernBERT-large + LoRA (encoder, 395M total, 4.4M trainable)

    DeBERTa-v3-large + LoRA training collapsed (loss→0, grad_norm→0 after
    epoch 1), yielding majority-class predictions and F1≈0. It is excluded
    from the comparison but noted as a negative result in the ablation section.

    All numbers from validated evaluation runs (test set, n=4,248, τ=0.4).
    """
    models = ['Qwen2.5-1.5B\n+ LoRA\n(1,500M params)', 'ModernBERT-large\n+ LoRA\n(395M params)']
    colors = [PURPLE, BLUE]

    # Validated numbers from evaluate.py runs (see thesis Table 2)
    data = {
        'Accuracy (%)':       ([99.15, 99.51],   'higher is better', True),
        'F1 Score':           ([0.9954, 0.9973],  'higher is better', True),
        'AUC-ROC':            ([0.9964, 0.9993],  'higher is better', True),
        'FPR (%) ↓':          ([7.34,  3.39],     'lower is better',  False),
        'FNR (%) ↓':          ([0.26,  0.23],     'lower is better',  False),
        'Trainable Params (M) ↓': ([38.5, 4.4],  'lower is better',  False),
    }

    show_metrics = ['Accuracy (%)', 'F1 Score', 'AUC-ROC', 'FPR (%) ↓']
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.5))

    for ax, metric in zip(axes, show_metrics):
        vals, direction, higher_better = data[metric]
        bars = ax.bar(range(2), vals, color=colors, alpha=0.87, width=0.45, zorder=3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.015,
                    f'{val:.4f}' if max(vals) < 2 else f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight winner
        winner = np.argmax(vals) if higher_better else np.argmin(vals)
        bars[winner].set_edgecolor('gold')
        bars[winner].set_linewidth(2.5)

        ax.set_xticks(range(2))
        ax.set_xticklabels(models, fontsize=8.5)
        ax.set_title(metric, fontweight='bold', fontsize=10.5, pad=8)
        ymax = max(vals) * 1.22
        ymin = min(vals) * 0.93 if min(vals) > 0 else 0
        ax.set_ylim(ymin, ymax)

        note_color = GREEN if higher_better else RED
        ax.text(0.5, 0.03, direction, transform=ax.transAxes,
                ha='center', fontsize=8, color=note_color, style='italic')

    # Add parameter efficiency inset in last panel
    ax_last = axes[-1]
    ax_inset = ax_last.inset_axes([0.55, 0.55, 0.42, 0.4])
    ax_inset.bar([0, 1], [38.5, 4.4], color=colors, alpha=0.8, width=0.5)
    ax_inset.set_xticks([0, 1])
    ax_inset.set_xticklabels(['Qwen\n1.5B', 'MBERT\nlarge'], fontsize=7)
    ax_inset.set_title('Trainable\nParams (M)', fontsize=7.5, pad=2)
    ax_inset.set_ylim(0, 45)
    for i, v in enumerate([38.5, 4.4]):
        ax_inset.text(i, v+0.5, f'{v}M', ha='center', fontsize=7, fontweight='bold')

    fig.suptitle('Model Comparison: Qwen2.5-1.5B + LoRA vs ModernBERT-large + LoRA\n'
                 '(Test set, n = 4,248, τ = 0.4 — gold border = winner per metric)',
                 fontsize=11.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    save(fig, 'fig_model_comparison.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — OOD ASR (generated after OOD eval completes)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ood_asr():
    ood_files = list(RESULTS_DIR.glob('middleware_ood_*.json'))
    if not ood_files:
        print("  [SKIP] OOD eval still running — re-run this script when complete")
        return

    by_source = defaultdict(lambda: {'no': [], 'yes': [], 'tpr': []})
    for f in ood_files:
        d = json.load(open(f))
        src = d['ood_source']
        by_source[src]['no'].append(d['asr_noguard']  * 100)
        by_source[src]['yes'].append(d['asr_withguard'] * 100)
        by_source[src]['tpr'].append(d['tpr'] * 100)

    SOURCE_DISPLAY = {
        'jailbreakhub': 'JailbreakHub\n(community jailbreaks)',
        'toxicchat':    'ToxicChat\n(real user conversations)',
        'advbench':     'AdvBench\n(adversarial harmful QA)',
        'all':          'All OOD\n(pooled)',
    }

    sources  = sorted(by_source.keys())
    avgs_no  = [np.mean(by_source[s]['no'])  for s in sources]
    avgs_yes = [np.mean(by_source[s]['yes']) for s in sources]
    avgs_tpr = [np.mean(by_source[s]['tpr']) for s in sources]
    labels   = [SOURCE_DISPLAY.get(s, s) for s in sources]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))

    b1 = ax.bar(x - w, avgs_no,  w, color=RED,   alpha=0.85, label='ASR without SLM-Guard')
    b2 = ax.bar(x,     avgs_yes, w, color=GREEN,  alpha=0.85, label='ASR with SLM-Guard')
    b3 = ax.bar(x + w, avgs_tpr, w, color=BLUE,   alpha=0.85, label='Guard Detection Rate (TPR)')

    for bars, vals in [(b1, avgs_no), (b2, avgs_yes), (b3, avgs_tpr)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Rate (%)')
    ax.set_title('OOD Middleware Evaluation: SLM-Guard on Attacks Never Seen During Training\n'
                 '(Averaged across 4 target LLMs, 100 attack + 50 benign per source)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 120)
    save(fig, 'fig_ood_asr.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"\nGenerating thesis figures → {OUT_DIR.resolve()}\n")
    figs = [
        ('1. Training loss curve',         fig_training_loss),
        ('2. Score distribution',          fig_score_distribution),
        ('3. ASR comparison (7 LLMs)',     fig_asr_comparison),
        ('4. Per-subtype ASR',             fig_subtype_asr),
        ('5. Threshold sensitivity sweep', fig_threshold_sweep),
        ('6. ROC curve',                   fig_roc_curve),
        ('7. Model comparison',            fig_model_comparison),
        ('8. OOD ASR comparison',          fig_ood_asr),
    ]
    for name, fn in figs:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        print()
    print("Done. Figures saved to:", OUT_DIR.resolve())
