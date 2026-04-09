import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data: highest F1 first
data = [
    ("Benign",       0.967),
    ("Scarcity",     0.930),
    ("Authority",    0.919),
    ("Liking",       0.909),
    ("Reciprocity",  0.862),
    ("Social Proof", 0.825),
    ("Commitment",   0.792),
]

labels, scores = zip(*data)

# Reverse so highest F1 appears at the top of the barh chart
labels = labels[::-1]
scores = scores[::-1]

# Gradient: lightest blue at bottom (lowest F1), darkest at top (highest F1)
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(scores)))

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Plot bars
bars = ax.barh(labels, scores, color=colors, edgecolor='white', height=0.65)

# Subtle horizontal gridlines behind the bars
ax.set_axisbelow(True)
ax.xaxis.grid(True, color='#e0e0e0', linewidth=0.8, linestyle='--')

# X-axis
ax.set_xlim(0.70, 1.02)
ax.set_xlabel('F1 Score', fontweight='bold', labelpad=8)
ax.set_title('Per-Class F1 — Cialdini Group Classification Head',
             fontsize=14, fontweight='bold', pad=16)

# Annotate each bar with its value
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}', va='center', ha='left', fontsize=11)

# Macro average vertical line — label placed INSIDE the plot at the top
macro_avg = 0.886
ax.axvline(x=macro_avg, color='#555555', linestyle='--', linewidth=1.5, alpha=0.85)

# Annotate the macro avg line just above the top bar (y = n_bars - 0.5)
n = len(labels)
ax.text(macro_avg + 0.003, n - 0.55,
        f'Macro avg\n({macro_avg})',
        color='#555555', ha='left', va='top',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('cialdini_f1_scores.png', bbox_inches='tight',
            facecolor='white', dpi=300)
print("Saved: cialdini_f1_scores.png")
