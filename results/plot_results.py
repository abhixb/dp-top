"""Plot GVL prediction results: predicted vs ground truth task completion."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_FILE = Path("results/Qwen_Qwen3-VL-8B-Instruct_2026-03-10T20-21-35.332089_predictions.jsonl")
OUT_DIR = Path("results/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
records = []
with open(RESULTS_FILE) as f:
    for line in f:
        records.append(json.loads(line))

num_eps = len(records)
print(f"Loaded {num_eps} episodes")

# --- Plot 1: Grid of per-episode predicted vs ground truth ---
cols = 5
rows = (num_eps + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
axes = axes.flatten()

for i, rec in enumerate(records):
    ax = axes[i]
    ep_idx = rec["eval_episode"]["episode_index"]
    gt_shuffled = rec["eval_episode"]["shuffled_frames_approx_completion_rates"]
    pred = rec["predicted_percentages"]
    voc = rec["metrics"]["voc"]

    paired = sorted(zip(gt_shuffled, pred))
    gt_sorted = [p[0] for p in paired]
    pred_sorted = [p[1] for p in paired]

    x = np.arange(len(gt_sorted))
    ax.plot(x, gt_sorted, 'o-', label='GT', color='#2196F3', linewidth=1.5, markersize=3)
    ax.plot(x, pred_sorted, 's--', label='Pred', color='#FF5722', linewidth=1.5, markersize=3)
    ax.fill_between(x, gt_sorted, pred_sorted, alpha=0.12, color='gray')
    ax.set_title(f'Ep {ep_idx} (VOC={voc:.2f})', fontsize=9, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

# Hide unused subplots
for j in range(num_eps, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('GVL Predictions per Episode: Qwen3-VL-8B (4-bit) on NYU Door Opening',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / "pred_vs_gt_grid.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'pred_vs_gt_grid.png'}")

# --- Plot 2: Scatter plot — all predictions vs ground truth ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
all_gt = []
all_pred = []
for rec in records:
    all_gt.extend(rec["eval_episode"]["shuffled_frames_approx_completion_rates"])
    all_pred.extend(rec["predicted_percentages"])

ax2.scatter(all_gt, all_pred, alpha=0.5, s=40, c='#4CAF50', edgecolors='black', linewidth=0.3)
ax2.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect prediction')
corr = np.corrcoef(all_gt, all_pred)[0, 1]
ax2.set_xlabel('Ground Truth Completion %', fontsize=12)
ax2.set_ylabel('Predicted Completion %', fontsize=12)
ax2.set_title(f'All Episodes Scatter (n={len(all_gt)}, r={corr:.3f})', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(-5, 105)
ax2.set_ylim(-5, 105)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(OUT_DIR / "scatter_gt_vs_pred.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'scatter_gt_vs_pred.png'}")

# --- Plot 3: VOC distribution ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 4.5))

voc_scores = [r["metrics"]["voc"] for r in records]
ep_labels = [f"Ep {r['eval_episode']['episode_index']}" for r in records]
mean_voc = np.mean(voc_scores)

# Bar chart
colors = ['#4CAF50' if v >= mean_voc else '#FF5722' for v in voc_scores]
bars = ax3a.bar(range(num_eps), voc_scores, color=colors, edgecolor='black', linewidth=0.3)
ax3a.axhline(mean_voc, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {mean_voc:.3f}')
ax3a.set_xticks(range(num_eps))
ax3a.set_xticklabels(ep_labels, rotation=45, ha='right', fontsize=7)
ax3a.set_ylabel('VOC Score', fontsize=11)
ax3a.set_title('VOC per Episode', fontsize=12, fontweight='bold')
ax3a.set_ylim(0, 1.05)
ax3a.legend(fontsize=9)
ax3a.grid(True, axis='y', alpha=0.3)

# Histogram
ax3b.hist(voc_scores, bins=10, color='#2196F3', edgecolor='black', alpha=0.7)
ax3b.axvline(mean_voc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_voc:.3f}')
ax3b.axvline(np.median(voc_scores), color='orange', linestyle=':', linewidth=2, label=f'Median: {np.median(voc_scores):.3f}')
ax3b.set_xlabel('VOC Score', fontsize=11)
ax3b.set_ylabel('Count', fontsize=11)
ax3b.set_title('VOC Distribution', fontsize=12, fontweight='bold')
ax3b.legend(fontsize=9)
ax3b.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(OUT_DIR / "voc_scores.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'voc_scores.png'}")

# --- Plot 4: Average predicted vs GT curve (aggregated) ---
fig4, ax4 = plt.subplots(figsize=(7, 5))
all_gt_sorted = []
all_pred_sorted = []
for rec in records:
    gt = rec["eval_episode"]["shuffled_frames_approx_completion_rates"]
    pred = rec["predicted_percentages"]
    paired = sorted(zip(gt, pred))
    all_gt_sorted.append([p[0] for p in paired])
    all_pred_sorted.append([p[1] for p in paired])

all_gt_sorted = np.array(all_gt_sorted)
all_pred_sorted = np.array(all_pred_sorted)
mean_gt = all_gt_sorted.mean(axis=0)
mean_pred = all_pred_sorted.mean(axis=0)
std_pred = all_pred_sorted.std(axis=0)

x = np.arange(len(mean_gt))
ax4.plot(x, mean_gt, 'o-', label='Mean GT', color='#2196F3', linewidth=2, markersize=5)
ax4.plot(x, mean_pred, 's-', label='Mean Predicted', color='#FF5722', linewidth=2, markersize=5)
ax4.fill_between(x, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color='#FF5722', label='Pred std')
ax4.set_xlabel('Frame (sorted by GT completion)', fontsize=12)
ax4.set_ylabel('Task Completion %', fontsize=12)
ax4.set_title(f'Average Across {num_eps} Episodes (Mean VOC={mean_voc:.3f})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(-5, 105)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig(OUT_DIR / "avg_pred_vs_gt.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'avg_pred_vs_gt.png'}")

print(f"\nAll plots saved! Mean VOC = {mean_voc:.3f}")
