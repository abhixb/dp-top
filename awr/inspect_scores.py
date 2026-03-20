"""Generate plots to sanity-check TOPReward scores and advantage weights.

Usage:
    python -m awr.inspect_scores
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from awr.config import OUTPUT_DIR

# ─── Dark theme ───
BG = "#0a0b0f"
SURFACE = "#12131a"
TEXT = "#e8e9ed"
TEXT_DIM = "#6b6f82"
GRID = "#1e2030"
ACCENT = "#22d3a7"
SECONDARY = "#6366f1"
WARNING = "#f59e0b"
DANGER = "#ef4444"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_DIM,
    "text.color": TEXT,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": GRID,
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
})

SAVE_KW = dict(dpi=150, facecolor=BG, bbox_inches="tight")


def load_advantages(adv_path: Path) -> dict:
    with open(adv_path) as f:
        return json.load(f)


def load_scores(scores_dir: Path) -> list[dict]:
    files = sorted(scores_dir.glob("episode_*.json"))
    episodes = []
    for f in files:
        with open(f) as fh:
            episodes.append(json.load(fh))
    episodes.sort(key=lambda e: e["episode_id"])
    return episodes


def plot_progress_curves(episodes: list[dict], adv_data: dict, out_dir: Path) -> None:
    """All episodes' normalized progress curves, color-coded by VOC."""
    fig, ax = plt.subplots(figsize=(12, 6))

    vocs = [ep.get("voc", 0.0) for ep in episodes]
    vmin, vmax = min(vocs), max(vocs)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Use per-frame progress from advantages if available
    adv_episodes = {ep["episode_id"]: ep for ep in adv_data.get("episodes", [])}

    for ep in episodes:
        ep_id = ep["episode_id"]
        voc = ep.get("voc", 0.0)
        color = cmap(norm(voc))

        if ep_id in adv_episodes and "per_frame_progress" in adv_episodes[ep_id]:
            progress = adv_episodes[ep_id]["per_frame_progress"]
            ax.plot(range(len(progress)), progress, color=color, alpha=0.7, linewidth=1.0)
        else:
            # Fall back to scored prefix points
            pl = ep.get("prefix_lengths", [])
            nl = ep.get("normalized", [])
            if pl and nl:
                ax.plot(pl, nl, color=color, alpha=0.7, linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="VOC")
    cbar.ax.yaxis.label.set_color(TEXT_DIM)
    cbar.ax.tick_params(colors=TEXT_DIM)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Progress")
    ax.set_title("Progress Curves (all episodes, colored by VOC)", color=TEXT)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "all_progress_curves.png", **SAVE_KW)
    plt.close(fig)
    print(f"  Saved all_progress_curves.png")


def plot_weight_distribution(adv_data: dict, out_dir: Path) -> None:
    """Histogram of all advantage weights."""
    all_weights = []
    for ep in adv_data.get("episodes", []):
        all_weights.extend(ep.get("weights", []))
    all_weights = np.array(all_weights)

    if len(all_weights) == 0:
        print("  Skipping weight_distribution.png (no weights)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(all_weights, bins=60, color=SECONDARY, edgecolor=SURFACE, alpha=0.85)
    ax.axvline(1.0, color=ACCENT, linestyle="--", linewidth=1.5, label="weight=1.0")

    mean_w = all_weights.mean()
    std_w = all_weights.std()
    ax.set_title(
        f"Advantage Weight Distribution (mean={mean_w:.3f}, std={std_w:.3f})",
        color=TEXT,
    )
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "weight_distribution.png", **SAVE_KW)
    plt.close(fig)
    print(f"  Saved weight_distribution.png")


def plot_weight_heatmap(adv_data: dict, out_dir: Path) -> None:
    """Heatmap: one row per episode, colored by weight at each timestep."""
    adv_episodes = adv_data.get("episodes", [])
    if not adv_episodes:
        print("  Skipping weight_heatmap.png (no episodes)")
        return

    max_len = max(len(ep.get("weights", [])) for ep in adv_episodes)
    n_eps = len(adv_episodes)

    # Build matrix, pad with NaN
    matrix = np.full((n_eps, max_len), np.nan)
    ep_labels = []
    for i, ep in enumerate(adv_episodes):
        w = ep.get("weights", [])
        matrix[i, : len(w)] = w
        ep_labels.append(f"Ep {ep['episode_id']}")

    fig, ax = plt.subplots(figsize=(14, max(4, n_eps * 0.3 + 1)))

    # Diverging colormap centered at 1.0
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    # Center at 1.0
    abs_max = max(abs(vmin - 1.0), abs(vmax - 1.0))
    norm = mcolors.TwoSlopeNorm(vmin=1.0 - abs_max, vcenter=1.0, vmax=1.0 + abs_max)

    im = ax.imshow(
        matrix, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax, label="Weight")
    cbar.ax.yaxis.label.set_color(TEXT_DIM)
    cbar.ax.tick_params(colors=TEXT_DIM)

    ax.set_yticks(range(n_eps))
    ax.set_yticklabels(ep_labels, fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_title("Advantage Weights per Episode", color=TEXT)

    fig.savefig(out_dir / "weight_heatmap.png", **SAVE_KW)
    plt.close(fig)
    print(f"  Saved weight_heatmap.png")


def plot_dataset_summary(episodes: list[dict], out_dir: Path) -> None:
    """Bar chart of VOC per episode."""
    ep_ids = [ep["episode_id"] for ep in episodes]
    vocs = [ep.get("voc", 0.0) for ep in episodes]

    fig, ax = plt.subplots(figsize=(max(8, len(ep_ids) * 0.4), 5))

    colors = []
    for v in vocs:
        if v > 0.8:
            colors.append(ACCENT)
        elif v > 0.5:
            colors.append(WARNING)
        else:
            colors.append(DANGER)

    bars = ax.bar(range(len(ep_ids)), vocs, color=colors, edgecolor=SURFACE, width=0.8)
    mean_voc = np.mean(vocs)
    ax.axhline(mean_voc, color=TEXT_DIM, linestyle="--", linewidth=1.0, label=f"mean={mean_voc:.3f}")

    ax.set_xticks(range(len(ep_ids)))
    ax.set_xticklabels([str(e) for e in ep_ids], fontsize=8, rotation=45)
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("VOC")
    ax.set_title(
        f"Dataset Summary ({len(episodes)} episodes)",
        color=TEXT,
    )
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_dir / "dataset_summary.png", **SAVE_KW)
    plt.close(fig)
    print(f"  Saved dataset_summary.png")


def run_inspect() -> None:
    scores_dir = OUTPUT_DIR / "scores"
    adv_path = OUTPUT_DIR / "advantages" / "advantages.json"
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    episodes = load_scores(scores_dir)
    if not episodes:
        print(f"No scored episodes found in {scores_dir}")
        return

    adv_data = {}
    if adv_path.exists():
        adv_data = load_advantages(adv_path)
    else:
        print(f"Warning: {adv_path} not found. Run compute_advantages first for weight plots.")

    print(f"Generating plots for {len(episodes)} episodes...")

    plot_progress_curves(episodes, adv_data, plots_dir)
    plot_weight_distribution(adv_data, plots_dir)
    plot_weight_heatmap(adv_data, plots_dir)
    plot_dataset_summary(episodes, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


def main():
    run_inspect()


if __name__ == "__main__":
    main()
