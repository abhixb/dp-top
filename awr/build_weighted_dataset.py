"""Build combined advantage weights: hub demos (fixed) + rollouts (scored).

Hub demos are all clean successes → fixed weight (no need to score).
Rollouts have variance → AWR weights from TOPReward scores.

Usage:
    python -m awr.build_weighted_dataset --round 1
    python -m awr.build_weighted_dataset --round 1 --hub-weight 2.0 --tau 3.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from awr.compute_advantages import interpolate_progress, load_episode_scores
from awr.config_hw import (
    DELTA_MAX,
    HUB_DATASET,
    HUB_DEMO_FIXED_WEIGHT,
    OUTPUT_DIR,
    SUBTRACT_MEAN,
    TAU,
    WEIGHT_CLIP_MIN,
)
from awr.inspect_scores import (
    plot_dataset_summary,
    plot_progress_curves,
    plot_weight_distribution,
    plot_weight_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined AWR weights for hub + rollouts")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--hub-weight", type=float, default=HUB_DEMO_FIXED_WEIGHT)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--delta-max", type=float, default=DELTA_MAX)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def compute_rollout_weights(
    episodes: list[dict],
    tau: float,
    delta_max: float,
    weight_clip_min: float,
    subtract_mean: bool,
) -> dict:
    """Compute per-frame AWR weights for rollout episodes.

    Returns dict with 'episodes' (annotated), 'flat_weights', and 'stats'.
    """
    all_deltas = []
    for ep in episodes:
        progress = interpolate_progress(ep)
        deltas = np.diff(progress, prepend=0.0)
        ep["per_frame_progress"] = progress.tolist()
        ep["deltas"] = deltas.tolist()
        all_deltas.append(deltas)

    # Mean delta from rollouts only
    all_deltas_flat = np.concatenate(all_deltas)
    mean_delta = float(all_deltas_flat.mean())

    for ep in episodes:
        deltas = np.array(ep["deltas"])
        if subtract_mean:
            advantages = deltas - mean_delta
        else:
            advantages = deltas
        weights = np.clip(tau * np.exp(advantages), weight_clip_min, delta_max)
        ep["weights"] = weights.tolist()

    # Flat weights array (indexed by position within rollout dataset)
    # Episodes have start_idx/end_idx relative to their own dataset
    all_weights = np.concatenate([np.array(ep["weights"]) for ep in episodes])

    stats = {
        "mean_weight": round(float(all_weights.mean()), 4),
        "std_weight": round(float(all_weights.std()), 4),
        "min_weight": round(float(all_weights.min()), 4),
        "max_weight": round(float(all_weights.max()), 4),
        "pct_above_1": round(float((all_weights > 1.0).mean()), 4),
        "pct_below_1": round(float((all_weights < 1.0).mean()), 4),
        "mean_delta": round(mean_delta, 6),
    }

    return {
        "episodes": episodes,
        "flat_weights": all_weights,
        "stats": stats,
    }


def main():
    args = parse_args()

    round_dir = OUTPUT_DIR / f"round_{args.round:03d}"
    scores_dir = round_dir / "scores"
    output_dir = args.output or round_dir

    # 1. Get hub dataset size
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading hub dataset metadata: {HUB_DATASET}")
    hub_ds = LeRobotDataset(repo_id=HUB_DATASET)
    hub_size = len(hub_ds)
    del hub_ds
    print(f"  Hub dataset: {hub_size} frames, fixed weight={args.hub_weight}")

    # 2. Load and compute rollout weights
    print(f"Loading rollout scores from: {scores_dir}")
    rollout_episodes = load_episode_scores(scores_dir)
    print(f"  Rollout episodes: {len(rollout_episodes)}")

    rollout_result = compute_rollout_weights(
        rollout_episodes,
        tau=args.tau,
        delta_max=args.delta_max,
        weight_clip_min=WEIGHT_CLIP_MIN,
        subtract_mean=SUBTRACT_MEAN,
    )
    rollout_weights = rollout_result["flat_weights"]
    rollout_stats = rollout_result["stats"]
    rollout_size = len(rollout_weights)

    print(f"  Rollout frames: {rollout_size}")
    print(f"  Rollout weight stats: mean={rollout_stats['mean_weight']:.3f}, "
          f"std={rollout_stats['std_weight']:.3f}, "
          f"range=[{rollout_stats['min_weight']:.3f}, {rollout_stats['max_weight']:.3f}]")

    if rollout_stats["std_weight"] < 0.1:
        print(f"\n  *** Warning: rollout weight std is very low ({rollout_stats['std_weight']:.3f}). ***")
        print(f"  *** AWR signal may be weak. Consider collecting more diverse rollouts. ***\n")

    # 3. Build combined weights array: [hub_fixed ... | rollout_awr ...]
    hub_weights = np.full(hub_size, args.hub_weight, dtype=np.float32)
    combined_weights = np.concatenate([hub_weights, rollout_weights.astype(np.float32)])

    print(f"\nCombined dataset: {len(combined_weights)} frames "
          f"({hub_size} hub + {rollout_size} rollout)")
    print(f"  Hub portion:    weight={args.hub_weight} (fixed)")
    print(f"  Rollout portion: mean={rollout_stats['mean_weight']:.3f}, "
          f"std={rollout_stats['std_weight']:.3f}")

    # 4. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "weights.npy", combined_weights)

    advantages_meta = {
        "round": args.round,
        "hub_dataset": HUB_DATASET,
        "hub_size": hub_size,
        "hub_fixed_weight": args.hub_weight,
        "rollout_size": rollout_size,
        "combined_size": len(combined_weights),
        "tau": args.tau,
        "delta_max": args.delta_max,
        "subtract_mean": SUBTRACT_MEAN,
        "rollout_stats": rollout_stats,
        "num_episodes": len(rollout_episodes),
        "episodes": [
            {
                "episode_id": ep["episode_id"],
                "num_frames": ep["num_frames"],
                "voc": ep.get("voc", 0.0),
                "weights": ep["weights"],
                "per_frame_progress": ep["per_frame_progress"],
                "deltas": ep["deltas"],
            }
            for ep in rollout_result["episodes"]
        ],
    }
    with open(output_dir / "advantages.json", "w") as f:
        json.dump(advantages_meta, f, indent=2)

    # 5. Generate plots
    plots_dir = round_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots...")
    plot_progress_curves(rollout_episodes, advantages_meta, plots_dir)
    plot_weight_distribution(advantages_meta, plots_dir)
    plot_weight_heatmap(advantages_meta, plots_dir)
    plot_dataset_summary(rollout_episodes, plots_dir)

    print(f"\nSaved weights.npy ({len(combined_weights)} entries) and advantages.json")
    print(f"Output: {output_dir}")
    print(f"Plots:  {plots_dir}")


if __name__ == "__main__":
    main()
