"""Compute per-step advantage weights from TOPReward scores.

Reads scored episode JSONs, interpolates to per-frame progress,
computes deltas, and produces a flat weights array for training.

Usage:
    python -m awr.compute_advantages
    python -m awr.compute_advantages --tau 3.0 --delta-max 2.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from awr.config import (
    DATASET_REPO_ID,
    DELTA_MAX,
    OUTPUT_DIR,
    SUBTRACT_MEAN,
    TAU,
    WEIGHT_CLIP_MIN,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute AWR advantage weights")
    p.add_argument("--scores-dir", type=Path, default=OUTPUT_DIR / "scores")
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--delta-max", type=float, default=DELTA_MAX)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "advantages")
    return p.parse_args()


def load_episode_scores(scores_dir: Path) -> list[dict]:
    """Load all episode_NNN.json files sorted by episode_id."""
    files = sorted(scores_dir.glob("episode_*.json"))
    if not files:
        raise FileNotFoundError(f"No episode JSON files found in {scores_dir}")
    episodes = []
    for f in files:
        with open(f) as fh:
            episodes.append(json.load(fh))
    episodes.sort(key=lambda e: e["episode_id"])
    return episodes


def interpolate_progress(ep: dict) -> np.ndarray:
    """Interpolate normalized prefix scores to per-frame progress."""
    num_frames = ep["num_frames"]
    prefix_lengths = np.array(ep["prefix_lengths"])
    normalized = np.array(ep["normalized"])

    # prefix_lengths are 1-indexed frame counts; convert to 0-indexed
    xp = prefix_lengths - 1
    return np.interp(np.arange(num_frames), xp, normalized)


def run_compute_advantages(args: argparse.Namespace) -> dict:
    episodes = load_episode_scores(args.scores_dir)
    print(f"Loaded scores for {len(episodes)} episodes")

    # Step 1: Interpolate to per-frame progress and compute deltas
    all_deltas = []
    for ep in episodes:
        progress = interpolate_progress(ep)
        deltas = np.diff(progress, prepend=0.0)
        ep["per_frame_progress"] = progress.tolist()
        ep["deltas"] = deltas.tolist()
        all_deltas.append(deltas)

    # Step 2: Dataset-wide mean delta
    all_deltas_flat = np.concatenate(all_deltas)
    mean_delta = float(all_deltas_flat.mean())
    print(f"Mean delta: {mean_delta:.6f}")

    # Step 3: Compute advantage weights
    for ep in episodes:
        deltas = np.array(ep["deltas"])
        if SUBTRACT_MEAN:
            advantages = deltas - mean_delta
        else:
            advantages = deltas
        weights = np.clip(args.tau * np.exp(advantages), WEIGHT_CLIP_MIN, args.delta_max)
        ep["weights"] = weights.tolist()

    # Step 4: Build flat weights array using episode boundaries from score JSONs
    # Each episode JSON has start_idx and end_idx
    max_end = max(ep["end_idx"] for ep in episodes)
    flat_weights = np.ones(max_end, dtype=np.float64)
    for ep in episodes:
        start = ep["start_idx"]
        end = ep["end_idx"]
        n = end - start
        flat_weights[start:end] = ep["weights"][:n]

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "weights.npy", flat_weights.astype(np.float32))

    # Compute stats
    all_weights = np.concatenate([np.array(ep["weights"]) for ep in episodes])
    stats = {
        "mean_weight": round(float(all_weights.mean()), 4),
        "std_weight": round(float(all_weights.std()), 4),
        "min_weight": round(float(all_weights.min()), 4),
        "max_weight": round(float(all_weights.max()), 4),
        "pct_above_1": round(float((all_weights > 1.0).mean()), 4),
    }

    # Build output JSON (strip large arrays from episodes for the summary)
    total_frames = sum(ep["num_frames"] for ep in episodes)
    output = {
        "dataset": DATASET_REPO_ID,
        "tau": args.tau,
        "delta_max": args.delta_max,
        "mean_delta": round(mean_delta, 6),
        "total_frames": total_frames,
        "num_episodes": len(episodes),
        "episodes": [
            {
                "episode_id": ep["episode_id"],
                "num_frames": ep["num_frames"],
                "weights": ep["weights"],
                "per_frame_progress": ep["per_frame_progress"],
                "deltas": ep["deltas"],
            }
            for ep in episodes
        ],
        "stats": stats,
    }

    with open(args.output / "advantages.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Total frames: {total_frames}")
    print(f"Weight stats: mean={stats['mean_weight']:.3f}, std={stats['std_weight']:.3f}")
    print(f"  min={stats['min_weight']:.3f}, max={stats['max_weight']:.3f}")
    print(f"  {stats['pct_above_1'] * 100:.1f}% above 1.0")
    print(f"Saved weights.npy ({flat_weights.shape[0]} entries) and advantages.json")
    print(f"Output: {args.output}")

    return output


def main():
    args = parse_args()
    run_compute_advantages(args)


if __name__ == "__main__":
    main()
