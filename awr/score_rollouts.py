"""Score collected rollouts with TOPReward.

Reuses the scoring logic from score_dataset.py, pointed at local rollout data.

Usage:
    python -m awr.score_rollouts --round 1
    python -m awr.score_rollouts --dataset path/to/local/rollouts/repo_id
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from awr.config_hw import (
    CAMERA_KEY,
    INSTRUCTION,
    MAX_EPISODE_FRAMES,
    NUM_EVAL_FRAMES,
    OUTPUT_DIR,
)
from awr.score_dataset import (
    get_episode_boundaries,
    list_dataset_keys,
    score_episode,
    tensor_to_numpy_hwc,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score rollout episodes with TOPReward")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--dataset", default=None, help="Explicit dataset path or repo_id (overrides --round)")
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--camera-key", default=CAMERA_KEY)
    p.add_argument("--eval-frames", type=int, default=NUM_EVAL_FRAMES)
    p.add_argument("--episodes", default=None, help="Comma-separated episode IDs")
    p.add_argument("--list-keys", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def resolve_dataset_path(args) -> tuple[str, Path]:
    """Return (repo_id_or_path, root) for the rollout dataset."""
    if args.dataset:
        p = Path(args.dataset)
        if p.exists():
            # Local path: repo_id is the directory name, root is parent
            return p.name, p.parent
        # Treat as repo_id
        return args.dataset, None

    # Infer from round
    round_dir = OUTPUT_DIR / f"round_{args.round:03d}"
    meta_path = round_dir / "collection_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        dataset_path = Path(meta["dataset_path"])
        return dataset_path.name, dataset_path.parent

    # Fallback: scan for directories in rollouts/
    rollouts_dir = round_dir / "rollouts"
    if rollouts_dir.exists():
        subdirs = [d for d in rollouts_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0].name, rollouts_dir
        elif len(subdirs) > 1:
            print(f"Multiple datasets found in {rollouts_dir}:")
            for d in subdirs:
                print(f"  {d}")
            print("Use --dataset to specify which one.")
            raise SystemExit(1)

    raise FileNotFoundError(
        f"No rollout dataset found for round {args.round}. "
        f"Run collect_rollouts first, or use --dataset to specify path."
    )


def main():
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id, root = resolve_dataset_path(args)
    print(f"Loading rollout dataset: repo_id={repo_id}, root={root}")

    kwargs = {"repo_id": repo_id}
    if root is not None:
        kwargs["root"] = str(root)
    dataset = LeRobotDataset(**kwargs)

    if args.list_keys:
        list_dataset_keys(dataset)
        return

    boundaries = get_episode_boundaries(dataset)
    num_episodes = len(boundaries)
    print(f"Dataset: {num_episodes} episodes, {len(dataset)} total frames")

    if args.episodes:
        ep_ids = [int(x) for x in args.episodes.split(",")]
    else:
        ep_ids = [b["episode_id"] for b in boundaries]

    ep_lookup = {b["episode_id"]: b for b in boundaries}

    # Output directory
    if args.output:
        scores_dir = args.output
    else:
        scores_dir = OUTPUT_DIR / f"round_{args.round:03d}" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load TOPReward
    print("Loading TOPReward model (Qwen 8B)...")
    from topreward.clients.qwen import QwenClient
    client = QwenClient(load_in_4bit=True)
    print("Model loaded.")

    results = []
    total_start = time.time()

    for i, ep_id in enumerate(ep_ids):
        b = ep_lookup[ep_id]
        ep_result = score_episode(
            client, dataset, ep_id, b["start_idx"], b["end_idx"],
            args.camera_key, args.instruction, args.eval_frames,
        )
        results.append(ep_result)

        out_path = scores_dir / f"episode_{ep_id:03d}.json"
        with open(out_path, "w") as f:
            json.dump(ep_result, f, indent=2)

        print(
            f"[{i + 1}/{len(ep_ids)}] Episode {ep_id}: "
            f"{ep_result['num_frames']} frames, "
            f"VOC={ep_result['voc']:.3f}, "
            f"{ep_result['scoring_time_s']:.1f}s"
        )

    total_time = time.time() - total_start
    vocs = [r["voc"] for r in results]

    print(f"\n{'=' * 50}")
    print(f"Scored {len(results)} rollout episodes in {total_time:.1f}s")
    print(f"VOC: mean={np.mean(vocs):.3f}, min={np.min(vocs):.3f}, max={np.max(vocs):.3f}")
    print(f"Scores saved to: {scores_dir}")

    # Check for variance
    if np.std(vocs) < 0.05:
        print(f"\nWarning: VOC std is very low ({np.std(vocs):.3f}). "
              f"Rollouts may be too uniform for AWR to help.")


if __name__ == "__main__":
    main()
