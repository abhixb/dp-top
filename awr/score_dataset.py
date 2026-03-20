"""Score every episode in a LeRobot dataset with TOPReward.

Usage:
    python -m awr.score_dataset
    python -m awr.score_dataset --list-keys
    python -m awr.score_dataset --episodes 0,1,2
    python -m awr.score_dataset --instruction "Pick up the red cube"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from awr.config import (
    CAMERA_KEY,
    DATASET_REPO_ID,
    INSTRUCTION,
    MAX_EPISODE_FRAMES,
    NUM_EVAL_FRAMES,
    OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score LeRobot episodes with TOPReward")
    p.add_argument("--dataset", default=DATASET_REPO_ID)
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--camera-key", default=CAMERA_KEY)
    p.add_argument("--eval-frames", type=int, default=NUM_EVAL_FRAMES)
    p.add_argument("--episodes", default=None, help="Comma-separated episode IDs")
    p.add_argument("--list-keys", action="store_true", help="Print dataset keys and exit")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "scores")
    return p.parse_args()


def list_dataset_keys(dataset) -> None:
    """Print every key in the dataset with shape/dtype info."""
    sample = dataset[0]
    print("\nDataset keys:")
    for key in sorted(sample.keys()):
        val = sample[key]
        if hasattr(val, "shape"):
            print(f"  {key}: shape={tuple(val.shape)} dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val).__name__} = {val}")
    print()


def tensor_to_numpy_hwc(tensor) -> np.ndarray:
    """Convert a CxHxW float torch tensor to HxWxC uint8 numpy array."""
    arr = tensor.cpu().numpy()
    # CxHxW -> HxWxC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    # float [0,1] -> uint8 [0,255]
    if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    return arr


def get_episode_boundaries(dataset) -> list[dict]:
    """Get episode start/end indices from dataset metadata."""
    episodes = dataset.meta.episodes
    boundaries = []
    for i in range(len(episodes)):
        ep = episodes[i]
        boundaries.append({
            "episode_id": ep["episode_index"],
            "start_idx": ep["dataset_from_index"],
            "end_idx": ep["dataset_to_index"],
            "length": ep["length"],
        })
    return boundaries


def score_episode(
    client,
    dataset,
    ep_id: int,
    start_idx: int,
    end_idx: int,
    camera_key: str,
    instruction: str,
    num_eval_frames: int,
) -> dict:
    """Score a single episode and return results dict."""
    num_frames = end_idx - start_idx

    # Extract frames as numpy HxWxC uint8
    frames_np = []
    for idx in range(start_idx, end_idx):
        frame_tensor = dataset[idx][camera_key]
        frames_np.append(tensor_to_numpy_hwc(frame_tensor))

    # Subsample if episode exceeds MAX_EPISODE_FRAMES to avoid OOM
    subsampled = False
    if len(frames_np) > MAX_EPISODE_FRAMES:
        indices = np.linspace(0, len(frames_np) - 1, MAX_EPISODE_FRAMES, dtype=int)
        frames_np = [frames_np[i] for i in indices]
        subsampled = True

    t0 = time.time()
    result = client.compute_instruction_rewards_for_prefixes(
        frames=frames_np,
        instruction=instruction,
        num_samples=num_eval_frames,
        reduction="mean",
    )
    scoring_time = time.time() - t0

    # Compute VOC from normalized prefix rewards
    from scipy.stats import spearmanr

    normalized = result.normalized_prefix_rewards or []
    if len(normalized) >= 2:
        voc, _ = spearmanr(list(range(len(normalized))), normalized)
        voc = float(voc) if not np.isnan(voc) else 0.0
    else:
        voc = 0.0

    return {
        "episode_id": ep_id,
        "num_frames": num_frames,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "voc": round(voc, 4),
        "raw_scores": result.prefix_rewards or [],
        "normalized": normalized,
        "prefix_lengths": result.prefix_lengths or [],
        "scoring_time_s": round(scoring_time, 2),
    }


def run_scoring(args: argparse.Namespace) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(repo_id=args.dataset)

    if args.list_keys:
        list_dataset_keys(dataset)
        sys.exit(0)

    # Determine episode boundaries
    boundaries = get_episode_boundaries(dataset)
    num_episodes = len(boundaries)
    print(f"Dataset has {num_episodes} episodes, {len(dataset)} total frames")

    # Parse episode filter
    if args.episodes:
        ep_ids = [int(x) for x in args.episodes.split(",")]
    else:
        ep_ids = [b["episode_id"] for b in boundaries]

    # Build lookup
    ep_lookup = {b["episode_id"]: b for b in boundaries}

    # Initialize TOPReward scorer (QwenClient)
    print("Loading TOPReward model (Qwen 8B)...")
    from topreward.clients.qwen import QwenClient

    client = QwenClient(load_in_4bit=True)
    print("Model loaded.")

    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for i, ep_id in enumerate(ep_ids):
        b = ep_lookup[ep_id]
        start_idx = b["start_idx"]
        end_idx = b["end_idx"]

        ep_result = score_episode(
            client, dataset, ep_id, start_idx, end_idx,
            args.camera_key, args.instruction, args.eval_frames,
        )
        results.append(ep_result)

        # Save per-episode JSON
        out_path = args.output / f"episode_{ep_id:03d}.json"
        with open(out_path, "w") as f:
            json.dump(ep_result, f, indent=2)

        print(
            f"[{i + 1}/{len(ep_ids)}] Episode {ep_id}: "
            f"{ep_result['num_frames']} frames, "
            f"VOC={ep_result['voc']:.3f}, "
            f"{ep_result['scoring_time_s']:.1f}s"
        )

    total_time = time.time() - total_start

    # Summary
    vocs = [r["voc"] for r in results]
    print(f"\n{'=' * 50}")
    print(f"Scored {len(results)} episodes in {total_time:.1f}s")
    print(f"VOC: mean={np.mean(vocs):.3f}, min={np.min(vocs):.3f}, max={np.max(vocs):.3f}")
    print(f"Scores saved to: {args.output}")

    return results


def main():
    args = parse_args()
    run_scoring(args)


if __name__ == "__main__":
    main()
