"""Deploy the trained policy on SO-100, record episodes as a LeRobot dataset.

Wraps LeRobot's `lerobot-record` command with the right policy and config.

Usage:
    python -m awr.collect_rollouts
    python -m awr.collect_rollouts --num-episodes 20 --round 1
    python -m awr.collect_rollouts --checkpoint path/to/other/checkpoint
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from awr.config_hw import (
    BC_CHECKPOINT,
    EPISODE_MAX_STEPS,
    FOLLOWER_PORT,
    FPS,
    INSTRUCTION,
    NUM_ROLLOUTS,
    OUTPUT_DIR,
    RESET_TIME_S,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect rollouts on SO-100 with trained policy")
    p.add_argument("--checkpoint", type=Path, default=BC_CHECKPOINT)
    p.add_argument("--num-episodes", type=int, default=NUM_ROLLOUTS)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--repo-id", default=None, help="HF repo to push dataset (default: local only)")
    p.add_argument("--reset-time", type=int, default=RESET_TIME_S)
    p.add_argument("--robot-port", default=FOLLOWER_PORT)
    p.add_argument("--robot-type", default="so100_follower")
    p.add_argument(
        "--cameras",
        default='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}',
        help="Camera config in YAML dict format",
    )
    p.add_argument("--task", default=INSTRUCTION)
    p.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return p.parse_args()


def main():
    args = parse_args()

    round_dir = OUTPUT_DIR / f"round_{args.round:03d}"
    rollouts_dir = round_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint
    checkpoint = args.checkpoint
    if args.round > 1 and checkpoint == BC_CHECKPOINT:
        prev_ckpt = OUTPUT_DIR / f"checkpoints/round_{args.round - 1:03d}/final"
        if prev_ckpt.exists():
            checkpoint = prev_ckpt
            print(f"Round {args.round}: using previous round's checkpoint: {checkpoint}")
        else:
            print(f"Warning: previous round checkpoint not found at {prev_ckpt}")
            print(f"Falling back to BC checkpoint: {checkpoint}")

    # Default repo_id for local storage
    repo_id = args.repo_id or f"awr_rollouts_round_{args.round:03d}"

    episode_time_s = EPISODE_MAX_STEPS / FPS

    cmd = [
        "lerobot-record",
        f"--robot.type={args.robot_type}",
        f"--robot.port={args.robot_port}",
        f"--robot.cameras={args.cameras}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={rollouts_dir}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.single_task={args.task}",
        f"--dataset.fps={FPS}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time}",
        "--dataset.push_to_hub=false",
        f"--policy.path={checkpoint}",
    ]

    print(f"Round {args.round} — Collecting {args.num_episodes} rollouts")
    print(f"  Policy:   {checkpoint}")
    print(f"  Robot:    {args.robot_type} @ {args.robot_port}")
    print(f"  Output:   {rollouts_dir / repo_id}")
    print(f"  Task:     {args.task}")
    print()
    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("(dry run — not executing)")
        return

    # Save metadata for downstream scripts
    import json
    meta = {
        "round": args.round,
        "checkpoint": str(checkpoint),
        "num_episodes": args.num_episodes,
        "repo_id": repo_id,
        "root": str(rollouts_dir),
        "dataset_path": str(rollouts_dir / repo_id),
    }
    with open(round_dir / "collection_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nlerobot-record exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nRollouts saved to: {rollouts_dir / repo_id}")
    print(f"Metadata saved to: {round_dir / 'collection_meta.json'}")


if __name__ == "__main__":
    main()
