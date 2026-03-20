"""One command to run a full AWR improvement round.

collect rollouts → score with TOPReward → build weights → inspect plots → fine-tune

Usage:
    python -m awr.run_loop --round 1
    python -m awr.run_loop --round 2 --num-episodes 20 --steps 5000
    python -m awr.run_loop --round 1 --skip-collect   # reuse existing rollouts
    python -m awr.run_loop --round 1 --skip-training   # collect+score only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from awr.config_hw import (
    BC_CHECKPOINT,
    LEARNING_RATE,
    NUM_ROLLOUTS,
    NUM_TRAIN_STEPS,
    OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a full AWR improvement round")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--num-episodes", type=int, default=NUM_ROLLOUTS)
    p.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--skip-collect", action="store_true", help="Skip rollout collection")
    p.add_argument("--skip-training", action="store_true", help="Only collect and score")
    # Pass-through args for collect_rollouts
    p.add_argument("--robot-port", default=None)
    p.add_argument("--robot-type", default=None)
    p.add_argument("--cameras", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_step(description: str, cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {description}")
    print(f"{'─' * 60}")
    print(f"  $ {' '.join(cmd)}\n")

    if dry_run:
        print("  (dry run — skipped)")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFailed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    args = parse_args()

    round_dir = OUTPUT_DIR / f"round_{args.round:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint
    if args.round == 1:
        checkpoint = BC_CHECKPOINT
    else:
        prev = OUTPUT_DIR / f"checkpoints/round_{args.round - 1:03d}/final"
        if prev.exists():
            checkpoint = prev
        else:
            print(f"Warning: {prev} not found, using BC checkpoint")
            checkpoint = BC_CHECKPOINT

    print(f"{'=' * 60}")
    print(f"  AWR Round {args.round}")
    print(f"  Policy: {checkpoint}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Training steps: {args.steps}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Step 1: Collect rollouts
    if not args.skip_collect:
        collect_cmd = [
            sys.executable, "-m", "awr.collect_rollouts",
            "--round", str(args.round),
            "--num-episodes", str(args.num_episodes),
            "--checkpoint", str(checkpoint),
        ]
        if args.robot_port:
            collect_cmd += ["--robot-port", args.robot_port]
        if args.robot_type:
            collect_cmd += ["--robot-type", args.robot_type]
        if args.cameras:
            collect_cmd += ["--cameras", args.cameras]
        if args.dry_run:
            collect_cmd.append("--dry-run")

        run_step(
            f"[1/4] Collecting {args.num_episodes} rollouts on SO-100",
            collect_cmd, dry_run=False,
        )
    else:
        print(f"\n[1/4] Skipping collection (--skip-collect)")

    # Step 2: Score rollouts
    score_cmd = [
        sys.executable, "-m", "awr.score_rollouts",
        "--round", str(args.round),
    ]
    run_step("[2/4] Scoring rollouts with TOPReward", score_cmd, dry_run=args.dry_run)

    # Step 3: Build combined weights
    build_cmd = [
        sys.executable, "-m", "awr.build_weighted_dataset",
        "--round", str(args.round),
    ]
    run_step("[3/4] Building advantage weights (hub fixed + rollout AWR)", build_cmd, dry_run=args.dry_run)

    print(f"\n  Check plots at: {round_dir / 'plots'}")
    print(f"  - weight_distribution.png should show REAL spread (not a spike at 2.0)")
    print(f"  - weight_heatmap.png should show red/yellow/green mix")

    if args.skip_training:
        print(f"\n[4/4] Skipping training (--skip-training)")
        print(f"  Review plots, then run:")
        print(f"  python -m awr.awr_finetune --round {args.round}")
        return

    # Step 4: AWR fine-tune
    train_cmd = [
        sys.executable, "-m", "awr.awr_finetune",
        "--round", str(args.round),
        "--steps", str(args.steps),
        "--lr", str(args.lr),
        "--checkpoint", str(checkpoint),
    ]
    run_step("[4/4] AWR fine-tuning", train_cmd, dry_run=args.dry_run)

    total_time = time.time() - t0

    # Log
    ckpt_dir = OUTPUT_DIR / f"checkpoints/round_{args.round:03d}"
    log_path = OUTPUT_DIR / "log.json"
    log = []
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)

    log.append({
        "round": args.round,
        "checkpoint": str(ckpt_dir / "final"),
        "num_episodes": args.num_episodes,
        "train_steps": args.steps,
        "lr": args.lr,
        "total_time_s": round(total_time, 1),
    })
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Round {args.round} complete! ({total_time:.0f}s)")
    print(f"  Checkpoint: {ckpt_dir / 'final'}")
    print(f"  Plots:      {round_dir / 'plots'}")
    print(f"  Log:        {log_path}")
    print(f"")
    print(f"  Next round:")
    print(f"    python -m awr.run_loop --round {args.round + 1}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
