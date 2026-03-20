"""Run the full AWR pipeline: score → advantages → inspect → train.

Usage:
    python -m awr.pipeline
    python -m awr.pipeline --skip-training
    python -m awr.pipeline --instruction "Pick up the cube" --steps 10000
"""

from __future__ import annotations

import argparse

from awr.config import (
    DATASET_REPO_ID,
    INSTRUCTION,
    LEARNING_RATE,
    NUM_TRAIN_STEPS,
    OUTPUT_DIR,
    TAU,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AWR Diffusion Policy with TOPReward")
    p.add_argument("--dataset", default=DATASET_REPO_ID)
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--skip-scoring", action="store_true", help="Skip step 1 if scores exist")
    p.add_argument("--skip-training", action="store_true", help="Only score and compute weights")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Stage 1: AWR Diffusion Policy with TOPReward")
    print("=" * 60)
    print(f"Dataset:     {args.dataset}")
    print(f"Instruction: {args.instruction}")
    print(f"TAU:         {args.tau}")
    print(f"Steps:       {args.steps}")
    print(f"LR:          {args.lr}")
    print()

    # Step 1: Score dataset
    if not args.skip_scoring:
        print("[1/4] Scoring dataset with TOPReward...")
        from awr.score_dataset import parse_args as score_parse, run_scoring

        score_args = score_parse.__wrapped__() if hasattr(score_parse, "__wrapped__") else argparse.Namespace(
            dataset=args.dataset,
            instruction=args.instruction,
            camera_key=None,
            eval_frames=None,
            episodes=None,
            list_keys=False,
            output=OUTPUT_DIR / "scores",
        )
        # Override with pipeline args
        score_args.dataset = args.dataset
        score_args.instruction = args.instruction
        from awr.config import CAMERA_KEY, NUM_EVAL_FRAMES
        score_args.camera_key = CAMERA_KEY
        score_args.eval_frames = NUM_EVAL_FRAMES
        score_args.episodes = None
        score_args.list_keys = False
        score_args.output = OUTPUT_DIR / "scores"

        run_scoring(score_args)
        print()
    else:
        print("[1/4] Skipping scoring (--skip-scoring)")

    # Step 2: Compute advantages
    print("[2/4] Computing advantage weights...")
    from awr.compute_advantages import run_compute_advantages

    adv_args = argparse.Namespace(
        scores_dir=OUTPUT_DIR / "scores",
        tau=args.tau,
        delta_max=2.0,
        output=OUTPUT_DIR / "advantages",
    )
    run_compute_advantages(adv_args)
    print()

    # Step 3: Inspect
    print("[3/4] Generating inspection plots...")
    from awr.inspect_scores import run_inspect
    run_inspect()
    print(f"Check {OUTPUT_DIR / 'plots'} before proceeding")
    print()

    if args.skip_training:
        print("Skipping training (--skip-training). Review plots, then run:")
        print("  python -m awr.weighted_trainer")
        return

    # Step 4: Train
    print("[4/4] Training diffusion policy with AWR...")
    from awr.weighted_trainer import run_training

    train_args = argparse.Namespace(
        dataset=args.dataset,
        weights=OUTPUT_DIR / "advantages" / "weights.npy",
        steps=args.steps,
        lr=args.lr,
        batch_size=8,
        output=OUTPUT_DIR / "checkpoints" / "diffusion_awr",
    )
    run_training(train_args)

    print()
    print("=" * 60)
    print("Done!")
    print(f"Checkpoint: {OUTPUT_DIR / 'checkpoints' / 'diffusion_awr' / 'final'}")
    print(f"Plots:      {OUTPUT_DIR / 'plots'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
