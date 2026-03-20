"""Train a diffusion policy with advantage-weighted loss.

Includes LR warmup + cosine decay, EMA weights, and gradient clipping.

Usage:
    python -m awr.weighted_trainer
    python -m awr.weighted_trainer --steps 100000 --lr 1e-4 --batch-size 16
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from awr.config import (
    BATCH_SIZE,
    DATASET_REPO_ID,
    DEVICE,
    LEARNING_RATE,
    LOG_EVERY,
    NUM_TRAIN_STEPS,
    OUTPUT_DIR,
    SAVE_EVERY,
)


# ─── EMA ───

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.data, 1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module):
        """Copy EMA weights into model (for eval/saving)."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ─── LR Schedule ───

def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-7) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ─── Args ───

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train diffusion policy with AWR")
    p.add_argument("--dataset", default=DATASET_REPO_ID)
    p.add_argument("--weights", type=Path, default=OUTPUT_DIR / "advantages" / "weights.npy")
    p.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "checkpoints" / "diffusion_awr")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from checkpoint dir (e.g. checkpoint_43000). Auto-detects start step.")
    p.add_argument("--warmup", type=int, default=500, help="LR warmup steps")
    p.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")
    return p.parse_args()


def run_training(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    # 1. Build policy config first (need delta_timestamps for dataset)
    print(f"Loading dataset metadata: {args.dataset}")
    ds_meta = LeRobotDataset(repo_id=args.dataset)
    features = dataset_to_policy_features(ds_meta.meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    policy_cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
    )

    # 2. Compute delta_timestamps from policy config
    fps = ds_meta.fps
    obs_delta_ts = [i / fps for i in policy_cfg.observation_delta_indices]
    act_delta_ts = [i / fps for i in policy_cfg.action_delta_indices]

    delta_timestamps = {}
    for key in input_features:
        delta_timestamps[key] = obs_delta_ts
    delta_timestamps["action"] = act_delta_ts
    del ds_meta

    # 3. Reload dataset with delta_timestamps (produces temporal windows + action_is_pad)
    print(f"Loading dataset with delta_timestamps (horizon={policy_cfg.horizon}, n_obs={policy_cfg.n_obs_steps})")
    dataset = LeRobotDataset(repo_id=args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(dataset)} samples")

    # 4. Load advantage weights
    weights = np.load(args.weights)
    weights = weights[: len(dataset)]
    print(f"Loaded weights: mean={weights.mean():.3f}, std={weights.std():.3f}")

    # 5. Create diffusion policy (or resume from checkpoint)
    start_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        policy = DiffusionPolicy.from_pretrained(str(args.resume))
        dirname = args.resume.name
        if dirname.startswith("checkpoint_"):
            start_step = int(dirname.split("_")[1])
        print(f"  Resuming from step {start_step}")
    else:
        policy = DiffusionPolicy(config=policy_cfg, dataset_stats=dataset.meta.stats)
    policy.to(DEVICE)
    print(f"Policy on {DEVICE}, params: {sum(p.numel() for p in policy.parameters()):,}")

    # 6. EMA
    ema = EMAModel(policy, decay=args.ema_decay)
    print(f"EMA decay: {args.ema_decay}")

    # 7. Dataloader — use WeightedRandomSampler so high-advantage samples are
    # drawn more often; this is proper per-sample AWR weighting without needing
    # per-sample losses from the policy.
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # 8. Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    policy.train()

    args.output.mkdir(parents=True, exist_ok=True)
    log_entries = []
    step = start_step
    t0 = time.time()

    print(f"Training: {args.steps} steps, lr={args.lr}, warmup={args.warmup}, "
          f"grad_clip={args.grad_clip}, bs={args.batch_size}")

    for epoch in range(9999):
        for batch in dataloader:
            if step >= args.steps:
                break

            # LR schedule
            lr = get_cosine_lr(step, args.steps, args.warmup, args.lr)
            set_lr(optimizer, lr)

            # Move batch to GPU
            batch = {
                k: v.to(DEVICE) if hasattr(v, "to") else v
                for k, v in batch.items()
            }

            # Forward pass
            loss, _ = policy.forward(batch)

            # Backward + clip + update
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            # EMA update
            ema.update(policy)

            # Logging
            if step % LOG_EVERY == 0:
                entry = {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "lr": round(lr, 7),
                    "grad_norm": round(grad_norm.item(), 3),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                log_entries.append(entry)
                print(
                    f"[{step}/{args.steps}] "
                    f"loss={entry['loss']:.4f} "
                    f"lr={lr:.2e} "
                    f"gnorm={entry['grad_norm']:.2f}"
                )

            # Save checkpoint
            if step > 0 and step % SAVE_EVERY == 0:
                ckpt_path = args.output / f"checkpoint_{step}"
                policy.save_pretrained(ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

            step += 1

        if step >= args.steps:
            break

    # Save final checkpoint with EMA weights
    final_path = args.output / "final"
    # Backup live weights, apply EMA, save, restore
    live_state = {n: p.clone() for n, p in policy.named_parameters()}
    ema.apply_to(policy)
    policy.save_pretrained(final_path)
    # Also save non-EMA version
    for n, p in policy.named_parameters():
        p.data.copy_(live_state[n])
    policy.save_pretrained(args.output / "final_no_ema")

    total_time = time.time() - t0

    # Save training log
    run_log = {
        "dataset": args.dataset,
        "steps": step,
        "lr": args.lr,
        "warmup": args.warmup,
        "ema_decay": args.ema_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "weighting": "WeightedRandomSampler",
        "total_time_s": round(total_time, 1),
        "entries": log_entries,
    }
    with open(OUTPUT_DIR / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\nTraining complete in {total_time:.0f}s ({step} steps)")
    print(f"Final checkpoint (EMA): {final_path}")
    print(f"Final checkpoint (no EMA): {args.output / 'final_no_ema'}")
    print(f"Run log: {OUTPUT_DIR / 'run_log.json'}")


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
