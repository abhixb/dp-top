"""Fine-tune the BC diffusion policy on hub demos + scored rollouts with AWR.

Combines hub dataset (fixed weight) and rollout dataset (AWR-scored weights)
using an indexed wrapper that tracks global position for weight lookup.

Usage:
    python -m awr.awr_finetune --round 1
    python -m awr.awr_finetune --round 1 --steps 5000 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from awr.config_hw import (
    BATCH_SIZE,
    BC_CHECKPOINT,
    DEVICE,
    HUB_DATASET,
    LEARNING_RATE,
    NUM_TRAIN_STEPS,
    OUTPUT_DIR,
)


class IndexedConcatDataset(Dataset):
    """ConcatDataset that injects a 'global_weight_idx' field into each sample.

    Hub samples get indices [0, hub_size), rollout samples get [hub_size, hub_size+rollout_size).
    This lets the training loop look up the correct weight for each sample.
    """

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        ds_idx = 0
        offset = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                ds_idx = i
                break
            offset = cum_size

        local_idx = idx - offset
        item = self.datasets[ds_idx][local_idx]
        item["global_weight_idx"] = torch.tensor(idx, dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AWR fine-tune on hub + rollout data")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Policy to fine-tune (default: BC for round 1, prev round otherwise)")
    p.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def resolve_checkpoint(args) -> Path:
    if args.checkpoint:
        return args.checkpoint
    if args.round == 1:
        return BC_CHECKPOINT
    prev = OUTPUT_DIR / f"checkpoints/round_{args.round - 1:03d}/final"
    if prev.exists():
        return prev
    print(f"Previous round checkpoint not found at {prev}, falling back to BC")
    return BC_CHECKPOINT


def resolve_rollout_dataset(round_num: int) -> tuple[str, Path | None]:
    round_dir = OUTPUT_DIR / f"round_{round_num:03d}"
    meta_path = round_dir / "collection_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        dataset_path = Path(meta["dataset_path"])
        return dataset_path.name, dataset_path.parent

    rollouts_dir = round_dir / "rollouts"
    if rollouts_dir.exists():
        subdirs = [d for d in rollouts_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0].name, rollouts_dir

    raise FileNotFoundError(
        f"No rollout dataset found for round {round_num}. Run collect_rollouts first."
    )


def main():
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    round_dir = OUTPUT_DIR / f"round_{args.round:03d}"
    ckpt_dir = args.output or OUTPUT_DIR / f"checkpoints/round_{args.round:03d}"
    weights_path = round_dir / "weights.npy"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Run build_weighted_dataset first."
        )

    # 1. Load policy
    checkpoint = resolve_checkpoint(args)
    print(f"Loading policy from: {checkpoint}")
    policy = DiffusionPolicy.from_pretrained(str(checkpoint))
    policy.to(DEVICE)
    cfg = policy.config
    print(f"  n_obs_steps={cfg.n_obs_steps}, horizon={cfg.horizon}, "
          f"params={sum(p.numel() for p in policy.parameters()):,}")

    # 2. Build delta_timestamps
    hub_ds_temp = LeRobotDataset(repo_id=HUB_DATASET)
    fps = hub_ds_temp.fps
    features = dataset_to_policy_features(hub_ds_temp.meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    del hub_ds_temp

    obs_delta_ts = [i / fps for i in cfg.observation_delta_indices]
    act_delta_ts = [i / fps for i in cfg.action_delta_indices]
    delta_timestamps = {key: obs_delta_ts for key in input_features}
    delta_timestamps["action"] = act_delta_ts

    # 3. Load datasets with delta_timestamps
    print(f"\nLoading hub dataset: {HUB_DATASET}")
    hub_ds = LeRobotDataset(repo_id=HUB_DATASET, delta_timestamps=delta_timestamps)
    hub_size = len(hub_ds)
    print(f"  Hub: {hub_size} samples")

    rollout_repo_id, rollout_root = resolve_rollout_dataset(args.round)
    print(f"Loading rollout dataset: {rollout_repo_id}")
    rollout_kwargs = {"repo_id": rollout_repo_id, "delta_timestamps": delta_timestamps}
    if rollout_root is not None:
        rollout_kwargs["root"] = str(rollout_root)
    rollout_ds = LeRobotDataset(**rollout_kwargs)
    rollout_size = len(rollout_ds)
    print(f"  Rollouts: {rollout_size} samples")

    # 4. Combine with indexed wrapper
    combined_ds = IndexedConcatDataset([hub_ds, rollout_ds])
    print(f"  Combined: {len(combined_ds)} samples")

    # 5. Load weights
    weights = np.load(weights_path)
    weights = weights[:len(combined_ds)]
    print(f"\nWeights: mean={weights.mean():.3f}, std={weights.std():.3f}")
    print(f"  Hub [{0}:{hub_size}]: mean={weights[:hub_size].mean():.3f}")
    print(f"  Rollout [{hub_size}:{hub_size + rollout_size}]: "
          f"mean={weights[hub_size:].mean():.3f}, std={weights[hub_size:].std():.3f}")

    # 6. Dataloader
    dataloader = DataLoader(
        combined_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 7. Train
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    policy.train()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_entries = []
    step = 0
    t0 = time.time()
    log_every = 100
    save_every = 1000

    print(f"\nTraining: {args.steps} steps, lr={args.lr}, bs={args.batch_size}")

    for epoch in range(9999):
        for batch in dataloader:
            if step >= args.steps:
                break

            # Look up per-sample weights using the global index
            global_indices = batch.pop("global_weight_idx").numpy()
            batch_weights = torch.tensor(
                weights[global_indices], device=DEVICE, dtype=torch.float32,
            )

            batch = {
                k: v.to(DEVICE) if hasattr(v, "to") else v
                for k, v in batch.items()
            }

            loss, _ = policy.forward(batch)
            weighted_loss = loss * batch_weights.mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            if step % log_every == 0:
                entry = {
                    "step": step,
                    "loss": round(weighted_loss.item(), 4),
                    "raw_loss": round(loss.item(), 4),
                    "mean_weight": round(batch_weights.mean().item(), 3),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                log_entries.append(entry)
                print(
                    f"[{step}/{args.steps}] "
                    f"loss={entry['loss']:.4f} "
                    f"raw={entry['raw_loss']:.4f} "
                    f"w={entry['mean_weight']:.2f}"
                )

            if step > 0 and step % save_every == 0:
                ckpt_path = ckpt_dir / f"checkpoint_{step}"
                policy.save_pretrained(str(ckpt_path))
                print(f"  Saved: {ckpt_path}")

            step += 1

        if step >= args.steps:
            break

    final_path = ckpt_dir / "final"
    policy.save_pretrained(str(final_path))
    total_time = time.time() - t0

    run_log = {
        "round": args.round,
        "checkpoint_source": str(checkpoint),
        "hub_size": hub_size,
        "rollout_size": rollout_size,
        "steps": step,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "total_time_s": round(total_time, 1),
        "entries": log_entries,
    }
    with open(ckpt_dir / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\nDone in {total_time:.0f}s ({step} steps)")
    print(f"Checkpoint: {final_path}")


if __name__ == "__main__":
    main()
