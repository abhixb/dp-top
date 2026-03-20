"""Offline evaluation of the trained diffusion policy.

Loads the policy, runs it autoregressively on dataset episodes,
and compares predicted actions to ground truth.

Usage:
    python -m awr.evaluate
    python -m awr.evaluate --checkpoint awr/outputs/checkpoints/diffusion_awr/final --episodes 0,1,2
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from awr.config import DATASET_REPO_ID, DEVICE, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained diffusion policy offline")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "checkpoints" / "diffusion_awr" / "final",
    )
    p.add_argument("--dataset", default=DATASET_REPO_ID)
    p.add_argument("--episodes", type=str, default=None, help="Comma-separated episode indices (default: 5 evenly spaced)")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "eval")
    return p.parse_args()


@torch.inference_mode()
def evaluate_episode(
    policy,
    dataset,
    episode_idx: int,
    preprocessor,
    postprocessor,
    device: str,
) -> dict:
    """Run policy autoregressively on one episode, return metrics."""
    ep = dataset.meta.episodes[episode_idx]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    ep_len = ep_end - ep_start

    policy.reset()

    gt_actions = []
    pred_actions = []

    for t in range(ep_len):
        global_idx = ep_start + t
        sample = dataset[global_idx]

        # Ground truth action (single step, before temporal windowing)
        gt_action = sample["action"].numpy()  # (action_dim,) from windowed: (horizon, action_dim)
        # For windowed dataset, action is (horizon, action_dim) — take the current step
        if gt_action.ndim == 2:
            # The "current" action in the window depends on n_obs_steps
            # For delta_indices starting at -(n_obs_steps-1), current = index n_obs_steps-1
            current_idx = policy.config.n_obs_steps - 1
            gt_action = gt_action[current_idx]

        gt_actions.append(gt_action)

        # Build observation dict for policy (single timestep, no batch dim)
        obs = {}
        for key in sample:
            if key.startswith("observation."):
                val = sample[key]
                if val.ndim >= 2 and val.shape[0] == policy.config.n_obs_steps:
                    # Windowed obs: (n_obs_steps, ...) — take last (current) frame
                    val = val[-1]
                obs[key] = val

        # Preprocess (adds batch dim, normalizes, moves to device)
        obs = preprocessor(obs)

        # Policy predicts action
        action = policy.select_action(obs)  # (batch, action_dim)

        # Postprocess (unnormalize, move to cpu)
        action = postprocessor(action)
        action = action.squeeze(0).numpy()  # (action_dim,)

        pred_actions.append(action)

    gt_actions = np.array(gt_actions)    # (T, action_dim)
    pred_actions = np.array(pred_actions)  # (T, action_dim)

    # Metrics
    mse = np.mean((gt_actions - pred_actions) ** 2)
    mae = np.mean(np.abs(gt_actions - pred_actions))
    per_dim_mse = np.mean((gt_actions - pred_actions) ** 2, axis=0)

    return {
        "episode_idx": episode_idx,
        "length": ep_len,
        "mse": float(mse),
        "mae": float(mae),
        "per_dim_mse": per_dim_mse.tolist(),
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
    }


def plot_episode_actions(result: dict, output_dir: Path):
    """Plot predicted vs ground truth actions for one episode."""
    gt = result["gt_actions"]
    pred = result["pred_actions"]
    ep_idx = result["episode_idx"]
    n_dims = gt.shape[1]

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True)
    fig.patch.set_facecolor("#1e1e2e")

    if n_dims == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.set_facecolor("#1e1e2e")
        ax.plot(gt[:, dim], color="#89b4fa", linewidth=1.5, label="Ground Truth", alpha=0.9)
        ax.plot(pred[:, dim], color="#f38ba8", linewidth=1.5, label="Predicted", alpha=0.9, linestyle="--")
        ax.set_ylabel(f"Dim {dim}", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        if dim == 0:
            ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white", fontsize=9)

    axes[-1].set_xlabel("Timestep", color="white", fontsize=11)
    fig.suptitle(
        f"Episode {ep_idx} — MSE: {result['mse']:.4f}, MAE: {result['mae']:.4f}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"episode_{ep_idx}_actions.png", dpi=150, facecolor="#1e1e2e")
    plt.close()


def plot_summary(results: list[dict], output_dir: Path):
    """Plot summary metrics across episodes."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1e1e2e")

    ep_indices = [r["episode_idx"] for r in results]
    mses = [r["mse"] for r in results]
    maes = [r["mae"] for r in results]
    n_dims = len(results[0]["per_dim_mse"])

    # Bar chart: MSE per episode
    ax = axes[0]
    ax.set_facecolor("#1e1e2e")
    ax.bar(range(len(ep_indices)), mses, color="#89b4fa", alpha=0.85)
    ax.set_xticks(range(len(ep_indices)))
    ax.set_xticklabels([str(e) for e in ep_indices], color="white")
    ax.set_ylabel("MSE", color="white")
    ax.set_title("MSE per Episode", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#45475a")

    # Bar chart: MAE per episode
    ax = axes[1]
    ax.set_facecolor("#1e1e2e")
    ax.bar(range(len(ep_indices)), maes, color="#a6e3a1", alpha=0.85)
    ax.set_xticks(range(len(ep_indices)))
    ax.set_xticklabels([str(e) for e in ep_indices], color="white")
    ax.set_ylabel("MAE", color="white")
    ax.set_title("MAE per Episode", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#45475a")

    # Per-dimension MSE (averaged across episodes)
    ax = axes[2]
    ax.set_facecolor("#1e1e2e")
    avg_dim_mse = np.mean([r["per_dim_mse"] for r in results], axis=0)
    ax.bar(range(n_dims), avg_dim_mse, color="#cba6f7", alpha=0.85)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f"Dim {i}" for i in range(n_dims)], color="white", fontsize=9)
    ax.set_ylabel("MSE", color="white")
    ax.set_title("Avg MSE per Action Dimension", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#45475a")

    plt.tight_layout()
    plt.savefig(output_dir / "eval_summary.png", dpi=150, facecolor="#1e1e2e")
    plt.close()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # 1. Load policy from checkpoint
    print(f"Loading policy from: {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(str(args.checkpoint))
    policy.to(DEVICE)
    policy.eval()
    print(f"Policy loaded, params: {sum(p.numel() for p in policy.parameters()):,}")

    # 2. Build delta_timestamps from policy config
    cfg = policy.config
    print(f"Policy config: n_obs_steps={cfg.n_obs_steps}, horizon={cfg.horizon}, n_action_steps={cfg.n_action_steps}")

    # Load dataset first without delta_timestamps to get fps
    ds_meta = LeRobotDataset(repo_id=args.dataset)
    fps = ds_meta.fps
    num_episodes = len(ds_meta.meta.episodes)

    obs_delta_ts = [i / fps for i in cfg.observation_delta_indices]
    act_delta_ts = [i / fps for i in cfg.action_delta_indices]

    features = dataset_to_policy_features(ds_meta.meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    delta_timestamps = {}
    for key in input_features:
        delta_timestamps[key] = obs_delta_ts
    delta_timestamps["action"] = act_delta_ts
    del ds_meta

    # 3. Load dataset with delta_timestamps
    print(f"Loading dataset with delta_timestamps...")
    dataset = LeRobotDataset(repo_id=args.dataset, delta_timestamps=delta_timestamps)

    # 4. Build preprocessor/postprocessor (create from scratch, not from pretrained path)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    # 5. Select episodes to evaluate
    if args.episodes:
        episode_indices = [int(x) for x in args.episodes.split(",")]
    else:
        # 5 evenly spaced episodes
        episode_indices = np.linspace(0, num_episodes - 1, min(5, num_episodes), dtype=int).tolist()

    print(f"Evaluating {len(episode_indices)} episodes: {episode_indices}")

    # 6. Run evaluation
    results = []
    for ep_idx in episode_indices:
        print(f"  Episode {ep_idx}...", end=" ", flush=True)
        result = evaluate_episode(policy, dataset, ep_idx, preprocessor, postprocessor, DEVICE)
        results.append(result)
        print(f"MSE={result['mse']:.4f}, MAE={result['mae']:.4f} ({result['length']} steps)")

        # Plot per-episode actions
        plot_episode_actions(result, args.output)

    # 7. Summary
    avg_mse = np.mean([r["mse"] for r in results])
    avg_mae = np.mean([r["mae"] for r in results])
    print(f"\nOverall: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}")

    # Plot summary
    plot_summary(results, args.output)

    # Save metrics JSON
    metrics = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "avg_mse": round(avg_mse, 6),
        "avg_mae": round(avg_mae, 6),
        "episodes": [
            {
                "episode_idx": r["episode_idx"],
                "length": r["length"],
                "mse": round(r["mse"], 6),
                "mae": round(r["mae"], 6),
                "per_dim_mse": [round(x, 6) for x in r["per_dim_mse"]],
            }
            for r in results
        ],
    }
    with open(args.output / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPlots saved to: {args.output}")
    print(f"Metrics saved to: {args.output / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
