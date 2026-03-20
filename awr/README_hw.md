# AWR from Real Rollouts

The Hub dataset was all clean successes — AWR had nothing to differentiate.
This pipeline deploys the BC policy, collects messy rollouts, and uses
TOPReward to extract the good from the bad via AWR.

## Quick Start

```bash
# Full round: collect → score → build weights → fine-tune
python -m awr.run_loop --round 1

# Each subsequent round uses the previous round's improved policy
python -m awr.run_loop --round 2
python -m awr.run_loop --round 3
```

## Step by Step

```bash
# 1. Deploy policy on SO-100, record rollouts
python -m awr.collect_rollouts --round 1
# Add --dry-run to preview the command without executing

# 2. Score rollouts with TOPReward
python -m awr.score_rollouts --round 1

# 3. Build combined weights (hub=fixed 1.5, rollouts=AWR scored)
python -m awr.build_weighted_dataset --round 1
# Check plots in awr/outputs/hw/round_001/plots/

# 4. Fine-tune
python -m awr.awr_finetune --round 1
```

## Inspect Before Training

```bash
python -m awr.run_loop --round 1 --skip-training
```

Review `awr/outputs/hw/round_001/plots/`:
- `weight_distribution.png` — should show real spread, not a spike at 2.0
- `weight_heatmap.png` — should show red/yellow/green mix across episodes
- `progress_curves.png` — should show varied curves (some ramp up, some plateau)

If weights are still uniform, rollouts are too similar — try harder initial conditions.

## Config

Edit `awr/config_hw.py`:
- `FOLLOWER_PORT` — your SO-100 USB port
- `NUM_ROLLOUTS` — episodes per round (default: 30)
- `HUB_DEMO_FIXED_WEIGHT` — weight for hub demos (default: 1.5)
- `TAU`, `DELTA_MAX` — AWR temperature and clipping
- `LEARNING_RATE` — lower than initial BC (default: 5e-6)

## Output Structure

```
awr/outputs/hw/
├── round_001/
│   ├── rollouts/             # Recorded LeRobot dataset
│   ├── scores/               # TOPReward per-episode JSONs
│   ├── plots/                # Inspection plots
│   ├── weights.npy           # Combined hub + rollout weights
│   ├── advantages.json       # Full metadata
│   └── collection_meta.json  # Rollout collection metadata
├── round_002/
│   └── ...
├── checkpoints/
│   ├── round_001/final/      # AWR fine-tuned policy
│   ├── round_002/final/
│   └── ...
└── log.json                  # Tracks rounds
```

## How It Works

1. **Hub demos** get fixed weight (1.5). They're all clean successes — no need to score.
2. **Rollouts** get AWR weights from TOPReward. Good moments → high weight, bad → low.
3. **Mean subtraction** uses rollout deltas only, so hub demos don't skew the baseline.
4. **Each round** deploys the previous round's checkpoint, collecting increasingly better data.
