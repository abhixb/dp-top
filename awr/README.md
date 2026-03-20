# AWR Diffusion Policy with TOPReward

Train a diffusion policy on lerobot/svla_so100_pickplace with advantage-weighted regression.
TOPReward scores every demo, good actions get amplified, bad actions get suppressed.

## Quick Start

```bash
# Run everything:
python -m awr.pipeline

# Or step by step:
python -m awr.score_dataset              # ~3 min, scores all episodes
python -m awr.compute_advantages         # instant
python -m awr.inspect_scores             # check plots before training
python -m awr.weighted_trainer           # ~20 min for 5000 steps
```

## First Run

Before scoring, discover your dataset's camera key:
```bash
python -m awr.score_dataset --list-keys
```
Then update `CAMERA_KEY` in `awr/config.py` if needed.

## Config

Edit `awr/config.py` for:
- `INSTRUCTION`: natural language task description
- `TAU` / `DELTA_MAX`: controls weight spread (paper defaults: 2.0 / 2.0)
- `NUM_TRAIN_STEPS` / `LEARNING_RATE`: training hyperparameters
- `CAMERA_KEY`: which camera in the dataset to use

## Output

```
awr/outputs/
├── scores/           # Per-episode TOPReward scores
├── advantages/       # Per-step weights + flat weights.npy
├── plots/            # Sanity check visualizations
└── checkpoints/      # Trained policy
```
