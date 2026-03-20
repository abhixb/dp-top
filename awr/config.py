from pathlib import Path

# ─── Dataset ───
DATASET_REPO_ID = "lerobot/svla_so100_pickplace"
CAMERA_KEY = "observation.images.top"
# NOTE: Run `python -m awr.score_dataset --list-keys` to check available keys.
# This dataset also has "observation.images.wrist"
INSTRUCTION = "Pick up the cube and place it in the box."

# ─── Scoring ───
NUM_EVAL_FRAMES = 20
MAX_FRAMES_PER_PREFIX = 16
MAX_EPISODE_FRAMES = 450  # Subsample longer episodes to avoid OOM on 12GB GPUs

# ─── AWR (paper Section 5.3 defaults) ───
TAU = 2.0
DELTA_MAX = 2.0
WEIGHT_CLIP_MIN = 0.1
SUBTRACT_MEAN = True

# ─── Training ───
POLICY_TYPE = "diffusion"
LEARNING_RATE = 1e-4
NUM_TRAIN_STEPS = 100000
BATCH_SIZE = 8
SAVE_EVERY = 5000
LOG_EVERY = 100
DEVICE = "cuda:0"

# ─── Output ───
OUTPUT_DIR = Path(__file__).parent / "outputs"
