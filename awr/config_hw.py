"""Hardware + rollout settings for AWR fine-tuning from real SO-100 rollouts."""

from pathlib import Path

# ─── Hardware ───
LEADER_PORT = "/dev/ttyACM0"
FOLLOWER_PORT = "/dev/ttyACM1"

# ─── Rollout Collection ───
NUM_ROLLOUTS = 30
EPISODE_MAX_STEPS = 600  # ~20s at 30fps
FPS = 30
RESET_TIME_S = 15

# ─── Paths ───
BC_CHECKPOINT = Path("awr/outputs/checkpoints/diffusion_awr/final")
HUB_DATASET = "lerobot/svla_so100_pickplace"

# ─── Scoring ───
INSTRUCTION = "Pick up the object and place it at the target location"
CAMERA_KEY = "observation.images.top"
NUM_EVAL_FRAMES = 20
MAX_FRAMES_PER_PREFIX = 16
MAX_EPISODE_FRAMES = 450

# ─── AWR ───
TAU = 2.0
DELTA_MAX = 2.0
WEIGHT_CLIP_MIN = 0.1
SUBTRACT_MEAN = True

# Hub demos are all clean successes — assign fixed weight instead of scoring.
HUB_DEMO_FIXED_WEIGHT = 1.5

# ─── Training ───
LEARNING_RATE = 5e-6
NUM_TRAIN_STEPS = 3000
BATCH_SIZE = 8
DEVICE = "cuda:0"

# ─── Output ───
OUTPUT_DIR = Path("awr/outputs/hw")
