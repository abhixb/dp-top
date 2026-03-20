"""Extract frames from nyudoor episodes 0 and 1 using LeRobot."""
import numpy as np
from pathlib import Path
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT_DIR = Path("results/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = LeRobotDataset("lerobot/nyu_door_opening_surprising_effectiveness", force_cache_sync=True)
ep_indices = np.array(ds.hf_dataset["episode_index"])

for ep_idx in [0, 1]:
    ep_dir = OUT_DIR / f"episode_{ep_idx}"
    ep_dir.mkdir(exist_ok=True)

    indices = np.where(ep_indices == ep_idx)[0]
    print(f"\nEpisode {ep_idx}: {len(indices)} frames")

    frames = []
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        img_tensor = item["observation.images.image"]  # (C, H, W)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(img_np)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_dir / f"frame_{i:03d}.jpg"), bgr)

    h, w = frames[0].shape[:2]
    video_path = str(OUT_DIR / f"episode_{ep_idx}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved {len(frames)} frames to {ep_dir}/")
    print(f"  Saved video to {video_path}")

print("\nDone!")
