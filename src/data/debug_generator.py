import os
import cv2
import numpy as np

def generate_debug_videos(out_dir="data/debug/videos", num_videos=100, res=128, frames=8, fps=8):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_videos):
        label = "real" if i < num_videos // 2 else "fake"
        name = f"{label}_{i:03d}.mp4"
        path = os.path.join(out_dir, name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (res, res))
        rng = np.random.RandomState(i)
        for _ in range(frames):
            if label == "real":
                frame = (rng.rand(res, res, 3) * 255).astype(np.uint8)
            else:
                # fake: add simple patterns to simulate artifacts
                frame = (rng.rand(res, res, 3) * 255).astype(np.uint8)
                frame[::8, ::8, :] = 255  # grid artifact
            writer.write(frame)
        writer.release()
    print(f"Generated {num_videos} debug videos in {out_dir}")

if __name__ == "__main__":
    generate_debug_videos()
