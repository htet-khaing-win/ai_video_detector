"""
Simple caching utility:
Given a folder of frame folders under data/processed/frames/<video_id>/*.jpg
it converts each to a torch tensor and saves to data/processed/genbuster_cached/<video_id>.pt
"""
import os
from glob import glob
import torch
import cv2
import numpy as np

def cache_from_frame_folder(frames_root="data/processed/frames", cache_root="data/processed/genbuster_cached"):
    os.makedirs(cache_root, exist_ok=True)
    video_dirs = [d for d in glob(os.path.join(frames_root, "*")) if os.path.isdir(d)]
    for vd in video_dirs:
        vid = os.path.basename(vd)
        cache_path = os.path.join(cache_root, f"{vid}.pt")
        if os.path.exists(cache_path):
            continue
        imgs = sorted(glob(os.path.join(vd, "*.jpg")))
        arrs = []
        for p in imgs:
            img = cv2.imread(p)
            if img is None:
                continue
            arrs.append(img)
        if len(arrs) == 0:
            continue
        arr_np = np.stack(arrs, axis=0)  # [T,H,W,C] uint8
        torch.save(arr_np, cache_path)
        print("Wrote", cache_path)

if __name__ == "__main__":
    cache_from_frame_folder()
