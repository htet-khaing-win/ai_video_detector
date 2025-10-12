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
from tqdm import tqdm

def cache_from_frame_folder(
    frames_root="data/processed/frames", 
    cache_root="data/processed/genbuster_cached",
    max_frames=None
):
    os.makedirs(cache_root, exist_ok=True)
    video_dirs = [d for d in glob(os.path.join(frames_root, "*")) if os.path.isdir(d)]
    
    print(f"Found {len(video_dirs)} video directories")
    
    skipped = 0
    cached = 0
    errors = []
    
    for vd in tqdm(video_dirs, desc="Caching videos"):
        vid = os.path.basename(vd)
        cache_path = os.path.join(cache_root, f"{vid}.pt")
        
        # Skip if already cached
        if os.path.exists(cache_path):
            skipped += 1
            continue
        
        try:
            imgs = sorted(glob(os.path.join(vd, "*.jpg")))
            
            if len(imgs) == 0:
                errors.append((vid, "No frames found"))
                continue
            
            # Limit frames if specified
            if max_frames:
                imgs = imgs[:max_frames]
            
            arrs = []
            for p in imgs:
                img = cv2.imread(p)
                if img is None:
                    continue
                arrs.append(img)
            
            if len(arrs) == 0:
                errors.append((vid, "All frames failed to load"))
                continue
            
            arr_np = np.stack(arrs, axis=0)  # [T,H,W,C] uint8
            torch.save(arr_np, cache_path)
            cached += 1
            
        except Exception as e:
            errors.append((vid, str(e)))
    
    print(f"\n Cached: {cached}")
    print(f"⏭  Skipped (already cached): {skipped}")
    print(f" Errors: {len(errors)}")
    
    if errors:
        print("\nError details:")
        for vid, err in errors[:10]:  # Show first 10
            print(f"  {vid}: {err}")

if __name__ == "__main__":
    cache_from_frame_folder()
