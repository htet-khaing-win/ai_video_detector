"""
PyTorch Dataset and utilities for GenBuster-only pipeline.

Key features:
- Reads sampled metadata CSV from data/splits/
- Uses cached preprocessed frames at data/processed/genbuster_cached/<split>/<video_id>.pt
- Lightweight augmentations: horizontal flip, small color jitter, random crop/resize to 224
- DataLoader params optimized for RTX 4070: batch_size 4-8 for clip models, num_workers=4, pin_memory=True, prefetch_factor=2
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

def default_transforms(resize=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomResizedCrop(resize, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

class GenBusterDataset(Dataset):
    def __init__(self, metadata_csv, cache_root="data/processed/genbuster_cached",
                 max_frames=8, resize=224, transforms=None, clip_mode=True):
        import csv
        self.cache_root = cache_root
        self.max_frames = max_frames
        self.resize = resize
        self.clip_mode = clip_mode
        self.transforms = transforms or default_transforms(resize)
        self.samples = []

        # Auto-detect split from CSV filename
        csv_name = Path(metadata_csv).stem
        split = csv_name.replace('_metadata', '') if '_metadata' in csv_name else None

        # load metadata CSV (robust to extra columns)
        with open(metadata_csv, "r") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                file_col = None
                for candidate in ["file", "filepath", "url", "path"]:
                    if candidate in row and row[candidate]:
                        file_col = candidate
                        break
                if file_col is None:
                    continue
                label = int(row.get("label", 0))
                
                # Get video_id from CSV column or extract from filepath
                vid_id = row.get("video_id") or os.path.splitext(os.path.basename(row[file_col]))[0]
                
                # Include split subdirectory in cache path
                if split:
                    cache_path = os.path.join(cache_root, split, f"{vid_id}.pt")
                else:
                    cache_path = os.path.join(cache_root, f"{vid_id}.pt")
                
                self.samples.append({"cache": cache_path, "label": label, "vid": vid_id})

        if len(self.samples) == 0:
            raise RuntimeError("No metadata samples found. Check CSV and file columns.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cache_path = s["cache"]
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache missing: {cache_path}. Run caching step.")
        data = torch.load(cache_path, weights_only=False)

        # ensure torch tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # ensure at least max_frames frames
        T = data.shape[0]
        if T < self.max_frames:
            pad = self.max_frames - T
            last = data[-1:].repeat(pad, 1, 1, 1)
            data = torch.cat([data, last], dim=0)
        else:
            data = data[:self.max_frames]

        # apply transforms per frame
        frames = []
        for t in range(self.max_frames):
            img = data[t].numpy().astype("uint8")
            img_t = self.transforms(img)
            frames.append(img_t)
        frames = torch.stack(frames, dim=1)
        label = torch.tensor(s["label"], dtype=torch.long)

        if self.clip_mode:
            return frames, label
        mid = frames[:, self.max_frames // 2, :, :].clone()
        return mid, label

def create_dataloader(metadata_csv, batch_size=4, num_workers=4, clip_mode=True,
                      shuffle=True, cache_root="data/processed/genbuster_cached"):
    ds = GenBusterDataset(metadata_csv, cache_root=cache_root, clip_mode=clip_mode)

    if num_workers == 0:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return loader