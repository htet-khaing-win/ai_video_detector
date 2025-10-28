import os
import csv
from pathlib import Path
from tqdm import tqdm

def create_dfd_metadata(
    dataset_root="D:/DFD",
    output_dir="C:/Personal project/ai_video_detector/data/splits"
):
    print("\n" + "="*60)
    print("Creating Metadata from DFD Directory Structure")
    print("="*60)

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_dir = dataset_root / "DFD_original sequences"
    fake_dir = dataset_root / "DFD_manipulated_sequences/DFD_manipulated_sequences"

    if not real_dir.exists():
        print(f" Missing real folder: {real_dir}")
        return None
    if not fake_dir.exists():
        print(f" Missing fake folder: {fake_dir}")
        return None

    all_metadata = []
    seen_ids = set()

    # ============ REAL videos ============
    real_videos = [
        f for f in real_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ]
    print(f"Found {len(real_videos)} real videos")

    for video_path in tqdm(real_videos, desc="Processing real", leave=False):
        video_id = video_path.stem
        if video_id in seen_ids:
            video_id = f"real_{video_id}_{hash(str(video_path)) % 10000}"
        seen_ids.add(video_id)

        all_metadata.append({
            "video_id": video_id,
            "filepath": str(video_path.relative_to(dataset_root)).replace("\\", "/"),
            "absolute_path": str(video_path.resolve()).replace("\\", "/"),
            "label": 0,
            "split": "train",          # you can reassign later if needed
            "dataset": "dfd",
            "generator": "real"
        })

    # ============ FAKE videos ============
    fake_videos = [
        f for f in fake_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ]
    print(f"Found {len(fake_videos)} fake videos")

    for video_path in tqdm(fake_videos, desc="Processing fake", leave=False):
        video_id = video_path.stem
        if video_id in seen_ids:
            video_id = f"fake_{video_id}_{hash(str(video_path)) % 10000}"
        seen_ids.add(video_id)

        all_metadata.append({
            "video_id": video_id,
            "filepath": str(video_path.relative_to(dataset_root)).replace("\\", "/"),
            "absolute_path": str(video_path.resolve()).replace("\\", "/"),
            "label": 1,
            "split": "train",
            "dataset": "dfd",
            "generator": "deepfake"  # all manipulated by DeepFake method
        })

    # Save CSV
    if all_metadata:
        csv_path = output_dir / "dfd_metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)

        real_count = sum(1 for m in all_metadata if m["label"] == 0)
        fake_count = sum(1 for m in all_metadata if m["label"] == 1)

        print("\n" + "="*60)
        print(f" DFD Metadata saved to {csv_path}")
        print(f"Total videos: {len(all_metadata)} ({real_count} real, {fake_count} fake)")
        print("="*60)

    else:
        print(" No videos found in DFD dataset!")

    return output_dir


if __name__ == "__main__":
    create_dfd_metadata()
