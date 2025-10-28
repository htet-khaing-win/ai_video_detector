import os
import csv
from pathlib import Path
from tqdm import tqdm

def create_metadata_from_directory(
    dataset_root="D:/GenBuster200k/raw/GenBuster-200K/GenBuster-200K",
    output_dir="C:/Personal project/ai_video_detector/data/splits"
):
   
    print("\n" + "="*60)
    print("Creating Metadata from GenBuster Directory Structure")
    print("="*60)

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify dataset root exists
    if not dataset_root.exists():
        print(f" Dataset root not found: {dataset_root}")
        return None

    all_metadata = []
    seen_video_ids = set()  # Track duplicates

    # Process splits
    for split in ['train', 'test', 'benchmark']:
        split_dir = dataset_root / split
        
        if not split_dir.exists():
            print(f"  {split} folder not found: {split_dir}")
            continue

        print(f"\nProcessing {split.upper()} split...")

        # ============ Real videos ============
        real_dir = split_dir / 'real'
        if real_dir.exists():
            real_videos = [
                f for f in real_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            ]
            print(f"  real: {len(real_videos)} videos")
            
            for video_path in tqdm(real_videos, desc="    Processing real", leave=False):
                # Generate unique video_id
                video_id = video_path.stem
                if video_id in seen_video_ids:
                    video_id = f"real_{video_path.stem}_{hash(str(video_path)) % 10000}"
                seen_video_ids.add(video_id)
                
                all_metadata.append({
                    'video_id': video_id,
                    'filepath': str(video_path.relative_to(dataset_root)).replace('\\', '/'),
                    'absolute_path': str(video_path.resolve()).replace('\\', '/'),
                    'label': 0,
                    'split': split,
                    'dataset': 'genbuster',
                    'generator': 'real'
                })
        else:
            print(f"  real folder missing: {real_dir}")

        # ============ Fake videos (recursive) ============
        fake_dir = split_dir / 'fake'
        if fake_dir.exists():
            fake_videos = []
            
            # Walk through fake directory (may have generator subfolders)
            for root, _, files in os.walk(fake_dir):
                root_path = Path(root)
                
                for filename in files:
                    if Path(filename).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                        video_path = root_path / filename
                        fake_videos.append(video_path)
            
            print(f"  fake: {len(fake_videos)} videos")
            
            for video_path in tqdm(fake_videos, desc="    Processing fake", leave=False):
                # Detect generator type from parent folder
                # e.g., fake/sora/video.mp4 -> generator: "sora"
                relative_to_fake = video_path.relative_to(fake_dir)
                
                if len(relative_to_fake.parts) > 1:
                    # Video is in subfolder (e.g., sora/video.mp4)
                    generator = relative_to_fake.parts[0]
                else:
                    # Video directly in fake/ folder
                    generator = 'ai_generated'
                
                # Generate unique video_id
                video_id = video_path.stem
                if video_id in seen_video_ids:
                    video_id = f"{generator}_{video_path.stem}_{hash(str(video_path)) % 10000}"
                seen_video_ids.add(video_id)
                
                all_metadata.append({
                    'video_id': video_id,
                    'filepath': str(video_path.relative_to(dataset_root)).replace('\\', '/'),
                    'absolute_path': str(video_path.resolve()).replace('\\', '/'),
                    'label': 1,
                    'split': split,
                    'dataset': 'genbuster',
                    'generator': generator
                })
        else:
            print(f"  fake folder missing: {fake_dir}")

        # Save split-specific CSV
        split_metadata = [m for m in all_metadata if m['split'] == split]
        
        if split_metadata:
            split_csv = output_dir / f"{split}_metadata.csv"
            with open(split_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=split_metadata[0].keys())
                writer.writeheader()
                writer.writerows(split_metadata)
            
            real_count = sum(1 for m in split_metadata if m['label'] == 0)
            fake_count = sum(1 for m in split_metadata if m['label'] == 1)
            
            print(f"  {split}_metadata.csv saved")
            print(f"    Total: {len(split_metadata)} ({real_count} real, {fake_count} fake)")

    # Save unified CSV
    if all_metadata:
        unified_csv = output_dir / "unified_metadata.csv"
        with open(unified_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)

        print(f"\n{'='*60}")
        print(" Metadata Creation Complete!")
        print(f"{'='*60}")
        print(f"Total videos: {len(all_metadata)}")
        print(f"Output directory: {output_dir}")
        
        # Summary by split
        print(f"\nSummary by split:")
        for split in ['train', 'test', 'benchmark']:
            split_data = [m for m in all_metadata if m['split'] == split]
            if split_data:
                real = sum(1 for m in split_data if m['label'] == 0)
                fake = sum(1 for m in split_data if m['label'] == 1)
                print(f"  {split:10s}: {len(split_data):6d} total ({real:6d} real, {fake:6d} fake)")
        
        # Summary by generator
        print(f"\nFake videos by generator:")
        from collections import Counter
        generators = Counter(m['generator'] for m in all_metadata if m['label'] == 1)
        for gen, count in generators.most_common():
            print(f"  {gen:20s}: {count:6d} videos")
        
        print(f"\n{'='*60}")

    else:
        print("\n No videos found! Check your directory structure.")
        print(f"Expected structure:")
        print(f"  {dataset_root}/")
        print(f"    train/real/*.mp4")
        print(f"    train/fake/**/*.mp4")
        print(f"    test/real/*.mp4")
        print(f"    test/fake/**/*.mp4")
        print(f"    benchmark/real/*.mp4")
        print(f"    benchmark/fake/**/*.mp4")

    return output_dir


if __name__ == "__main__":
    create_metadata_from_directory()