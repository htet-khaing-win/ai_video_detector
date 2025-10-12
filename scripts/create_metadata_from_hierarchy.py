"""
Generate metadata CSVs from GenBuster hierarchical dataset structure.

Scans train/, test/, and benchmark/ splits and creates CSV files with:
- split: train/test/benchmark
- label: 0 (real) or 1 (fake)
- generator: real, cogvideox, easyanimate, etc.
- filepath: relative path to video file
- video_id: unique identifier (hash from filename)

Usage:
    python scripts/create_metadata_from_hierarchy.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import argparse
from tqdm import tqdm


def scan_split(split_dir, split_name):
    """
    Scan a single split directory (train/test/benchmark).
    
    Returns:
        List of dicts with metadata for each video
    """
    split_path = Path(split_dir)
    records = []
    
    print(f"\n[{split_name.upper()}] Scanning videos...")
    
    # Process real videos
    real_dir = split_path / "real"
    if real_dir.exists():
        real_videos = list(real_dir.glob("*.mp4"))
        print(f"  Found {len(real_videos)} real videos")
        
        for video_path in tqdm(real_videos, desc=f"  Processing real"):
            video_id = video_path.stem  # Filename without extension
            relative_path = video_path.relative_to(split_path.parent)
            
            records.append({
                "split": split_name,
                "label": 0,
                "generator": "real",
                "category": "real",
                "filepath": str(relative_path).replace("\\", "/"),
                "video_id": video_id
            })
    
    # Process fake videos (nested by generator)
    fake_dir = split_path / "fake"
    if fake_dir.exists():
        # Get all generator subdirectories
        generator_dirs = [d for d in fake_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(generator_dirs)} generator types: {[d.name for d in generator_dirs]}")
        
        for gen_dir in generator_dirs:
            generator_name = gen_dir.name
            fake_videos = list(gen_dir.glob("*.mp4"))
            
            print(f"    {generator_name}: {len(fake_videos)} videos")
            
            for video_path in tqdm(fake_videos, desc=f"    Processing {generator_name}", leave=False):
                video_id = video_path.stem
                relative_path = video_path.relative_to(split_path.parent)
                
                records.append({
                    "split": split_name,
                    "label": 1,
                    "generator": generator_name,
                    "category": "synthetic",
                    "filepath": str(relative_path).replace("\\", "/"),
                    "video_id": video_id
                })
    
    return records


def create_metadata(dataset_root, output_dir):
    """
    Scan entire dataset and create metadata CSVs for each split.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GenBuster Metadata Generator")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    
    splits = ["train", "test", "benchmark"]
    all_records = []
    split_stats = {}
    
    for split in splits:
        split_dir = dataset_root / split
        
        if not split_dir.exists():
            print(f"\n[WARNING] Split not found: {split_dir}")
            continue
        
        # Scan split
        records = scan_split(split_dir, split)
        all_records.extend(records)
        
        # Save split-specific CSV
        df_split = pd.DataFrame(records)
        output_csv = output_dir / f"{split}_metadata.csv"
        df_split.to_csv(output_csv, index=False)
        
        # Collect stats
        real_count = len(df_split[df_split["label"] == 0])
        fake_count = len(df_split[df_split["label"] == 1])
        split_stats[split] = {
            "total": len(df_split),
            "real": real_count,
            "fake": fake_count,
            "csv": output_csv
        }
        
        print(f"\n   Saved: {output_csv}")
        print(f"     Total: {len(df_split)} videos")
        print(f"     Real: {real_count}, Fake: {fake_count}")
        
        # Show generator distribution for fake videos
        if fake_count > 0:
            gen_dist = df_split[df_split["label"] == 1]["generator"].value_counts()
            print(f"     Generator distribution:")
            for gen, count in gen_dist.items():
                print(f"       {gen}: {count}")
    
    # Save combined metadata
    df_all = pd.DataFrame(all_records)
    combined_csv = output_dir / "combined_metadata.csv"
    df_all.to_csv(combined_csv, index=False)
    
    print("\n" + "=" * 60)
    print("Metadata Generation Complete!")
    print("=" * 60)
    
    # Summary table
    print("\nSummary:")
    print(f"{'Split':<12} {'Total':>8} {'Real':>8} {'Fake':>8} {'Balance':>10}")
    print("-" * 50)
    
    for split, stats in split_stats.items():
        balance = f"{stats['real']/stats['total']*100:.1f}% real"
        print(f"{split:<12} {stats['total']:>8} {stats['real']:>8} {stats['fake']:>8} {balance:>10}")
    
    total_videos = sum(s["total"] for s in split_stats.values())
    total_real = sum(s["real"] for s in split_stats.values())
    total_fake = sum(s["fake"] for s in split_stats.values())
    
    print("-" * 50)
    print(f"{'TOTAL':<12} {total_videos:>8} {total_real:>8} {total_fake:>8} {total_real/total_videos*100:.1f}% real")
    
    print(f"\nOutput files:")
    for split, stats in split_stats.items():
        print(f"  - {stats['csv']}")
    print(f"  - {combined_csv}")
    
    print("\nNext steps:")
    print("  1. Run: python scripts/run_full_pipeline.py --skip_download")
    print("  2. Or extract frames: python scripts/extract_frames.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata CSVs from GenBuster hierarchical dataset"
    )
    parser.add_argument(
        "--dataset_root",
        default="data/raw/genbuster-200k-mini/GenBuster-200K-mini",
        help="Root directory of GenBuster dataset (contains train/test/benchmark/)"
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits",
        help="Output directory for metadata CSVs"
    )
    
    args = parser.parse_args()
    
    create_metadata(args.dataset_root, args.output_dir)