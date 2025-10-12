"""
Verify data integrity for GenBuster preprocessed dataset.

Checks:
1. All videos in metadata have corresponding cached .pt files
2. All cached .pt files can be loaded without errors
3. Tensor shapes are correct
4. Reports statistics per split

Usage:
    python scripts/verify_data_integrity.py
    python scripts/verify_data_integrity.py --splits train test
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import argparse
from tqdm import tqdm


def verify_split(metadata_csv, cache_root, split_name, expected_shape=(8, 224, 224, 3)):
    """
    Verify a single split's cached data.
    
    Returns:
        dict with verification results
    """
    print(f"\n{'='*60}")
    print(f"Verifying {split_name.upper()} Split")
    print(f"{'='*60}")
    print(f"Metadata: {metadata_csv}")
    print(f"Cache:    {cache_root}")
    
    if not metadata_csv.exists():
        print(f" Metadata CSV not found!")
        return None
    
    if not cache_root.exists():
        print(f" Cache directory not found!")
        return None
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"\nTotal videos in metadata: {len(df)}")
    
    # Check label distribution
    if 'label' in df.columns:
        real_count = len(df[df['label'] == 0])
        fake_count = len(df[df['label'] == 1])
        print(f"  Real: {real_count} ({real_count/len(df)*100:.1f}%)")
        print(f"  Fake: {fake_count} ({fake_count/len(df)*100:.1f}%)")
    
    # Check generator distribution
    if 'generator' in df.columns:
        gen_dist = df['generator'].value_counts()
        print(f"\nGenerator distribution:")
        for gen, count in gen_dist.items():
            print(f"  {gen}: {count}")
    
    # Verify cached files
    missing = []
    corrupted = []
    shape_mismatch = []
    valid = []
    
    print(f"\nVerifying cached .pt files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking cache"):
        video_id = row["video_id"]
        cache_path = cache_root / f"{video_id}.pt"
        
        # Check if file exists
        if not cache_path.exists():
            missing.append(video_id)
            continue
        
        # Try to load
        try:
            data = torch.load(cache_path, map_location="cpu")
            
            # Check shape
            if hasattr(data, 'shape'):
                if data.shape != expected_shape:
                    shape_mismatch.append((video_id, data.shape))
                else:
                    valid.append(video_id)
            else:
                corrupted.append((video_id, "Not a tensor"))
        
        except Exception as e:
            corrupted.append((video_id, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Verification Results - {split_name.upper()}")
    print(f"{'='*60}")
    print(f" Valid:          {len(valid)}/{len(df)} ({len(valid)/len(df)*100:.1f}%)")
    print(f" Missing:        {len(missing)}")
    print(f" Corrupted:      {len(corrupted)}")
    print(f"  Shape mismatch: {len(shape_mismatch)}")
    
    # Show examples
    if missing:
        print(f"\nMissing cache files (first 5):")
        for vid_id in missing[:5]:
            print(f"  - {vid_id}.pt")
    
    if corrupted:
        print(f"\nCorrupted files (first 5):")
        for vid_id, error in corrupted[:5]:
            print(f"  - {vid_id}.pt: {error}")
    
    if shape_mismatch:
        print(f"\nShape mismatches (first 5):")
        for vid_id, shape in shape_mismatch[:5]:
            print(f"  - {vid_id}.pt: {shape} (expected {expected_shape})")
    
    # Return stats
    return {
        "split": split_name,
        "total": len(df),
        "valid": len(valid),
        "missing": len(missing),
        "corrupted": len(corrupted),
        "shape_mismatch": len(shape_mismatch),
        "success_rate": len(valid) / len(df) * 100
    }


def verify_all_splits(
    metadata_dir="data/splits",
    cache_dir="data/processed/genbuster_cached",
    splits=None,
    expected_shape=(8, 224, 224, 3)
):
    """
    Verify all splits in the dataset.
    """
    metadata_dir = Path(metadata_dir)
    cache_dir = Path(cache_dir)
    
    if splits is None:
        splits = ["train", "test", "benchmark"]
    
    print("=" * 60)
    print("GenBuster Data Integrity Verification")
    print("=" * 60)
    print(f"Metadata directory: {metadata_dir}")
    print(f"Cache directory:    {cache_dir}")
    print(f"Expected shape:     {expected_shape}")
    
    results = []
    
    for split in splits:
        metadata_csv = metadata_dir / f"{split}_metadata.csv"
        split_cache_dir = cache_dir / split
        
        result = verify_split(
            metadata_csv=metadata_csv,
            cache_root=split_cache_dir,
            split_name=split,
            expected_shape=expected_shape
        )
        
        if result:
            results.append(result)
    
    # Overall summary
    if results:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        print(f"{'Split':<12} {'Total':>8} {'Valid':>8} {'Missing':>8} {'Success Rate':>12}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['split']:<12} {r['total']:>8} {r['valid']:>8} "
                  f"{r['missing']:>8} {r['success_rate']:>11.1f}%")
        
        # Totals
        total_videos = sum(r['total'] for r in results)
        total_valid = sum(r['valid'] for r in results)
        total_missing = sum(r['missing'] for r in results)
        total_corrupted = sum(r['corrupted'] for r in results)
        
        print("-" * 60)
        print(f"{'TOTAL':<12} {total_videos:>8} {total_valid:>8} "
              f"{total_missing:>8} {total_valid/total_videos*100:>11.1f}%")
        
        print(f"\nTotal corrupted: {total_corrupted}")
        
        # Pass/Fail criteria
        overall_success_rate = total_valid / total_videos * 100
        
        print("\n" + "=" * 60)
        if overall_success_rate >= 95:
            print(" PASS: Dataset integrity is excellent (≥95% valid)")
        elif overall_success_rate >= 90:
            print("  WARNING: Dataset has some issues (90-95% valid)")
        else:
            print(" FAIL: Dataset has significant issues (<90% valid)")
        print("=" * 60)
        
        # Recommendations
        if total_missing > 0:
            print("\nRecommendations:")
            print("  1. Re-run frame extraction for missing videos")
            print("  2. Re-run caching step")
        
        if total_corrupted > 0:
            print("\nCorrupted files detected:")
            print("  1. Check disk space")
            print("  2. Re-run caching for affected videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify GenBuster preprocessed dataset integrity"
    )
    parser.add_argument(
        "--metadata_dir",
        default="data/splits",
        help="Directory containing metadata CSVs"
    )
    parser.add_argument(
        "--cache_dir",
        default="data/processed/genbuster_cached",
        help="Directory containing cached .pt files"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=["train", "test", "benchmark"],
        help="Specific splits to verify (default: all)"
    )
    parser.add_argument(
        "--expected_frames",
        type=int,
        default=6,
        help="Expected number of frames per video"
    )
    parser.add_argument(
        "--expected_size",
        type=int,
        default=224,
        help="Expected frame size (height/width)"
    )
    
    args = parser.parse_args()
    
    expected_shape = (args.expected_frames, args.expected_size, args.expected_size, 3)
    
    verify_all_splits(
        metadata_dir=args.metadata_dir,
        cache_dir=args.cache_dir,
        splits=args.splits,
        expected_shape=expected_shape
    )