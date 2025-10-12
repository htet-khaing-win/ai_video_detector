"""
Run full GenBuster preprocessing pipeline for hierarchical dataset.

Steps:
1. Generate metadata CSVs from directory structure
2. Extract frames from all videos (train/test/benchmark)
3. Cache frames to .pt tensors

Usage:
    # Full pipeline
    python scripts/run_full_pipeline_v2.py
    
    # Skip metadata generation
    python scripts/run_full_pipeline_v2.py --skip_metadata
    
    # Process only specific splits
    python scripts/run_full_pipeline_v2.py --splits train test
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from tqdm import tqdm
from src.data.frame_extractor import FrameExtractor
from src.data.cache_preprocessed import cache_from_frame_folder


def generate_metadata(dataset_root, output_dir):
    """Step 1: Generate metadata CSVs"""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Metadata CSVs")
    print("=" * 60)
    
    from scripts.create_metadata_from_hierarchy import create_metadata
    create_metadata(dataset_root, output_dir)


def extract_frames_from_metadata(
    metadata_csv,
    dataset_root,
    frame_output_dir,
    fps_sample=1,
    max_frames=8,
    resize=224
):
    """Step 2: Extract frames for videos listed in metadata CSV"""
    
    print(f"\nProcessing: {metadata_csv}")
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"  Found {len(df)} videos")
    
    # Initialize extractor
    extractor = FrameExtractor(
        out_root=frame_output_dir,
        fps_sample=fps_sample,
        max_frames=max_frames,
        resize=resize
    )
    
    # Extract frames for each video
    successful = 0
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Extracting frames"):
        video_rel_path = row["filepath"]
        video_id = row["video_id"]
        video_full_path = Path(dataset_root) / video_rel_path
        
        if not video_full_path.exists():
            failed.append((video_id, f"File not found: {video_full_path}"))
            continue
        
        try:
            extractor.extract_to_folder(str(video_full_path), video_id=video_id)
            successful += 1
        except Exception as e:
            failed.append((video_id, str(e)))
    
    print(f"\n  Successfully extracted: {successful}/{len(df)}")
    if failed:
        print(f"   Failed: {len(failed)}")
        print(f"     (First 5 errors:)")
        for vid_id, error in failed[:5]:
            print(f"       {vid_id}: {error}")
    
    return successful, failed


def cache_frames_for_split(frame_dir, cache_dir, split_name):
    """Step 3: Cache frames to .pt tensors for a specific split"""
    
    split_frame_dir = Path(frame_dir) / split_name
    split_cache_dir = Path(cache_dir) / split_name
    
    if not split_frame_dir.exists():
        print(f"   Frame directory not found: {split_frame_dir}")
        return
    
    print(f"\n  Caching {split_name} split...")
    print(f"    Frames: {split_frame_dir}")
    print(f"    Cache:  {split_cache_dir}")
    
    cache_from_frame_folder(
        frames_root=str(split_frame_dir),
        cache_root=str(split_cache_dir)
    )


def run_pipeline(
    dataset_root="data/raw/genbuster-200k-mini/GenBuster-200K-mini",
    metadata_dir="data/splits",
    frame_dir="data/processed/frames",
    cache_dir="data/processed/genbuster_cached",
    splits=None,
    fps_sample=1,
    max_frames=8,
    resize=224,
    skip_metadata=False,
    skip_extraction=False,
    skip_caching=False
):
    """
    Run full preprocessing pipeline.
    """
    
    print("=" * 60)
    print("GenBuster Preprocessing Pipeline v2.0")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Metadata dir: {metadata_dir}")
    print(f"Frame dir:    {frame_dir}")
    print(f"Cache dir:    {cache_dir}")
    
    if splits is None:
        splits = ["train", "test", "benchmark"]
    
    print(f"Processing splits: {', '.join(splits)}")
    
    # Step 1: Generate metadata
    if not skip_metadata:
        generate_metadata(dataset_root, metadata_dir)
    else:
        print("\n[STEP 1] Skipping metadata generation (--skip_metadata flag)")
    
    # Step 2: Extract frames
    if not skip_extraction:
        print("\n" + "=" * 60)
        print("STEP 2: Extracting Frames")
        print("=" * 60)
        
        for split in splits:
            metadata_csv = Path(metadata_dir) / f"{split}_metadata.csv"
            
            if not metadata_csv.exists():
                print(f"\n  Metadata not found: {metadata_csv}")
                continue
            
            split_frame_dir = Path(frame_dir) / split
            
            extract_frames_from_metadata(
                metadata_csv=metadata_csv,
                dataset_root=dataset_root,
                frame_output_dir=str(split_frame_dir),
                fps_sample=fps_sample,
                max_frames=max_frames,
                resize=resize
            )
    else:
        print("\n[STEP 2] Skipping frame extraction (--skip_extraction flag)")
    
    # Step 3: Cache frames
    if not skip_caching:
        print("\n" + "=" * 60)
        print("STEP 3: Caching Frames to .pt Tensors")
        print("=" * 60)
        
        for split in splits:
            cache_frames_for_split(frame_dir, cache_dir, split)
    else:
        print("\n[STEP 3] Skipping caching (--skip_caching flag)")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    # Count outputs
    for split in splits:
        metadata_csv = Path(metadata_dir) / f"{split}_metadata.csv"
        split_cache_dir = Path(cache_dir) / split
        
        if metadata_csv.exists():
            df = pd.read_csv(metadata_csv)
            print(f"\n{split.upper()}:")
            print(f"  Metadata: {len(df)} videos")
        
        if split_cache_dir.exists():
            cache_files = list(split_cache_dir.glob("**/*.pt"))
            print(f"  Cached:   {len(cache_files)} tensors")
    
    print("\nNext steps:")
    print("  1. Verify data: python scripts/verify_data.py")
    print("  2. Start training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full GenBuster preprocessing pipeline"
    )
    parser.add_argument(
        "--dataset_root",
        default="data/raw/genbuster-200k-mini/GenBuster-200K-mini",
        help="Root directory of GenBuster dataset"
    )
    parser.add_argument(
        "--metadata_dir",
        default="data/splits",
        help="Output directory for metadata CSVs"
    )
    parser.add_argument(
        "--frame_dir",
        default="data/processed/frames",
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--cache_dir",
        default="data/processed/genbuster_cached",
        help="Output directory for cached .pt tensors"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=["train", "test", "benchmark"],
        help="Specific splits to process (default: all)"
    )
    parser.add_argument(
        "--fps_sample",
        type=int,
        default=1,
        help="Frames per second to extract"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=8,
        help="Maximum frames per video"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        choices=[128, 224, 256],
        help="Target frame size"
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        help="Skip metadata generation"
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip frame extraction"
    )
    parser.add_argument(
        "--skip_caching",
        action="store_true",
        help="Skip caching to .pt"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        dataset_root=args.dataset_root,
        metadata_dir=args.metadata_dir,
        frame_dir=args.frame_dir,
        cache_dir=args.cache_dir,
        splits=args.splits,
        fps_sample=args.fps_sample,
        max_frames=args.max_frames,
        resize=args.resize,
        skip_metadata=args.skip_metadata,
        skip_extraction=args.skip_extraction,
        skip_caching=args.skip_caching
    )