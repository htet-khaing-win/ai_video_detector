"""
Download GenBuster-200K FULL dataset from Hugging Face.

Usage:
  python src/data/download_genbuster_200k.py

Changes from mini version:
- Downloads FULL dataset (no sampling)
- Parallel downloads (4 workers)
- Streaming extraction to avoid memory overflow
- Progress tracking with ETA
"""

import argparse
import os
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import py7zr
from tqdm import tqdm
import csv

# --------------------------
# Set DATA_ROOT on D: drive
# --------------------------
DATA_ROOT = Path("D:/GenBuster200k")
RAW_DIR = DATA_ROOT / "raw"
SPLITS_DIR = DATA_ROOT / "splits"
CACHE_DIR = DATA_ROOT / "cache"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# Worker function
# --------------------------
def download_and_extract_7z_parallel(args):
    """Worker function for parallel .7z extraction"""
    repo_id, filename, output_dir, token = args
    
    try:
        print(f"[Worker] Downloading {filename}...")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            cache_dir=str(CACHE_DIR)
        )
        
        print(f"[Worker] Extracting {filename}...")
        os.makedirs(output_dir, exist_ok=True)
        
        with py7zr.SevenZipFile(downloaded_path, mode='r') as archive:
            archive.extractall(path=output_dir)
        
        # Cleanup downloaded .7z
        os.remove(downloaded_path)
        return True, filename
        
    except Exception as e:
        return False, f"{filename}: {str(e)}"


# --------------------------
# Download all .7z files
# --------------------------
def download_7z_files_parallel(repo_id, out_dir, token=True, max_workers=4):
    """Download and extract all .7z files in parallel"""
    
    print(f"\n{'='*60}")
    print(f"Listing files in {repo_id}...")
    print(f"{'='*60}")
    
    try:
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
        seven_z_files = sorted([f for f in files if f.startswith("GenBuster-200K") and ".7z." in f])
        
        if not seven_z_files:
            print("❌ No .7z files found in repository")
            return False
        
        print(f"\n✓ Found {len(seven_z_files)} archive files:")
        for f in seven_z_files:
            print(f"  • {f}")
        
        print(f"\n⚠️  This will download ~200GB of data")
        print(f"   Estimated time: 8-12 hours (depends on connection)")
        
        response = input("\nProceed with download? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled")
            return False
        
        # Prepare worker arguments
        worker_args = [
            (repo_id, f, out_dir, token) 
            for f in seven_z_files
        ]
        
        # Parallel download with progress bar
        print(f"\n{'='*60}")
        print(f"Downloading with {max_workers} parallel workers...")
        print(f"{'='*60}\n")
        
        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(
                pool.imap(download_and_extract_7z_parallel, worker_args),
                total=len(worker_args),
                desc="Overall Progress"
            ))
        
        # Report results
        successful = sum(1 for ok, _ in results if ok)
        failed = [(name, err) for ok, name_err in results if not ok for name, err in [name_err]]
        
        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"{'='*60}")
        print(f"✓ Success: {successful}/{len(seven_z_files)}")
        
        if failed:
            print(f"❌ Failed: {len(failed)}")
            for name, err in failed:
                print(f"   • {name}: {err}")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# --------------------------
# Create metadata CSV
# --------------------------
def create_metadata_from_structure(dataset_root, output_csv):
    """
    Create metadata CSV from GenBuster directory structure.
    Expected structure:
      dataset_root/
        train/
          real/
          fake/
        test/
          real/
          fake/
        benchmark/
          real/
          fake/
    """
    print(f"\n{'='*60}")
    print(f"Creating metadata CSV from directory structure")
    print(f"{'='*60}")
    
    dataset_root = Path(dataset_root)
    metadata = []
    
    for split in ['train', 'test', 'benchmark']:
        split_dir = dataset_root / split
        
        if not split_dir.exists():
            print(f"⚠️  Split not found: {split_dir}")
            continue
        
        for label_name in ['real', 'fake']:
            label_dir = split_dir / label_name
            
            if not label_dir.exists():
                continue
            
            label = 0 if label_name == 'real' else 1
            
            # Find all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            videos = []
            for ext in video_extensions:
                videos.extend(label_dir.glob(f'*{ext}'))
            
            print(f"  {split}/{label_name}: {len(videos)} videos")
            
            for video_path in videos:
                video_id = video_path.stem
                relative_path = str(video_path.relative_to(dataset_root))
                
                metadata.append({
                    'video_id': video_id,
                    'filepath': relative_path,
                    'label': label,
                    'split': split,
                    'generator': 'real' if label == 0 else 'unknown'
                })
    
    # Write CSV
    os.makedirs(Path(output_csv).parent, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        if metadata:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
    
    print(f"\n✓ Metadata saved: {output_csv}")
    print(f"  Total videos: {len(metadata)}")
    
    # Split summary
    split_counts = defaultdict(lambda: {'real': 0, 'fake': 0})
    for row in metadata:
        split_counts[row['split']][row['generator']] += 1
    
    print(f"\nDataset split summary:")
    for split, counts in split_counts.items():
        print(f"  {split:10s}: {counts['real']:6d} real, {counts['fake']:6d} fake")
    
    return len(metadata)


# --------------------------
# Main
# --------------------------
def main(max_workers):
    # Output directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Repository
    repo_id = "l8cv/GenBuster-200K"  # FULL dataset (not mini)
    
    print(f"\n{'='*60}")
    print(f"GenBuster-200K Full Dataset Download")
    print(f"{'='*60}")
    print(f"Repository: {repo_id}")
    print(f"Output: {RAW_DIR}")
    print(f"Workers: {max_workers}")
    
    success = download_7z_files_parallel(repo_id, RAW_DIR, token=True, max_workers=max_workers)
    
    if not success:
        print("\n❌ Download failed. Check your Hugging Face token.")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return
    
    # Create metadata CSV
    metadata_csv = SPLITS_DIR / "genbuster_full_metadata.csv"
    total_videos = create_metadata_from_structure(RAW_DIR, metadata_csv)
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset Ready!")
    print(f"{'='*60}")
    print(f"Location: {RAW_DIR}")
    print(f"Metadata: {metadata_csv}")
    print(f"Total videos: {total_videos}")
    print(f"\nNext steps:")
    print(f"  1. Create train/val/test splits:")
    print(f"     python scripts/split_metadata_csv.py")
    print(f"  2. Run preprocessing pipeline:")
    print(f"     python scripts/run_full_pipeline.py")


# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download GenBuster-200K FULL dataset")
    p.add_argument("--max_workers", type=int, default=4,
                   help="Number of parallel download workers")
    args = p.parse_args()
    
    main(max_workers=args.max_workers)
