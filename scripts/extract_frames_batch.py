# extract_frames_batch_optimized.py
"""
- ProcessPoolExecutor for true parallelism
- Hybrid GPU/CPU routing
"""

import sys
from pathlib import Path

project_root = Path("C:/Personal project/ai_video_detector")
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm
from src.data.frame_extractor import FrameExtractor
import time
import json
from datetime import datetime
import logging
import signal

# Critical: Reduce logging overhead
logging.basicConfig(level=logging.ERROR)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self, *args):
        print("\n\nStopping ...")
        self.kill_now = True


def extract_frames_from_csv(
    metadata_csv: str,
    output_root: str = "D:/GenBuster200k/processed/frames",
    fps_sample: float = 1.0,
    max_frames: int = 16,
    resize: int = 224,
    cpu_workers: int = 15,
    batch_size: int = 250,
    gpu_id: int = 0
):
    
    killer = GracefulKiller()
    base_name = Path(metadata_csv).stem
    split_name = base_name.replace('_metadata', '').replace('_split', '').replace('_filtered', '')
    
    print(f"\n{'='*80}")
    print(f"  Processing: {split_name.upper()}")
    print(f"{'='*80}")
    
    df = pd.read_csv(metadata_csv)
    split_output = Path(output_root) / split_name
    split_output.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint_file = split_output / 'checkpoint.json'
    processed_ids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
            processed_ids = set(checkpoint_data.get('processed', []))
    
    # Filter videos to process
    videos_to_process = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = row['absolute_path']
        
        # Skip if already processed
        if video_id in processed_ids:
            continue
        
        # Quick check for existing output
        video_output_dir = split_output / video_id
        if video_output_dir.exists():
            existing_frames = len(list(video_output_dir.glob('*.jpg')))
            if existing_frames >= max_frames:
                processed_ids.add(video_id)
                continue
        
        if not Path(video_path).exists():
            continue
        
        videos_to_process.append((video_id, video_path))
    
    total = len(videos_to_process)
    if total == 0:
        print(f"  All videos already processed")
        return 0, 0, 0, 0
    
    print(f"  Videos to process: {total:,}")
    print(f"  CPU Workers: {cpu_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*80}\n")
    
    # Initialize extractor
    extractor = FrameExtractor(
        out_root=str(split_output),
        fps_sample=fps_sample,
        max_frames=max_frames,
        resize=resize,
        cpu_workers=cpu_workers,
        gpu_id=gpu_id
    )
    
    # Process in batches
    total_success = 0
    total_failed = 0
    total_cached = 0
    method_counts = {'gpu': 0, 'cpu': 0}
    start_time = time.time()
    
    with tqdm(total=total, desc=f"{split_name:>12}", ncols=100) as pbar:
        for i in range(0, total, batch_size):
            if killer.kill_now:
                break
            
            batch = videos_to_process[i:i+batch_size]
            batch_paths = [vp for _, vp in batch]
            batch_ids = [vid for vid, _ in batch]
            
            # Process entire batch
            results = extractor.extract_batch(batch_paths, batch_ids)
            
            # Update counters
            total_success += results['success']
            total_failed += results['failed']
            total_cached += results['cached']
            method_counts['gpu'] += results['gpu']
            method_counts['cpu'] += results['cpu']
            
            # Update processed set
            for vid in batch_ids:
                if vid not in [e[0] for e in results['errors']]:
                    processed_ids.add(vid)
            
            # Update progress
            pbar.update(len(batch))
            
            # Save checkpoint (batched to reduce I/O)
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed': list(processed_ids),
                    'timestamp': datetime.now().isoformat(),
                    'stats': {
                        'success': total_success,
                        'failed': total_failed,
                        'cached': total_cached,
                        'methods': method_counts
                    }
                }, f)
            
            if killer.kill_now:
                break
    
    elapsed = time.time() - start_time
    
    # Log errors if any
    if total_failed > 0:
        error_log = split_output / "errors.txt"
        with open(error_log, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Run: {datetime.now().isoformat()}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"{'='*60}\n")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"   Success: {total_success:,}")
    print(f"   Cached: {total_cached:,}")
    print(f"   Failed: {total_failed:,}")
    print(f"  GPU: {method_counts['gpu']:,} | CPU: {method_counts['cpu']:,}")
    print(f"  Time: {elapsed/60:.1f}m | Speed: {total_success/elapsed:.2f} videos/sec")
    print(f"{'='*80}\n")
    
    return total_success, total_cached, total_failed, elapsed


def main():
    metadata_dir = "C:/Personal project/ai_video_detector/data/splits"
    output_root = "D:/GenBuster200k/processed/frames"
    
    splits = [
        'train_split',
        'val_split', 
        'test_metadata',
        'benchmark_metadata',
        'dfd_metadata'
    ]
    
    print(f"  Output: {output_root}")
    print("="*80)
    
    overall_start = time.time()
    total_success = 0
    total_cached = 0
    total_failed = 0
    
    for split in splits:
        csv_path = Path(metadata_dir) / f"{split}.csv"
        if not csv_path.exists():
            print(f"  Skipping {split} (not found)")
            continue
        
        success, cached, failed, split_time = extract_frames_from_csv(
            metadata_csv=str(csv_path),
            output_root=output_root,
            fps_sample=1.0,
            max_frames=16,
            resize=224,
            cpu_workers=15, 
            batch_size=250,
            gpu_id=0
        )
        
        total_success += success
        total_cached += cached
        total_failed += failed
    
    overall_time = time.time() - overall_start
    
    print("="*80)
    print(f"{' EXTRACTION COMPLETE ':^80}")
    print("="*80)
    print(f"\n  Total Processed: {total_success:,}")
    print(f"  Cached: {total_cached:,}")
    print(f"  Failed: {total_failed:,}")
    print(f"\n  Total Time: {overall_time/3600:.2f}h ({overall_time/60:.1f}m)")
    
    if total_success > 0:
        avg_speed = total_success / overall_time
        print(f"  Average Speed: {avg_speed:.2f} videos/sec")
    
    print(f"\n  Output: {output_root}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()