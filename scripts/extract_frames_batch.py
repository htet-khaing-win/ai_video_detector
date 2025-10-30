# extract_frames_batch.py
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
from collections import defaultdict
import logging
import signal

# Completely suppress all logging output
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

# Suppress all loggers including frame_extractor
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

# Specifically target the frame extractor logger
logging.getLogger('src.data.frame_extractor').setLevel(logging.CRITICAL)
logging.getLogger('src.data.extract_frames_batch').setLevel(logging.CRITICAL)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self, *args):
        print("\n\n⚠️  Stopping gracefully... Saving checkpoint...")
        self.kill_now = True

class ProgressTracker:
    def __init__(self):
        self.successful = 0
        self.skipped = 0
        self.failed = []
        self.method_counts = defaultdict(int)
        self.start_time = time.time()
        self.last_update = None
        self.video_times = []

    def update_success(self, method='GPU'):
        now = time.time()
        if self.last_update is not None:
            dt = now - self.last_update
            if dt > 0:
                self.video_times.append(dt)
                if len(self.video_times) > 200:
                    self.video_times.pop(0)
        self.last_update = now
        self.successful += 1
        self.method_counts[method] += 1

    def update_skip(self):
        self.skipped += 1

    def update_fail(self, video_id, error):
        self.failed.append((video_id, str(error)))

    def get_avg_speed(self):
        if len(self.video_times) >= 3:
            avg_dt = sum(self.video_times) / len(self.video_times)
            return 1.0 / avg_dt if avg_dt > 0 else 0.0
        return 0.0

    def get_summary(self, total):
        elapsed = time.time() - self.start_time
        avg_speed = self.get_avg_speed()
        remaining = total - (self.successful + self.skipped)
        eta_seconds = remaining / max(avg_speed, 1e-6) if avg_speed > 0 else float('inf')
        return {
            'total': total,
            'successful': self.successful,
            'skipped': self.skipped,
            'failed': len(self.failed),
            'elapsed': elapsed,
            'avg_speed': avg_speed,
            'eta_seconds': eta_seconds,
            'methods': dict(self.method_counts)
        }

def extract_frames_from_csv(
    metadata_csv,
    output_root="D:/GenBuster200k/processed/frames",
    fps_sample=1,
    max_frames=8,
    resize=224,
    num_workers=68,
    gpu_id=0
):
    killer = GracefulKiller()
    split_name = Path(metadata_csv).stem.replace('_metadata', '').replace('_split', '')

    print("\n" + "═"*80)
    print(f"  Extracting: {split_name.upper()}")
    print("═"*80)

    df = pd.read_csv(metadata_csv)
    split_output = Path(output_root) / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    extractor = FrameExtractor(
        out_root=str(split_output),
        fps_sample=fps_sample,
        max_frames=max_frames,
        resize=resize,
        num_workers=num_workers,
        gpu_id=gpu_id
    )

    tracker = ProgressTracker()
    checkpoint_file = split_output / 'checkpoint.json'
    processed_ids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed_ids = set(json.load(f).get('processed', []))

    videos_to_process = []
    for idx, row in df.iterrows():
        video_id = row['video_id']
        video_path = row['absolute_path']
        if video_id in processed_ids:
            tracker.update_skip()
            continue
        video_output_dir = split_output / video_id
        if video_output_dir.exists() and len(list(video_output_dir.glob('*.jpg'))) >= 4:
            tracker.update_skip()
            processed_ids.add(video_id)
            continue
        if not Path(video_path).exists():
            tracker.update_fail(video_id, "Not found")
            continue
        videos_to_process.append((video_id, video_path))

    total = len(df)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        batch_size = 500
        
        # Single clean progress bar with dynamic ETA
        with tqdm(
            total=len(videos_to_process),
            desc=f"{split_name:>12}",
            bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100,
            leave=True
        ) as pbar:
            for i in range(0, len(videos_to_process), batch_size):
                if killer.kill_now:
                    break
                batch = videos_to_process[i:i+batch_size]
                for video_id, video_path in batch:
                    if killer.kill_now:
                        break
                    future = executor.submit(extractor.extract_to_folder, video_path, video_id)
                    futures[future] = video_id

                for future in as_completed(list(futures.keys())):
                    if killer.kill_now:
                        break
                    video_id = futures.pop(future)
                    try:
                        future.result(timeout=300)
                        tracker.update_success('GPU')
                        processed_ids.add(video_id)
                    except Exception as e:
                        tracker.update_fail(video_id, str(e))
                    pbar.update(1)

                # Silent checkpoint save
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed': list(processed_ids), 'timestamp': datetime.now().isoformat()}, f)

    for future in futures.keys():
        if future.running():
            future.cancel()
    
    summary = tracker.get_summary(total)
    
    # Clean summary output
    print(f"\n  ✓ Processed: {summary['successful']:,} | ⊘ Skipped: {summary['skipped']:,} | ✗ Failed: {summary['failed']:,} | Time: {summary['elapsed']/60:.1f}m\n")

    if tracker.failed:
        error_log = split_output / "errors.txt"
        with open(error_log, 'w') as f:
            for vid, err in tracker.failed:
                f.write(f"{vid}: {err}\n")

    return summary['successful'], summary['skipped'], summary['failed'], summary['elapsed']

def main():
    metadata_dir = "C:/Personal project/ai_video_detector/data/splits"
    output_root = "D:/GenBuster200k/processed/frames"
    splits = ['train_split', 'val_split', 'test_metadata', 'benchmark_metadata', 'dfd_metadata']
    total_success = 0
    total_skipped = 0
    total_failed = 0
    overall_start = time.time()

    print("\n" + "="*80)
    print(f"{'GPU-ACCELERATED FRAME EXTRACTION':^80}")
    print("="*80)
    print(f"  Output: {output_root}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    for split in splits:
        csv_path = Path(metadata_dir) / f"{split}.csv"
        if not csv_path.exists():
            continue
        success, skipped, failed, split_time = extract_frames_from_csv(
            metadata_csv=str(csv_path),
            output_root=output_root,
            fps_sample=1,
            max_frames=8,
            resize=224,
            num_workers=68,
            gpu_id=0
        )
        total_success += success
        total_skipped += skipped
        total_failed += failed

    overall_time = time.time() - overall_start
    
    print("="*80)
    print(f"{'🎉 EXTRACTION COMPLETE 🎉':^80}")
    print("="*80)
    print(f"\n  Total Videos Processed: {total_success:,}")
    print(f"  Skipped (Already Done): {total_skipped:,}")
    print(f"  Failed: {total_failed:,}")
    print(f"\n  📊 Total Time: {overall_time/3600:.2f} hours ({overall_time/60:.1f} minutes)")
    print(f"  ⚡ Average Throughput: {total_success/max(overall_time,1):.2f} videos/second")
    print(f"\n  Output Directory: {output_root}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()