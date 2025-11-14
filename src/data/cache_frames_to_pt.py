"""
CTHW-Native Video Tensor Caching Pipeline

Converts JPEG frame sequences to PyTorch tensors in [C, T, H, W] format:
  [3, 16, 224, 224] = [RGB, 16_frames, 224px_height, 224px_width]

Input:  D:/GenBuster200k/processed/frames/<split>/<video_id>/*.jpg
Output: D:/GenBuster200k/processed/cached/<split>/<video_id>.pt

Features:
- Direct CTHW conversion (no runtime permutation needed)
- GPU-accelerated processing
- Robust validation at every step
- Atomic writes with integrity checks
- Automatic cleanup of failed writes
"""

import sys
from pathlib import Path
project_root = Path("C:/Personal project/ai_video_detector")
sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import time
import logging
import signal
from typing import Optional, Tuple, Any


from utils.cache_manager import CacheManager, CacheConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress OpenCV warnings
import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'


# Configuration

CONFIG = CacheConfig(
    source_root="D:/GenBuster200k/processed/frames",
    cache_root="D:/GenBuster200k/processed/cached",  # NEW LOCATION
    num_frames=16,
    resolution=224,
    normalize=False,  # Keep uint8 for now, normalize in DataLoader
    num_workers=4,
    batch_size=50,
    verify_integrity=True
)

CHECKPOINT_FILE = Path(CONFIG.cache_root) / "caching_checkpoint.json"
GPU_BATCH_SIZE = 32
DISK_WRITER_THREADS = 2

# Graceful Interruption


class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self, *args):
        print("\n\n  Interrupt received. Saving checkpoint...")
        self.kill_now = True


# Frame Loading with CTHW Conversion


def load_single_frame(frame_path: Path, resolution: int) -> Optional[np.ndarray]:
    """
    Load and validate a single frame.
    
    Returns:
        RGB numpy array [H, W, 3] or None if failed
    """
    try:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"cv2.imread returned None: {frame_path}")
            return None
        
        if img.ndim != 3 or img.shape[2] != 3:
            logger.error(f"Invalid shape {img.shape}: {frame_path}")
            return None
        
        h, w = img.shape[:2]
        if h != resolution or w != resolution:
            logger.error(f"Resolution mismatch {h}x{w}, expected {resolution}x{resolution}: {frame_path}")
            return None
        
        if img.sum() == 0:
            logger.error(f"All-zero image: {frame_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
        
    except Exception as e:
        logger.error(f"Exception loading {frame_path}: {e}")
        return None


def load_frames_to_cthw_tensor(video_folder: Path,
                                num_frames: int,
                                resolution: int,
                                normalize: bool = False) -> Tuple[bool, any]:
    """
    Load frames and convert directly to CTHW format.
    
    This is the CRITICAL function that defines the tensor format.
    
    Args:
        video_folder: Path to video_id folder with frames
        num_frames: Expected number of frames
        resolution: Expected resolution
        normalize: Convert to float32 [0,1] if True
        
    Returns:
        (success, tensor_or_error)
        - Success: (True, torch.Tensor [C, T, H, W])
        - Failure: (False, error_message)
    """
    try:
        # Get frame files
        frame_files = sorted(video_folder.glob('*.jpg'))
        
        if len(frame_files) == 0:
            return False, f"No frames found in {video_folder}"
        
        if len(frame_files) != num_frames:
            return False, f"Expected {num_frames} frames, found {len(frame_files)}"
        
        # Load all frames
        frames = []
        for i, frame_file in enumerate(frame_files):
            img = load_single_frame(frame_file, resolution)
            
            if img is None:
                return False, f"Failed to load frame {i}: {frame_file.name}"
            
            frames.append(img)
        
        # Stack into numpy array: [T, H, W, C]
        frames_array = np.stack(frames, axis=0)  # Shape: [16, 224, 224, 3]
        
        # Convert to tensor
        tensor_thwc = torch.from_numpy(frames_array)  # [T, H, W, C]
        
        # CRITICAL: Permute to CTHW format
        # [T, H, W, C] → [C, T, H, W]
        # [16, 224, 224, 3] → [3, 16, 224, 224]
        tensor_cthw = tensor_thwc.permute(3, 0, 1, 2)
        
        # Verify shape
        expected_shape = (3, num_frames, resolution, resolution)
        if tensor_cthw.shape != expected_shape:
            return False, f"Shape error after permute: {tensor_cthw.shape} != {expected_shape}"
        
        # Normalize if requested
        if normalize:
            if tensor_cthw.dtype == torch.uint8:
                tensor_cthw = tensor_cthw.float() / 255.0
        
        # Final validation
        if tensor_cthw.dtype not in (torch.uint8, torch.float32):
            return False, f"Invalid dtype: {tensor_cthw.dtype}"
        
        if tensor_cthw.min() < 0:
            return False, f"Negative values detected: {tensor_cthw.min()}"
        
        if tensor_cthw.dtype == torch.uint8 and tensor_cthw.max() > 255:
            return False, f"uint8 overflow: {tensor_cthw.max()}"
        
        if tensor_cthw.dtype == torch.float32 and tensor_cthw.max() > 1:
            return False, f"float32 overflow: {tensor_cthw.max()}"
        
        return True, tensor_cthw
        
    except Exception as e:
        return False, f"Exception in load_frames_to_cthw_tensor: {e}"


# Batch Processing

def process_video_batch(video_batch: list, config: CacheConfig) -> list:
    """
    Process a batch of videos and save as CTHW tensors.
    
    Args:
        video_batch: List of (video_folder, cache_path) tuples
        config: Cache configuration
        
    Returns:
        List of processing results
    """
    results = []
    cache_manager = CacheManager(
        cache_root=config.cache_root,
        num_frames=config.num_frames,
        resolution=config.resolution
    )
    
    for video_folder, cache_path in video_batch:
        result = {
            'video_id': video_folder.name,
            'split': video_folder.parent.name,
            'cache_file': str(cache_path),
            'success': False,
            'error': None
        }
        
        try:
            # Load and convert to CTHW
            success, tensor_or_error = load_frames_to_cthw_tensor(
                video_folder,
                num_frames=config.num_frames,
                resolution=config.resolution,
                normalize=config.normalize
            )
            
            if not success:
                result['error'] = tensor_or_error
                logger.error(f"Load failed {video_folder.name}: {tensor_or_error}")
                results.append(result)
                continue
            
            tensor_cthw = tensor_or_error
            
            # Prepare metadata
            save_metadata = {
                'video_id': video_folder.name,
                'split': video_folder.parent.name,
                'source_path': str(video_folder),
                'cached_at': datetime.now().isoformat(),
                'format': 'CTHW',
                'shape_description': 'Channels (RGB), Time (frames), Height, Width'
            }
            
            # Save with atomic write
            save_success = cache_manager.save(
                tensor=tensor_cthw,
                output_path=cache_path,
                metadata=save_metadata,
                verify=config.verify_integrity
            )
            
            if not save_success:
                result['error'] = "Cache save failed"
                # Cleanup partial file
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except:
                        pass
                logger.error(f"Save failed {video_folder.name}")
                results.append(result)
                continue
            
            # Verify saved file is loadable
            verify_load = cache_manager.load(cache_path, verify=True, device='cpu')
            
            if verify_load is None:
                result['error'] = "Post-save verification failed"
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except:
                        pass
                logger.error(f"Verification failed {video_folder.name}")
                results.append(result)
                continue
            
            # Success!
            result['success'] = True
            result['shape'] = list(tensor_cthw.shape)
            result['dtype'] = str(tensor_cthw.dtype)
            
        except Exception as e:
            result['error'] = f"Exception: {str(e)}"
            logger.error(f"Exception processing {video_folder.name}: {e}")
            # Cleanup
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except:
                    pass
        
        results.append(result)
    
    return results

# Disk Writer Pool

class DiskWriterPool:
    """Thread pool for controlled disk writes."""
    
    def __init__(self, num_threads: int = 2):
        self.num_threads = num_threads
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.running = False
    
    def _worker(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                video_batch, config = task
                results = process_video_batch(video_batch, config)
                
                for result in results:
                    self.result_queue.put(result)
                
                self.task_queue.task_done()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Worker exception: {e}")
    
    def start(self):
        self.running = True
        for _ in range(self.num_threads):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"DiskWriterPool started ({self.num_threads} threads)")
    
    def submit(self, video_batch: list, config: CacheConfig):
        self.task_queue.put((video_batch, config))
    
    def get_results(self, timeout: float = 0.1) -> list:
        results = []
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except:
                break
        return results
    
    def shutdown(self):
        self.running = False
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("DiskWriterPool shut down")


# Checkpoint Management

def load_checkpoint() -> set:
    if not CHECKPOINT_FILE.exists():
        return set()
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_videos', []))
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return set()


def save_checkpoint(processed_videos: set, stats: dict):
    try:
        checkpoint_data = {
            'processed_videos': list(processed_videos),
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'format': 'CTHW',
            'version': '3.0.0'
        }
        tmp_file = CHECKPOINT_FILE.with_suffix('.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        tmp_file.replace(CHECKPOINT_FILE)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# Main Pipeline

def collect_video_folders() -> list:
    source_root = Path(CONFIG.source_root)
    cache_root = Path(CONFIG.cache_root)
    
    video_tasks = []
    splits = ['train', 'val', 'test', 'benchmark', 'dfd']
    
    for split in splits:
        split_path = source_root / split
        
        if not split_path.exists():
            logger.warning(f"Split not found: {split}")
            continue
        
        cache_split_path = cache_root / split
        cache_split_path.mkdir(parents=True, exist_ok=True)
        
        for video_folder in split_path.iterdir():
            if video_folder.is_dir():
                cache_file = cache_split_path / f"{video_folder.name}.pt"
                video_tasks.append((video_folder, cache_file))
    
    return video_tasks


def main():
    killer = GracefulKiller()
    
    print("\n" + "="*80)
    print(f"{'CTHW-NATIVE TENSOR CACHING PIPELINE':^80}")
    print("="*80)
    print(f"  Source: {CONFIG.source_root}")
    print(f"  Output: {CONFIG.cache_root}")
    print(f"  Format: CTHW [3, {CONFIG.num_frames}, {CONFIG.resolution}, {CONFIG.resolution}]")
    print(f"  Normalize: {CONFIG.normalize}")
    print(f"  Batch size: {GPU_BATCH_SIZE} videos/batch")
    print(f"  Disk writers: {DISK_WRITER_THREADS}")
    print(f"  Verification: {'Enabled' if CONFIG.verify_integrity else 'Disabled'}")
    print("="*80 + "\n")
    
    # Save configuration
    config_path = Path(CONFIG.cache_root) / "cache_config.json"
    CONFIG.save(config_path)
    logger.info(f"Configuration saved: {config_path}")
    
    # Load checkpoint
    processed_videos = load_checkpoint()
    logger.info(f"Checkpoint: {len(processed_videos)} already processed")
    
    # Collect tasks
    logger.info("Collecting video folders...")
    video_tasks = collect_video_folders()
    logger.info(f"Found {len(video_tasks)} videos total")
    
    # Filter already processed
    tasks_to_process = [
        (vf, cp) for vf, cp in video_tasks
        if f"{vf.parent.name}/{vf.name}" not in processed_videos
    ]
    
    if not tasks_to_process:
        print("\n✅ All videos already cached in CTHW format!")
        return 0
    
    logger.info(f"Videos to process: {len(tasks_to_process)}")
    
    # Statistics
    stats = defaultdict(int)
    stats['total'] = len(video_tasks)
    stats['skipped'] = len(processed_videos)
    
    # Start disk writer pool
    writer_pool = DiskWriterPool(num_threads=DISK_WRITER_THREADS)
    writer_pool.start()
    
    start_time = time.time()
    
    try:
        from tqdm import tqdm
        
        # Submit batches
        for i in range(0, len(tasks_to_process), GPU_BATCH_SIZE):
            if killer.kill_now:
                break
            batch = tasks_to_process[i:i+GPU_BATCH_SIZE]
            writer_pool.submit(batch, CONFIG)
        
        # Collect results
        with tqdm(total=len(tasks_to_process), desc="Caching (CTHW)") as pbar:
            collected = 0
            checkpoint_counter = 0
            
            while collected < len(tasks_to_process):
                if killer.kill_now:
                    break
                
                results = writer_pool.get_results(timeout=0.5)
                
                for result in results:
                    collected += 1
                    checkpoint_counter += 1
                    
                    if result['success']:
                        stats['success'] += 1
                        video_key = f"{result['split']}/{result['video_id']}"
                        processed_videos.add(video_key)
                    else:
                        stats['failed'] += 1
                    
                    pbar.update(1)
                    
                    if checkpoint_counter >= CONFIG.batch_size:
                        save_checkpoint(processed_videos, dict(stats))
                        checkpoint_counter = 0
                
                time.sleep(0.1)
    
    finally:
        writer_pool.shutdown()
        save_checkpoint(processed_videos, dict(stats))
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print(f"{'CACHING SUMMARY':^80}")
    print("="*80)
    print(f"  Total videos: {stats['total']:,}")
    print(f"   Successfully cached: {stats['success']:,}")
    print(f"   Skipped: {stats['skipped']:,}")
    print(f"   Failed: {stats['failed']:,}")
    print(f"\n    Time: {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")
    
    if stats['success'] > 0:
        speed = stats['success'] / elapsed
        print(f"  ⚡ Speed: {speed:.2f} videos/sec")
    
    print("\n   Tensor Format: CTHW [3, 16, 224, 224]")
    print(f"   Location: {CONFIG.cache_root}")
    print("="*80)
    
    if stats['failed'] > 0:
        print(f"\n  {stats['failed']} videos failed.")
        print("   Check logs. Re-run to retry failed videos.")
    else:
        print("\n All videos cached successfully in CTHW format!")
    
    print("="*80 + "\n")
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())