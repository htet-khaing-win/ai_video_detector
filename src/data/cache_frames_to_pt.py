# cache_frames_to_pt_gpu_optimized.py

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
import hashlib

from utils.cache_manager import CacheManager, CacheConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress OpenCV warnings
import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Configuration


CONFIG = CacheConfig(
    source_root="D:/GenBuster200k/processed/frames",
    cache_root="C:/Personal project/ai_video_detector/data/processed/cached",
    format_type="THWC",
    num_frames=16,
    resolution=224,
    normalize=False,
    num_workers=4,  
    batch_size=50,  
    verify_integrity=True
)

# GPU Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_BATCH_SIZE = 32  
DISK_WRITER_THREADS = 2  

CHECKPOINT_FILE = Path(CONFIG.cache_root) / "caching_checkpoint.json"


# Graceful Interruption Handler

class GracefulKiller:
    """Handle Ctrl+C gracefully and save checkpoint."""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        print("\n\n  Interrupt received. Saving checkpoint...")
        self.kill_now = True


# Robust Frame Loading with Validation

def load_single_frame(frame_path: Path) -> np.ndarray:
    """
    Load a single frame with robust error handling.
    
    Args:
        frame_path: Path to JPEG file
        
    Returns:
        RGB numpy array [H, W, 3] or None if failed
    """
    try:
        # Read with OpenCV
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"cv2.imread returned None: {frame_path}")
            return None
        
        # Validate image dimensions
        if img.ndim != 3 or img.shape[2] != 3:
            logger.error(f"Invalid image shape {img.shape}: {frame_path}")
            return None
        
        # Check for all-zero image (corrupted)
        if img.sum() == 0:
            logger.error(f"All-zero image detected: {frame_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
        
    except Exception as e:
        logger.error(f"Exception loading {frame_path}: {e}")
        return None


def load_frames_to_tensor_robust(video_folder: Path, 
                                  num_frames: int,
                                  resolution: int) -> tuple:
    """
    Load video frames with comprehensive validation.
    
    Args:
        video_folder: Path to video_id folder containing frames
        num_frames: Expected number of frames
        resolution: Expected resolution (H=W)
        
    Returns:
        (success, tensor_or_error_msg)
        - If success: (True, torch.Tensor [T, H, W, C])
        - If failed: (False, error_message_string)
    """
    try:
        # Get frame files
        frame_files = sorted(video_folder.glob('*.jpg'))
        
        if len(frame_files) == 0:
            return False, f"No frames found in {video_folder}"
        
        if len(frame_files) != num_frames:
            return False, f"Expected {num_frames} frames, found {len(frame_files)}"
        
        # Load and validate each frame
        frames = []
        for i, frame_file in enumerate(frame_files):
            img = load_single_frame(frame_file)
            
            if img is None:
                return False, f"Failed to load frame {i}: {frame_file.name}"
            
            # Validate resolution
            h, w = img.shape[:2]
            if h != resolution or w != resolution:
                return False, f"Frame {i} resolution mismatch: expected {resolution}x{resolution}, got {h}x{w}"
            
            frames.append(img)
        
        # Stack into tensor [T, H, W, C]
        try:
            tensor = torch.from_numpy(np.stack(frames, axis=0))
        except Exception as e:
            return False, f"Failed to stack frames: {e}"
        
        # Validate tensor
        if tensor.shape != (num_frames, resolution, resolution, 3):
            return False, f"Invalid tensor shape: {tensor.shape}"
        
        # Check value range
        if tensor.min() < 0 or tensor.max() > 255:
            return False, f"Invalid value range: [{tensor.min()}, {tensor.max()}]"
        
        # Ensure uint8 dtype
        if tensor.dtype != torch.uint8:
            tensor = tensor.byte()
        
        return True, tensor
        
    except Exception as e:
        return False, f"Exception in load_frames_to_tensor_robust: {e}"


# ============================================================================
# GPU-Accelerated Tensor Processing
# ============================================================================

def process_video_batch_gpu(video_batch: list, config: CacheConfig) -> list:
    """
    Process a batch of videos on GPU for maximum throughput.
    
    Args:
        video_batch: List of (video_folder, cache_path) tuples
        config: Cache configuration
        
    Returns:
        List of processing results
    """
    results = []
    
    # Load all videos in batch (CPU)
    loaded_tensors = []
    metadata_list = []
    
    for video_folder, cache_path in video_batch:
        success, tensor_or_error = load_frames_to_tensor_robust(
            video_folder,
            num_frames=config.num_frames,
            resolution=config.resolution
        )
        
        if success:
            loaded_tensors.append(tensor_or_error)
            metadata_list.append({
                'video_folder': video_folder,
                'cache_path': cache_path,
                'video_id': video_folder.name,
                'split': video_folder.parent.name
            })
        else:
            # Record failure immediately
            results.append({
                'video_id': video_folder.name,
                'split': video_folder.parent.name,
                'success': False,
                'error': tensor_or_error,
                'cache_file': str(cache_path)
            })
    
    # If no valid tensors in batch, return
    if not loaded_tensors:
        return results
    
    
    # Save each tensor with atomic writes
    cache_manager = CacheManager(
        cache_root=config.cache_root,
        format_type=config.format_type
    )
    
    for tensor, meta in zip(loaded_tensors, metadata_list):
        result = {
            'video_id': meta['video_id'],
            'split': meta['split'],
            'cache_file': str(meta['cache_path']),
            'success': False,
            'error': None
        }
        
        try:
            # Prepare metadata
            save_metadata = {
                'video_id': meta['video_id'],
                'split': meta['split'],
                'source_path': str(meta['video_folder']),
                'cached_at': datetime.now().isoformat()
            }
            
            # Atomic save with verification
            success = cache_manager.save(
                tensor=tensor,
                output_path=meta['cache_path'],
                metadata=save_metadata,
                verify=config.verify_integrity
            )
            
            if success:
                # Verify the saved file is readable
                verify_result = cache_manager.load(
                    meta['cache_path'],
                    verify=True,
                    device='cpu'
                )
                
                if verify_result is None:
            
                    if meta['cache_path'].exists():
                        meta['cache_path'].unlink()
                    result['error'] = "Post-save verification failed"
                    logger.error(f"Verification failed after save: {meta['cache_path']}")
                else:
                    result['success'] = True
                    result['shape'] = list(tensor.shape)
                    result['dtype'] = str(tensor.dtype)
            else:
                result['error'] = "Cache manager save failed"
                # Cleanup any partial files
                if meta['cache_path'].exists():
                    try:
                        meta['cache_path'].unlink()
                    except:
                        pass
        
        except Exception as e:
            result['error'] = f"Exception during save: {str(e)}"
            logger.error(f"Failed to save {meta['video_id']}: {e}")
            # Cleanup
            if meta['cache_path'].exists():
                try:
                    meta['cache_path'].unlink()
                except:
                    pass
        
        results.append(result)
    
    return results


# Controlled Disk I/O Writer

class DiskWriterPool:
    """
    Thread pool for controlled disk writes to prevent race conditions.
    Ensures only N concurrent disk operations at a time.
    """
    
    def __init__(self, num_threads: int = 2):
        self.num_threads = num_threads
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.running = False
    
    def _worker(self):
        """Worker thread that processes save tasks."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                # Process the batch
                video_batch, config = task
                results = process_video_batch_gpu(video_batch, config)
                
                # Put results in result queue
                for result in results:
                    self.result_queue.put(result)
                
                self.task_queue.task_done()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Worker exception: {e}")
    
    def start(self):
        """Start worker threads."""
        self.running = True
        for _ in range(self.num_threads):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"DiskWriterPool started with {self.num_threads} threads")
    
    def submit(self, video_batch: list, config: CacheConfig):
        """Submit a batch for processing."""
        self.task_queue.put((video_batch, config))
    
    def get_results(self, timeout: float = 0.1) -> list:
        """Get all available results without blocking."""
        results = []
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except:
                break
        return results
    
    def shutdown(self):
        """Shutdown worker threads gracefully."""
        self.running = False
        
        # Send stop signals
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        logger.info("DiskWriterPool shut down")


# Checkpoint Management

def load_checkpoint() -> set:
    """Load checkpoint of already processed videos."""
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
    """Save checkpoint with processed videos and stats."""
    try:
        checkpoint_data = {
            'processed_videos': list(processed_videos),
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }
        
        # Atomic write
        tmp_file = CHECKPOINT_FILE.with_suffix('.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        tmp_file.replace(CHECKPOINT_FILE)
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

# Main Caching Pipeline

def collect_video_folders() -> list:
    """Collect all video folders from source directory."""
    source_root = Path(CONFIG.source_root)
    cache_root = Path(CONFIG.cache_root)
    
    video_tasks = []
    splits = ['train', 'val', 'test', 'benchmark', 'dfd']
    
    for split in splits:
        split_path = source_root / split
        
        if not split_path.exists():
            logger.warning(f"Split not found: {split}")
            continue
        
        # Create corresponding cache directory
        cache_split_path = cache_root / split
        cache_split_path.mkdir(parents=True, exist_ok=True)
        
        # Collect video folders
        for video_folder in split_path.iterdir():
            if video_folder.is_dir():
                cache_file = cache_split_path / f"{video_folder.name}.pt"
                video_tasks.append((video_folder, cache_file))
    
    return video_tasks


def main():
    """Main caching workflow with GPU optimization."""
    
    killer = GracefulKiller()
    
    print("\n" + "="*80)
    print(f"{'GPU-OPTIMIZED TENSOR CACHING PIPELINE':^80}")
    print("="*80)
    print(f"  Source: {CONFIG.source_root}")
    print(f"  Output: {CONFIG.cache_root}")
    print(f"  Format: {CONFIG.format_type} ({CONFIG.num_frames} frames @ {CONFIG.resolution}x{CONFIG.resolution})")
    print(f"  GPU: {'CUDA Available' if torch.cuda.is_available() else 'CPU Only'}")
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
    logger.info(f"Loaded checkpoint: {len(processed_videos)} already processed")
    
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
        print("\n All videos already cached!")
        return 0
    
    logger.info(f"Videos remaining: {len(tasks_to_process)}")
    
    # Statistics
    stats = defaultdict(int)
    stats['total'] = len(video_tasks)
    stats['skipped'] = len(processed_videos)
    
    # Start disk writer pool
    writer_pool = DiskWriterPool(num_threads=DISK_WRITER_THREADS)
    writer_pool.start()
    
    # Process in batches
    start_time = time.time()
    batch_count = 0
    
    try:
        from tqdm import tqdm
        
        # Submit batches to writer pool
        for i in range(0, len(tasks_to_process), GPU_BATCH_SIZE):
            if killer.kill_now:
                break
            
            batch = tasks_to_process[i:i+GPU_BATCH_SIZE]
            writer_pool.submit(batch, CONFIG)
            batch_count += 1
        
        # Progress bar for collecting results
        with tqdm(total=len(tasks_to_process), desc="Caching progress") as pbar:
            collected_results = 0
            checkpoint_counter = 0
            
            while collected_results < len(tasks_to_process):
                if killer.kill_now:
                    break
                
                # Get results from writer pool
                results = writer_pool.get_results(timeout=0.5)
                
                for result in results:
                    collected_results += 1
                    checkpoint_counter += 1
                    
                    if result['success']:
                        stats['success'] += 1
                        video_key = f"{result['split']}/{result['video_id']}"
                        processed_videos.add(video_key)
                    else:
                        stats['failed'] += 1
                        logger.error(f"Failed {result['video_id']}: {result.get('error', 'Unknown')}")
                    
                    pbar.update(1)
                    
                    # Periodic checkpoint
                    if checkpoint_counter >= CONFIG.batch_size:
                        save_checkpoint(processed_videos, dict(stats))
                        checkpoint_counter = 0
                
                time.sleep(0.1)  # Prevent busy waiting
    
    finally:
        # Shutdown writer pool
        writer_pool.shutdown()
        
        # Final checkpoint
        save_checkpoint(processed_videos, dict(stats))
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"{'CACHING SUMMARY':^80}")
    print("="*80)
    print(f"  Total videos: {stats['total']:,}")
    print(f"   Successfully cached: {stats['success']:,}")
    print(f"   Skipped (already done): {stats['skipped']:,}")
    print(f"  Failed: {stats['failed']:,}")
    print(f"\n    Time elapsed: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    
    if stats['success'] > 0:
        speed = stats['success'] / elapsed
        print(f"   Average speed: {speed:.2f} videos/second")
    
    print("="*80)
    
    if stats['failed'] > 0:
        print(f"\n  {stats['failed']} videos failed to cache.")
        print("   Check logs for details. These videos will be skipped during training.")
        print("   To retry failed videos, delete checkpoint and re-run.")
    else:
        print("\n All videos cached successfully!")
    
    print(f"\n Cached data location: {CONFIG.cache_root}")
    print("="*80 + "\n")
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())