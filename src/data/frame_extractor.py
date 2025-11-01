# src/data/frame_extractor_optimized.py
"""
Optimized hybrid approach:
- GPU for videos >5sec or >720p
- CPU parallel processing for small videos
- Eliminates FFmpeg overhead for short clips
"""

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import subprocess
import logging

logger = logging.getLogger(__name__)

# Hardware-tuned parameters
GPU_MIN_DURATION = 5.0  # seconds
GPU_MIN_RESOLUTION = 480  # pixels
CPU_WORKERS = 15  # processes for parallel CPU decode
IO_WORKERS = 6  # threads for disk writes


def get_video_info(video_path: str) -> Tuple[float, int, int]:
    """Fast probe for video duration and resolution."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        parts = result.stdout.strip().split(',')
        width, height = int(parts[0]), int(parts[1])
        duration = float(parts[2]) if len(parts) > 2 else 0.0
        return duration, width, height
    except:
        return 0.0, 0, 0


def extract_cpu_opencv(video_path: str, out_dir: Path, max_frames: int, 
                       resize: int, fps_sample: float) -> bool:
    """
    Fast CPU extraction for small videos.
    No temp files, direct memory processing.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if total_frames <= max_frames:
            # Extract all frames
            indices = list(range(total_frames))
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        saved = 0
        for target_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Center crop + resize in one go
            h, w = frame.shape[:2]
            s = min(h, w)
            y, x = (h - s) // 2, (w - s) // 2
            crop = frame[y:y+s, x:x+s]
            
            resized = cv2.resize(crop, (resize, resize), 
                                interpolation=cv2.INTER_AREA)  # INTER_AREA faster for downscaling
            
            # Write directly
            out_path = out_dir / f"{saved:03d}.jpg"
            cv2.imwrite(str(out_path), resized,
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
            
            if saved >= max_frames:
                break
        
        cap.release()
        return saved > 0
        
    except Exception as e:
        logger.error(f"CPU extraction failed: {e}")
        return False


def extract_gpu_ffmpeg(video_path: str, out_dir: Path, max_frames: int,
                       resize: int, gpu_id: int = 0) -> bool:
    """
    GPU extraction for large/long videos.
    Optimized FFmpeg pipeline with direct output.
    """
    try:
        # Get duration for fps calculation
        duration, _, _ = get_video_info(video_path)
        if duration <= 0:
            duration = max_frames  # fallback
        
        target_fps = max_frames / duration
        
        # Single-pass FFmpeg with all operations
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', str(video_path),
            # Combined filter: fps, crop, scale
            '-vf', f'fps={target_fps},crop=ih:ih,scale={resize}:{resize}',
            '-frames:v', str(max_frames),
            '-q:v', '2',  # Quality
            str(out_dir / '%03d.jpg')
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
            text=True
        )
        
        return result.returncode == 0 and len(list(out_dir.glob('*.jpg'))) > 0
        
    except Exception as e:
        logger.error(f"GPU extraction failed: {e}")
        return False


def process_single_video(args: Tuple) -> Tuple[str, bool, str]:
    """Worker function for parallel processing."""
    video_path, video_id, out_root, max_frames, resize, fps_sample, gpu_id = args
    
    out_dir = Path(out_root) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    existing = len(list(out_dir.glob('*.jpg')))
    if existing >= max_frames:
        return video_id, True, 'cached'
    
    # Decide GPU vs CPU
    duration, width, height = get_video_info(video_path)
    use_gpu = (duration > GPU_MIN_DURATION or 
               max(width, height) > GPU_MIN_RESOLUTION)
    
    if use_gpu:
        success = extract_gpu_ffmpeg(video_path, out_dir, max_frames, resize, gpu_id)
        method = 'gpu'
    else:
        success = extract_cpu_opencv(video_path, out_dir, max_frames, resize, fps_sample)
        method = 'cpu'
    
    return video_id, success, method


class FrameExtractor:
    """
    Hybrid extractor with intelligent GPU/CPU routing.
    """
    
    def __init__(self, out_root: str = "data/processed/frames",
                 fps_sample: float = 1.0, max_frames: int = 16,
                 resize: int = 224, cpu_workers: int = CPU_WORKERS,
                 gpu_id: int = 0):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.fps_sample = fps_sample
        self.max_frames = max_frames
        self.resize = resize
        self.cpu_workers = cpu_workers
        self.gpu_id = gpu_id
    
    def extract_to_folder(self, video_path: str, 
                         video_id: Optional[str] = None) -> str:
        """Single video extraction."""
        if video_id is None:
            video_id = Path(video_path).stem
        
        args = (video_path, video_id, str(self.out_root), 
                self.max_frames, self.resize, self.fps_sample, self.gpu_id)
        
        vid_id, success, method = process_single_video(args)
        
        if not success:
            raise RuntimeError(f"Extraction failed: {video_path}")
        
        return str(self.out_root / video_id)
    
    def extract_batch(self, video_paths: list, video_ids: list = None) -> dict:
        """
        Batch extraction with parallel processing.
        Returns dict with stats.
        """
        if video_ids is None:
            video_ids = [Path(p).stem for p in video_paths]
        
        args_list = [
            (vp, vid, str(self.out_root), self.max_frames, 
             self.resize, self.fps_sample, self.gpu_id)
            for vp, vid in zip(video_paths, video_ids)
        ]
        
        results = {'success': 0, 'failed': 0, 'cached': 0, 
                  'gpu': 0, 'cpu': 0, 'errors': []}
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.cpu_workers) as executor:
            futures = {executor.submit(process_single_video, args): args[1] 
                      for args in args_list}
            
            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    vid_id, success, method = future.result(timeout=120)
                    
                    if method == 'cached':
                        results['cached'] += 1
                    elif success:
                        results['success'] += 1
                        results[method] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append((vid_id, 'extraction_failed'))
                        
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append((video_id, str(e)))
        
        return results


# Backward compatibility wrapper
def _extract_single_video(args):
    """Legacy interface for existing code."""
    vid_id, success, method = process_single_video(args)
    return success