# src/data/frame_extractor.py
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

# NVML optional: install with `pip install nvidia-ml-py` for accurate VRAM info
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

os.add_dll_directory(r"C:\Projects\ffmpeg\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")

try:
    import PyNvCodec as nvc
    PYNVCODEC_AVAILABLE = True
except Exception:
    PYNVCODEC_AVAILABLE = False

logger = logging.getLogger("src.data.frame_extractor")
logger.setLevel(logging.CRITICAL)  # Silence all logs

# ============================================================================
# OPTIMIZED PARAMETERS FOR HIGHER GPU UTILIZATION
# ============================================================================
TARGET_VRAM_PCT = 0.88      # Increased from 0.80 → use more VRAM
MAX_VRAM_PCT = 0.95         # Increased from 0.92 → allow higher usage
DEFAULT_INITIAL_BATCH = 192 # Doubled from 96 → decode more frames per batch
DEFAULT_MAX_BATCH = 384     # Doubled from 192 → higher ceiling
SAVE_THREAD_WORKERS = 16    # Increased from 6 → prevent CPU I/O bottleneck
GPU_ENCODE_QUALITY = 90
LOG_EVERY_N_VIDEOS = 99999  # Effectively disable intermediate logging
# ============================================================================

def _get_gpu_mem_info(gpu_id=0):
    """Return (used_bytes, total_bytes) or (None, None) if NVML unavailable."""
    if not NVML_AVAILABLE:
        return (None, None)
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return (int(info.used), int(info.total))
    except Exception:
        return (None, None)

def _extract_opencv(video_path, out_dir, fps_sample, max_frames, resize):
    """Fallback OpenCV extraction (keeps original behavior)."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / fps_sample)))
        saved = 0
        idx = 0
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                h, w = frame.shape[:2]
                s = min(h, w)
                y, x = (h - s) // 2, (w - s) // 2
                crop = frame[y:y+s, x:x+s]
                resized = cv2.resize(crop, (resize, resize), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(out_dir / f"{saved:03d}.jpg"), resized,
                           [cv2.IMWRITE_JPEG_QUALITY, 90, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                saved += 1
            idx += 1
        cap.release()
        return saved > 0
    except Exception:
        return False

def _try_build_gpu_encoder(width, height, gpu_id):
    """Return a PyNvEncoder if available for this system; else None."""
    if not PYNVCODEC_AVAILABLE:
        return None
    tries = [
        {"codec": "mjpeg", "format": "rgb", "quality": str(GPU_ENCODE_QUALITY)},
        f"mjpeg:format=rgb:quality={GPU_ENCODE_QUALITY}",
    ]
    for p in tries:
        try:
            enc = nvc.PyNvEncoder(p, gpu_id)
            return enc
        except Exception:
            continue
    return None

def _async_write_numpy_image(path: str, arr: np.ndarray):
    """Thread worker: encode RGB->JPEG and write to disk."""
    try:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, GPU_ENCODE_QUALITY])
        if ok:
            Path(path).write_bytes(buf.tobytes())
    except Exception:
        pass

def _async_write_bytes(path: str, data: bytes):
    try:
        Path(path).write_bytes(data)
    except Exception:
        pass

def _compute_per_worker_batch(width, height, gpu_id, num_workers,
                              target_pct=TARGET_VRAM_PCT, cap=DEFAULT_INITIAL_BATCH):
    """
    Compute a safe per-worker batch size so that num_workers * batch_size roughly fills target VRAM.
    Uses per-surface RGB bytes = 3 * width * height (uint8).
    Falls back to cap if NVML unavailable.
    """
    used, total = _get_gpu_mem_info(gpu_id)
    per_surface_bytes = 3 * width * height
    # if NVML available, compute from total
    if total is not None and total > 0:
        target_bytes = int(total * target_pct)
        # reserve some headroom for non-surface allocations
        safe_target = max(1, target_bytes - int(total * 0.03))  # Reduced from 0.05 to 0.03
        per_worker_target = max(1, safe_target // max(1, num_workers))
        bs = max(8, int(per_worker_target / max(1, per_surface_bytes)))  # Min 8 instead of 4
        # clamp
        bs = min(bs, DEFAULT_MAX_BATCH)
        # but don't return huge zero-length batch
        if bs < 8:
            bs = min(cap, DEFAULT_INITIAL_BATCH)
        return bs
    # NVML not available: return conservative cap
    return min(cap, DEFAULT_INITIAL_BATCH)

def _extract_single_video(args):
    """
    Optimized per-video extractor.
    args expected:
      (video_path, video_id, out_root, fps_sample, max_frames, resize, gpu_id, num_workers)
    num_workers is the total concurrent producers running (used to compute per-worker batch)
    """
    # unpack defensively (support older call signatures without num_workers)
    if len(args) == 7:
        video_path, video_id, out_root, fps_sample, max_frames, resize, gpu_id = args
        num_workers = 1
    else:
        video_path, video_id, out_root, fps_sample, max_frames, resize, gpu_id, num_workers = args

    out_dir = Path(out_root) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not PYNVCODEC_AVAILABLE:
        return _extract_opencv(video_path, out_dir, fps_sample, max_frames, resize)

    try:
        # decoder + probe
        nvDec = nvc.PyNvDecoder(video_path, gpu_id)
        width, height = nvDec.Width(), nvDec.Height()

        try:
            nvDmx = nvc.PyFFmpegDemuxer(video_path)
            fps = float(nvDmx.Framerate()) or 30.0
        except Exception:
            fps = 30.0
        step = max(1, int(round(fps / fps_sample)))

        test_s = nvDec.DecodeSingleSurface()
        if test_s.Empty():
            return False
        src_fmt = test_s.Format()

        try:
            cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.MPEG)
        except Exception:
            cc_ctx = None

        two_stage = False
        if src_fmt in (nvc.PixelFormat.P10, nvc.PixelFormat.P12):
            to_nv12 = nvc.PySurfaceConverter(width, height, src_fmt, nvc.PixelFormat.NV12, gpu_id)
            to_rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpu_id)
            two_stage = True
        elif src_fmt == nvc.PixelFormat.NV12:
            to_rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpu_id)
        else:
            return _extract_opencv(video_path, out_dir, fps_sample, max_frames, resize)

        # recreate decoder cleanly
        nvDec = nvc.PyNvDecoder(video_path, gpu_id)
        downloader = nvc.PySurfaceDownloader(width, height, nvc.PixelFormat.RGB, gpu_id)

        # try GPU encoder (in-process)
        gpu_encoder = _try_build_gpu_encoder(width, height, gpu_id)
        use_gpu_encoder = gpu_encoder is not None

        # writer pool (threads) for CPU encode & disk I/O - INCREASED TO 16
        save_executor = ThreadPoolExecutor(max_workers=SAVE_THREAD_WORKERS)
        save_futures = []

        # compute per-worker batch size based on GPU VRAM and total concurrent producers
        computed_bs = _compute_per_worker_batch(width, height, gpu_id, num_workers,
                                               target_pct=TARGET_VRAM_PCT, cap=DEFAULT_INITIAL_BATCH)
        batch_size = computed_bs

        saved = 0
        idx = 0
        processed_count = 0
        last_log_ts = time.time()

        # allocate a host buffer (channels-first) for downloads
        host_full = np.empty((3, height, width), dtype=np.uint8)

        while saved < max_frames:
            # dynamic adjust based on instantaneous VRAM (more aggressive)
            used, total = _get_gpu_mem_info(gpu_id)
            if total is not None:
                vram_pct = used / total if total else 0.0
                if vram_pct > MAX_VRAM_PCT:
                    batch_size = max(8, int(batch_size * 0.75))  # More gradual reduction
                elif vram_pct < TARGET_VRAM_PCT and batch_size < DEFAULT_MAX_BATCH:
                    # More aggressive growth
                    batch_size = min(DEFAULT_MAX_BATCH, int(batch_size * 1.25))

            # decode a batch of surfaces
            surfaces = []
            for _ in range(batch_size):
                try:
                    surf = nvDec.DecodeSingleSurface()
                    if surf.Empty():
                        break
                    surfaces.append((idx, surf))
                except Exception:
                    break
                idx += 1

            if not surfaces:
                break

            # convert on GPU
            rgb_surfaces = []
            for frame_idx, surf in surfaces:
                try:
                    if two_stage:
                        nv12 = to_nv12.Execute(surf) if cc_ctx is None else to_nv12.Execute(surf, cc_ctx)
                        rgb = to_rgb.Execute(nv12) if cc_ctx is None else to_rgb.Execute(nv12, cc_ctx)
                    else:
                        rgb = to_rgb.Execute(surf) if cc_ctx is None else to_rgb.Execute(surf, cc_ctx)
                    rgb_surfaces.append((frame_idx, rgb))
                except Exception:
                    continue

            # download + save selected frames from the rgb_surfaces
            for frame_idx, rgb_surf in rgb_surfaces:
                if saved >= max_frames:
                    break
                if frame_idx % step != 0:
                    continue
                try:
                    # download into host buffer
                    if not downloader.DownloadSingleSurface(rgb_surf, host_full):
                        continue
                    frame = np.transpose(host_full, (1,2,0))  # HWC RGB
                    h, w = frame.shape[:2]
                    s = min(h, w)
                    y, x = (h - s) // 2, (w - s) // 2
                    crop = frame[y:y+s, x:x+s]
                    resized = cv2.resize(crop, (resize, resize), interpolation=cv2.INTER_LINEAR)

                    out_path = str(out_dir / f"{saved:03d}.jpg")
                    # submit CPU encode/write to thread pool
                    fut = save_executor.submit(_async_write_numpy_image, out_path, resized)
                    save_futures.append(fut)
                    saved += 1
                    processed_count += 1

                    # REMOVED ALL INTERMEDIATE LOGGING - LOG_EVERY_N_VIDEOS = 99999

                except Exception:
                    continue

        # wait for writes
        for fut in as_completed(save_futures):
            try:
                fut.result()
            except Exception:
                pass
        save_executor.shutdown(wait=True)

        # REMOVED FINAL VRAM PRINT - No more "[GPU END]" logs

        return saved > 0

    except Exception as e:
        # Silent failure, no logging
        return _extract_opencv(video_path, out_dir, fps_sample, max_frames, resize)


class FrameExtractor:
    def __init__(self, out_root="data/processed/frames", fps_sample=1, max_frames=8,
                 resize=224, num_workers=32, gpu_id=0):
        Path(out_root).mkdir(parents=True, exist_ok=True)
        self.out_root = out_root
        self.fps_sample = fps_sample
        self.max_frames = max_frames
        self.resize = resize
        self.num_workers = num_workers
        self.gpu_id = gpu_id

    def extract_to_folder(self, video_path: str, video_id: Optional[str] = None) -> str:
        if video_id is None:
            video_id = Path(video_path).stem
        args = (video_path, video_id, self.out_root, self.fps_sample,
                self.max_frames, self.resize, self.gpu_id, self.num_workers)
        if _extract_single_video(args):
            return str(Path(self.out_root) / video_id)
        raise RuntimeError(f"Failed: {video_path}")