import os
import cv2
import math
from typing import List, Optional

class FrameExtractor:
    """
    Streams frames from a video file without loading entire video.
    - fps_sample: frames per second to sample
    - max_frames: cap frames per video (8-16)
    - resize: target resolution (int) max 256
    """
    def __init__(self, out_root="data/processed/frames", fps_sample=1, max_frames=8, resize=224):
        assert resize in (128, 224, 256), "resize must be one of 128,224,256"
        os.makedirs(out_root, exist_ok=True)
        self.out_root = out_root
        self.fps_sample = fps_sample
        self.max_frames = max_frames
        self.resize = resize

    def extract_to_folder(self, video_path: str, video_id: Optional[str]=None) -> str:
        if video_id is None:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(self.out_root, video_id)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / float(self.fps_sample))))
        saved = 0
        idx = 0
        while saved < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                # resize while preserving aspect ratio and center crop to square
                h, w = frame.shape[:2]
                short = min(h, w)
                # center crop
                start_y = (h - short) // 2
                start_x = (w - short) // 2
                crop = frame[start_y:start_y+short, start_x:start_x+short]
                resized = cv2.resize(crop, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
                out_path = os.path.join(out_dir, f"{saved:03d}.jpg")
                cv2.imwrite(out_path, resized)
                saved += 1
            idx += 1
        cap.release()
        return out_dir

    def extract_batch(self, video_paths: List[str]):
        out_dirs = []
        for vp in video_paths:
            out_dirs.append(self.extract_to_folder(vp))
        return out_dirs