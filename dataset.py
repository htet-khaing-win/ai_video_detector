import os
import tensorflow as tf
import logging
from utils.setup_env import setup_environment

class BaseDataset:
    """Abstract dataset handler for multimodal data."""

    def __init__(self, data_dir="data/raw/sample_dataset", img_size=(224, 224), frame_count=8):
        self.data_dir = data_dir
        self.img_size = img_size
        self.frame_count = frame_count
        self.config = setup_environment()
        logging.info(f"Initialized BaseDataset at {data_dir}")

    def list_videos(self):
        """List all video file paths."""
        exts = (".mp4", ".avi", ".mov")
        videos = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(exts)]
        logging.info(f"Found {len(videos)} videos")
        return videos

    def synthetic_sample(self, batch_size=2):
        """Generate dummy tensors to test model pipelines."""
        frames = tf.random.uniform((batch_size, self.frame_count, *self.img_size, 3))
        labels = tf.random.uniform((batch_size,), maxval=2, dtype=tf.int32)
        return frames, labels