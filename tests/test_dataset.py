from dataset import BaseDataset
import tensorflow as tf


def test_synthetic_sample_shapes():
    ds = BaseDataset()
    frames, labels = ds.synthetic_sample(batch_size=2)
    assert frames.shape == (2, ds.frame_count, *ds.img_size, 3)
    assert labels.shape == (2,)
    assert labels.dtype == tf.int32
