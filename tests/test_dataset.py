import os
from src.data.debug_generator import generate_debug_videos
from src.data.frame_extractor import FrameExtractor
from src.data.cache_preprocessed import cache_from_frame_folder
from src.data.preprocess_pytorch import create_dataloader, GenBusterDataset

def test_full_pipeline(tmp_path):
    out_debug = str(tmp_path / "debug")
    cache_dir = str(tmp_path / "cache")
    frames_dir = str(tmp_path / "frames")
    meta = str(tmp_path / "meta.csv")

    # 1. Generate debug videos
    generate_debug_videos(out_dir=out_debug, num_videos=20, res=128, frames=8)

    # 2. Extract frames
    fe = FrameExtractor(out_root=frames_dir, fps_sample=1, max_frames=8, resize=128)
    vids = [os.path.join(out_debug, p) for p in os.listdir(out_debug)]
    fe.extract_batch(vids)

    # 3. Cache preprocessed tensors
    cache_from_frame_folder(frames_root=frames_dir, cache_root=cache_dir)

    # 4. Create fake metadata CSV
    with open(meta, "w") as fh:
        fh.write("file,label\n")
        for v in os.listdir(out_debug):
            vid_id = os.path.splitext(v)[0]
            label = 0 if "real" in v else 1
            fh.write(f"{vid_id},{label}\n")

    # 5. Create DataLoader — must pass cache_root explicitly
    dl = create_dataloader(meta, batch_size=2, num_workers=0, clip_mode=True, cache_root=cache_dir)

    batch = next(iter(dl))
    assert batch[0].shape[0] == 2

