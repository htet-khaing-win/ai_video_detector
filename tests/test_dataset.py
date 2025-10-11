import os
from src.data.debug_generator import generate_debug_videos
from src.data.frame_extractor import FrameExtractor
from src.data.cache_preprocessed import cache_from_frame_folder
from src.data.preprocess_pytorch import create_dataloader

def test_full_pipeline(tmp_path):
    out_debug = tmp_path / "debug"
    out_debug = str(out_debug)
    generate_debug_videos(out_dir=out_debug, num_videos=20, res=128, frames=8)
    fe = FrameExtractor(out_root=str(tmp_path / "frames"), fps_sample=1, max_frames=8, resize=128)
    vids = [os.path.join(out_debug, p) for p in os.listdir(out_debug)]
    fe.extract_batch(vids)
    # cache
    import shutil
    from src.data.cache_preprocessed import cache_from_frame_folder
    cache_from_frame_folder(frames_root=str(tmp_path / "frames"), cache_root=str(tmp_path / "cache"))
    # create fake metadata CSV
    meta = str(tmp_path / "meta.csv")
    with open(meta, "w") as fh:
        fh.write("file,label\n")
        for i,v in enumerate(os.listdir(out_debug)):
            lab = 0 if "real" in v else 1
            fh.write(f"{v},{lab}\n")
    dl = create_dataloader(meta, batch_size=2, num_workers=0, clip_mode=True)
    batch = next(iter(dl))
    assert batch[0].shape[0] == 2
