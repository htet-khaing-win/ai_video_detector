import os
from src.data.debug_generator import generate_debug_videos
from src.data.frame_extractor import FrameExtractor

def test_extractor_runs(tmp_path):
    out_debug = tmp_path / "debug"
    out_debug = str(out_debug)
    # generate small debug set (10 videos)
    from src.data.debug_generator import generate_debug_videos
    generate_debug_videos(out_dir=out_debug, num_videos=10, res=128, frames=8)
    video = os.listdir(out_debug)[0]
    ve = FrameExtractor(out_root=str(tmp_path / "frames"), fps_sample=1, max_frames=8, resize=128)
    out = ve.extract_to_folder(os.path.join(out_debug, video))
    files = os.listdir(out)
    assert len(files) > 0
