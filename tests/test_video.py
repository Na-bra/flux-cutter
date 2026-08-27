from pathlib import Path

import pytest

from app.video.loader import VideoLoadError, get_video_info, load_video

VIDEO_DIR = Path(__file__).resolve().parents[1] / "assets" / "test-videos"
MP4_VIDEO_PATH = VIDEO_DIR / "test.mp4"
MOV_VIDEO_PATH = VIDEO_DIR / "test_2.mov"


@pytest.mark.parametrize(
    "video_path",
    [
        pytest.param(MP4_VIDEO_PATH, id="mp4"),
        pytest.param(
            MOV_VIDEO_PATH,
            id="mov",
            marks=pytest.mark.skipif(
                not MOV_VIDEO_PATH.exists(), reason=f"Test file not found: {MOV_VIDEO_PATH}"
            ),
        ),
    ],
)
def test_load_video_returns_container_for_supported_files(video_path):
    """Tests that supported video file types can be loaded."""
    container = load_video(video_path)

    try:
        assert container is not None
        assert any(stream.type == "video" for stream in container.streams)
    finally:
        container.close()


def test_load_video_raises_for_missing_file():
    missing = VIDEO_DIR / "missing.mp4"

    with pytest.raises(VideoLoadError, match="does not exist"):
        load_video(missing)


def test_load_video_raises_for_unsupported_format(tmp_path):
    bad_file = tmp_path / "video.txt"
    bad_file.write_text("not a video")

    with pytest.raises(VideoLoadError, match="Unsupported video format"):
        load_video(bad_file)


def test_get_video_info_returns_basic_stream_data():
    container = load_video(MP4_VIDEO_PATH)

    try:
        info = get_video_info(container)
    finally:
        container.close()

    assert info["width"] == 2160
    assert info["height"] == 2160
    assert info["codec"] == "h264"
    assert info["frames"] == 701
    assert info["fps"] == pytest.approx(30.0)
    assert info["duration"] == pytest.approx(23.3666666667, rel=1e-6)


def test_a_file_that_is_not_a_video_raises_a_readable_error(tmp_path):
    """PyAV 18 dropped av.AVError, which the loader used to catch.

    The except clause itself then raised AttributeError, so a corrupt file
    surfaced as "module 'av' has no attribute 'AVError'" rather than as
    anything a user could act on.
    """
    not_a_video = tmp_path / "broken.mp4"
    not_a_video.write_text("this is not a video at all")

    with pytest.raises(VideoLoadError) as caught:
        load_video(not_a_video)

    assert "Could not open video" in str(caught.value)
