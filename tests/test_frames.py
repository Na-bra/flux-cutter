from pathlib import Path

import numpy as np
import pytest

from app.video.frames import FrameExtractionError, extract_frames
from app.video.loader import load_video

VIDEO_PATH = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test.mp4"


@pytest.fixture(scope="module")
def video_container():
    """Fixture to open and close a video container for tests."""
    container = load_video(VIDEO_PATH)
    try:
        yield container
    finally:
        container.close()


def test_extract_frames_at_interval(video_container):
    """
    Tests that frames are extracted at the correct interval.
    The test video is ~23.36s long. An interval of 10s should yield 3 frames.
    (t=0s, t=10s, t=20s)
    """
    frames = list(extract_frames(video_container, sample_interval=10.0))

    assert len(frames) == 3

    # Check properties of the first extracted frame
    timestamp, image = frames[0]
    assert isinstance(timestamp, float)
    assert timestamp == pytest.approx(0.0, abs=0.1)
    assert isinstance(image, np.ndarray)
    assert image.shape == (2160, 2160, 3)  # height, width, channels
    assert image.dtype == np.uint8

    # Check that subsequent timestamps are approximately correct
    assert frames[1][0] >= 10.0
    assert frames[2][0] >= 20.0


def test_extract_frames_with_short_interval(video_container):
    """
    Tests extraction with a 1-second interval.
    The video is ~23.36s long, so it should yield 24 frames (for t=0 through t=23).
    """
    frames = list(extract_frames(video_container, sample_interval=1.0))
    assert len(frames) == 24


def test_extract_frames_with_long_interval(video_container):
    """
    Tests that an interval longer than the video duration yields only the first frame.
    """
    frames = list(extract_frames(video_container, sample_interval=30.0))
    assert len(frames) == 1


def test_extract_frames_with_invalid_interval_raises_error(video_container):
    """Tests that a zero or negative interval raises a ValueError."""
    with pytest.raises(ValueError, match="sample_interval must be greater than 0"):
        extract_frames(video_container, sample_interval=0)

    with pytest.raises(ValueError, match="sample_interval must be greater than 0"):
        extract_frames(video_container, sample_interval=-10.0)


def test_extract_frames_raises_for_no_video_stream():
    """Tests that FrameExtractionError is raised if the container has no video."""

    class MockStream:
        def __init__(self, stream_type):
            self.type = stream_type

    class MockContainer:
        streams = [MockStream("audio")]

        # The methods below are not called before the stream check,
        # but are included for completeness.
        def seek(self, *args, **kwargs):
            pass

        def decode(self, *args, **kwargs):
            return []

    with pytest.raises(FrameExtractionError, match="does not contain a video stream"):
        extract_frames(MockContainer())

def test_extract_frames_streams_lazily(video_container):
    """Tests that extraction is lazy and does not materialize the video.

    This is the property that lets the tool run on a full-length video at
    all: holding every sampled frame at once cost roughly
    width x height x 3 x (duration / interval) bytes, which on a 22-minute
    720p episode was ~4.9 GB at a 1.0s interval and ~15 GB at 0.25s.
    """
    frames = extract_frames(video_container, sample_interval=1.0)

    # An iterator, not a sequence: no length, no indexing.
    assert iter(frames) is frames
    with pytest.raises(TypeError):
        len(frames)

    # Taking one frame must not require decoding the rest.
    first_timestamp, first_image = next(iter(frames))
    assert first_timestamp == pytest.approx(0.0, abs=0.1)
    assert first_image.shape == (2160, 2160, 3)


def test_extract_frames_validates_arguments_eagerly(video_container):
    """Tests that bad arguments raise at the call, not at first iteration.

    A plain generator function would defer the whole body -- including
    validation -- until something iterated it, so a caller passing a bad
    interval would get no error until much later, far from the mistake.
    """
    with pytest.raises(ValueError):
        extract_frames(video_container, sample_interval=0)

    with pytest.raises(ValueError):
        extract_frames(video_container, sample_interval=-1.0)
