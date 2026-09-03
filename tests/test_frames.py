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


# ------------------------------------------- sampling an irregular timeline


class FakeFrame:
    def __init__(self, time: float):
        self.time = time

    def to_ndarray(self, format=None):
        return np.zeros((2, 2, 3), dtype=np.uint8)


class FakeContainer:
    """Just enough container to drive the sampling schedule.

    Real footage in this repository starts at zero and has no gaps, so the
    schedule can only be exercised against timings that no sample file
    here provides.
    """

    class _Stream:
        type = "video"

    def __init__(self, times):
        self._times = times
        self.streams = [self._Stream()]

    def seek(self, offset):
        pass

    def decode(self, stream):
        return (FakeFrame(t) for t in self._times)


def sampled_times(times, sample_interval):
    return [t for t, _ in extract_frames(FakeContainer(times), sample_interval)]


@pytest.mark.runs_without_assets
def test_footage_that_starts_late_is_not_sampled_in_a_burst():
    """An MP4 with a start offset, or a clip cut from the middle of a file.

    The schedule starts at zero, so the first frame arrives with the
    target far behind it. Advancing by a single interval would leave it
    still behind, and every following frame would qualify until it caught
    up -- six frames spanning 0.2 seconds where one was wanted.
    """
    times = [5.0 + i / 24 for i in range(240)]

    got = sampled_times(times, 1.0)

    assert got[:4] == [5.0, 6.0, 7.0, 8.0]
    assert min(round(b - a, 6) for a, b in zip(got, got[1:])) >= 1.0


@pytest.mark.runs_without_assets
def test_a_gap_does_not_produce_near_duplicate_samples():
    """Dropped frames and variable frame rates both leave gaps."""
    times = [i / 24 for i in range(48)] + [8.0 + i / 24 for i in range(48)]

    assert sampled_times(times, 1.0) == [0.0, 1.0, 8.0, 9.0]


@pytest.mark.runs_without_assets
def test_a_regular_timeline_still_lands_on_the_interval():
    """The ordinary case has to be untouched by the fix above."""
    times = [i / 25 for i in range(250)]

    assert sampled_times(times, 2.0) == [0.0, 2.0, 4.0, 6.0, 8.0]


@pytest.mark.runs_without_assets
def test_sampling_does_not_drift_over_a_long_timeline():
    """Spacing from each yielded frame instead of the interval grid would
    accumulate a fraction of a frame every sample."""
    times = [i / 30 for i in range(30 * 600)]

    got = sampled_times(times, 1.0)

    assert len(got) == 600
    assert got[-1] == pytest.approx(599.0, abs=1 / 30)
