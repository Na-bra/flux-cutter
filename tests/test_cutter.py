"""Tests for the in-process cutter.

These encode real video, which is why there are few of them and why they use
the short test clip. The expensive properties -- that the cut lands where it
was asked to, that the joins are seamless, that no frames are lost at a
segment boundary -- cannot be checked without actually writing a file.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.video.cutter import CutterError, cut_segments
from app.video.source import VideoSource
from app.video.timeline import AppearanceInterval

VIDEO = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test.mp4"


def video_frame_count(path: Path) -> int:
    """Counts decoded video frames, which is the only count that matters."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip("ffprobe unavailable; cannot verify the written file")
    return int(result.stdout.strip())


def spans(*pairs):
    return [AppearanceInterval(start_time=s, end_time=e) for s, e in pairs]


def test_no_segments_is_refused():
    with pytest.raises(CutterError, match="No segments"):
        cut_segments(VIDEO, [], Path("unused.mp4"))


def test_overlapping_segments_are_refused(tmp_path):
    """Overlaps would repeat footage in the reel."""
    with pytest.raises(CutterError, match="overlap"):
        cut_segments(VIDEO, spans((1.0, 5.0), (4.0, 8.0)), tmp_path / "out.mp4")


def test_a_missing_source_is_reported_not_raised_raw(tmp_path):
    with pytest.raises(CutterError, match="Could not open"):
        cut_segments(tmp_path / "nope.mp4", spans((1.0, 2.0)), tmp_path / "out.mp4")


@pytest.mark.slow
def test_one_segment_keeps_every_frame_it_asked_for(tmp_path):
    """30fps for 2s is 60 frames, and all 60 should survive the cut."""
    output = tmp_path / "one.mp4"
    result = cut_segments(VIDEO, spans((2.0, 4.0)), output, quality=30)

    assert output.exists()
    assert result.segment_count == 1
    assert video_frame_count(output) == 60


@pytest.mark.slow
def test_joining_segments_loses_no_frames(tmp_path):
    """The bug this guards is subtle and was real.

    Video and audio decode interleaved, and audio runs ahead. Ending a
    segment on the first frame past its end therefore stopped on an audio
    frame and discarded the video still to come -- 7 frames a segment, 339
    of an expected 360 across a three-segment reel, with nothing in the
    output looking obviously wrong.
    """
    output = tmp_path / "joined.mp4"
    segments = spans((1.0, 5.0), (9.0, 13.0), (18.0, 22.0))

    result = cut_segments(VIDEO, segments, output, quality=30)

    assert result.segment_count == 3
    # 12 seconds of 30fps footage, whole.
    assert video_frame_count(output) == 360


@pytest.mark.slow
def test_dropping_audio_still_produces_a_whole_reel(tmp_path):
    output = tmp_path / "silent.mp4"

    cut_segments(VIDEO, spans((1.0, 5.0), (9.0, 13.0)), output,
                 include_audio=False, quality=30)

    assert video_frame_count(output) == 240


@pytest.mark.slow
def test_on_segment_reports_each_cut_in_order(tmp_path):
    seen = []
    cut_segments(
        VIDEO, spans((1.0, 3.0), (5.0, 7.0)), tmp_path / "out.mp4", quality=30,
        on_segment=lambda i, total, seg: seen.append((i, total)),
    )

    assert seen == [(0, 2), (1, 2)]


@pytest.mark.slow
def test_raising_from_on_segment_aborts_the_cut(tmp_path):
    """This is how the window implements cancellation."""
    class Stop(Exception):
        pass

    def stop(index, total, segment):
        raise Stop

    with pytest.raises(Stop):
        cut_segments(VIDEO, spans((1.0, 3.0), (5.0, 7.0)), tmp_path / "out.mp4",
                     quality=30, on_segment=stop)


@pytest.mark.slow
def test_a_video_moved_since_the_scan_still_cuts(tmp_path):
    """The reason VideoSource exists, proved by encoding through it.

    A scan takes minutes; the user is free to reorganise their folders
    while looking at the gallery. The held descriptor means the export
    never notices -- here the file is unlinked outright, so no path on the
    machine reaches the footage, and the reel is cut anyway.
    """
    copy = tmp_path / "moved.mp4"
    shutil.copy(VIDEO, copy)
    output = tmp_path / "out.mp4"

    with VideoSource(copy, keep_open=True) as source:
        os.unlink(copy)
        assert not copy.exists()
        result = cut_segments(source, spans((2.0, 4.0)), output, quality=30)

    assert result.segment_count == 1
    assert video_frame_count(output) == 60


def test_a_source_whose_footage_is_gone_reports_it(tmp_path):
    """With no descriptor held, a lost file has to surface as a CutterError."""
    copy = tmp_path / "gone.mp4"
    copy.write_bytes(b"placeholder" * 100)

    source = VideoSource(copy, keep_open=False)
    os.unlink(copy)

    with pytest.raises(CutterError, match="no longer at"):
        cut_segments(source, spans((1.0, 2.0)), tmp_path / "out.mp4")
