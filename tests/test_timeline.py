import numpy as np
import pytest

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.video.timeline import (
    AppearanceInterval,
    build_appearance_intervals,
    format_timestamp,
)


def make_group(timestamps, group_id: int = 1) -> FaceIdentityGroup:
    """Builds a FaceIdentityGroup whose observations only carry the fields timeline.py reads."""
    box = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
    detection = FaceDetection(box=box, confidence=0.9)
    face_crop = np.zeros((4, 4, 3), dtype=np.uint8)
    embedding = np.array([1.0], dtype=np.float32)

    observations = [
        FaceObservation(
            embedding=embedding,
            detection=detection,
            face_crop=face_crop,
            source_timestamp=timestamp,
        )
        for timestamp in timestamps
    ]
    return FaceIdentityGroup(group_id=group_id, observations=observations)


def test_single_appearance_from_nearby_detections():
    """Detections close together in time collapse into one interval."""
    group = make_group([4.23, 4.57, 5.10, 5.63])
    intervals = build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0)

    assert len(intervals) == 1
    assert intervals[0].start_time < 4.23
    assert intervals[0].end_time > 5.63


def test_multiple_appearances_from_separated_detections():
    """Detections split by a large gap become two separate appearances."""
    group = make_group([4.23, 4.57, 5.10, 20.43, 20.77])
    intervals = build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0)

    assert len(intervals) == 2
    assert intervals[0].end_time < intervals[1].start_time
    assert intervals[0].start_time < 4.23
    assert intervals[1].end_time > 20.77


def test_gap_handling_uses_configured_tolerance():
    """A gap right at the configured tolerance boundary is respected."""
    # Default tolerance at sample_interval=1.0 is 2.0s.
    within_tolerance = make_group([0.0, 2.0])
    across_tolerance = make_group([0.0, 2.1])

    merged = build_appearance_intervals(within_tolerance, video_duration=30.0, sample_interval=1.0)
    split = build_appearance_intervals(across_tolerance, video_duration=30.0, sample_interval=1.0)

    assert len(merged) == 1
    assert len(split) == 2


def test_explicit_gap_tolerance_overrides_default():
    group = make_group([0.0, 3.0])

    split = build_appearance_intervals(
        group, video_duration=30.0, sample_interval=1.0, gap_tolerance_seconds=1.0
    )
    merged = build_appearance_intervals(
        group, video_duration=30.0, sample_interval=1.0, gap_tolerance_seconds=5.0
    )

    assert len(split) == 2
    assert len(merged) == 1


def test_overlapping_intervals_are_merged():
    """Padding that pushes two spans into overlap must merge them back together."""
    # Two spans 2.5s apart; with padding_seconds=2.0 they overlap after padding.
    group = make_group([0.0, 10.0, 10.5])
    intervals = build_appearance_intervals(
        group,
        video_duration=30.0,
        sample_interval=1.0,
        gap_tolerance_seconds=0.6,
        padding_seconds=2.0,
    )

    # [0.0] -> [-2, 2] clamped to [0, 2]; [10.0, 10.5] -> [8, 12.5]; these don't touch,
    # but a second case with closer raw spans should merge:
    assert len(intervals) == 2

    close_group = make_group([0.0, 3.5, 4.0])
    close_intervals = build_appearance_intervals(
        close_group,
        video_duration=30.0,
        sample_interval=1.0,
        gap_tolerance_seconds=0.6,
        padding_seconds=2.0,
    )
    # [0.0] -> [0, 2.0]; [3.5, 4.0] -> [1.5, 6.0]; these overlap and must merge.
    assert len(close_intervals) == 1
    assert close_intervals[0].start_time == pytest.approx(0.0)
    assert close_intervals[0].end_time == pytest.approx(6.0)


def test_sorting_from_unordered_detections():
    """Intervals come back chronologically sorted regardless of input order."""
    group = make_group([31.0, 32.0, 2.0, 3.0, 15.0, 16.0])
    intervals = build_appearance_intervals(group, video_duration=40.0, sample_interval=1.0)

    starts = [interval.start_time for interval in intervals]
    assert starts == sorted(starts)


def test_intervals_are_clamped_to_video_duration():
    """Padding must never push an interval below 0 or past the video's duration."""
    group = make_group([0.05, 59.9])
    intervals = build_appearance_intervals(
        group, video_duration=60.0, sample_interval=1.0, gap_tolerance_seconds=1000.0, padding_seconds=5.0
    )

    assert len(intervals) == 1
    assert intervals[0].start_time == pytest.approx(0.0)
    assert intervals[0].end_time == pytest.approx(60.0)
    for interval in intervals:
        assert 0.0 <= interval.start_time <= interval.end_time <= 60.0


def test_empty_input_produces_no_intervals():
    group = make_group([])
    intervals = build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0)
    assert intervals == []


def test_single_detection_produces_a_valid_short_interval():
    group = make_group([12.0])
    intervals = build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0)

    assert len(intervals) == 1
    interval = intervals[0]
    assert interval.start_time < 12.0 < interval.end_time
    assert interval.end_time - interval.start_time > 0


def test_timestamps_come_from_original_video_timeline_not_sample_index():
    """Interval boundaries must track real timestamps, not a reconstructed 0,1,2... index."""
    # Three detections whose *sample index* would be 0,1,2 but whose real
    # video timestamps are far apart and irregular (as PyAV's real PTS
    # values are, unlike an evenly-spaced synthetic index).
    group = make_group([0.033, 4.9, 19.97])
    intervals = build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0)

    all_bounds = [t for interval in intervals for t in (interval.start_time, interval.end_time)]
    assert min(all_bounds) < 0.5
    assert max(all_bounds) > 19.9
    # If timestamps had been reconstructed as sample index * interval
    # (0, 1, 2 seconds), every bound would sit under 3s -- they must not.
    assert any(bound > 3.0 for bound in all_bounds)


def test_padding_and_gap_tolerance_scale_with_sample_interval():
    """Sampling-aware defaults should widen/narrow with the actual sampling rate used."""
    group = make_group([10.0])

    fine = build_appearance_intervals(group, video_duration=30.0, sample_interval=0.5)
    coarse = build_appearance_intervals(group, video_duration=30.0, sample_interval=2.0)

    fine_width = fine[0].end_time - fine[0].start_time
    coarse_width = coarse[0].end_time - coarse[0].start_time
    assert coarse_width > fine_width


def test_rejects_invalid_arguments():
    group = make_group([1.0])

    with pytest.raises(ValueError):
        build_appearance_intervals(group, video_duration=-1.0, sample_interval=1.0)

    with pytest.raises(ValueError):
        build_appearance_intervals(group, video_duration=30.0, sample_interval=0.0)

    with pytest.raises(ValueError):
        build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0, gap_tolerance_seconds=-1.0)

    with pytest.raises(ValueError):
        build_appearance_intervals(group, video_duration=30.0, sample_interval=1.0, padding_seconds=-1.0)


def test_appearance_interval_is_a_plain_frozen_dataclass():
    interval = AppearanceInterval(start_time=1.0, end_time=2.0)
    assert interval.start_time == 1.0
    assert interval.end_time == 2.0
    with pytest.raises(Exception):
        interval.start_time = 5.0


def test_format_timestamp_matches_expected_display_format():
    assert format_timestamp(0.0) == "00:00.00"
    assert format_timestamp(4.23) == "00:04.23"
    assert format_timestamp(65.5) == "01:05.50"


def test_format_timestamp_rolls_seconds_over_into_minutes():
    """59.999s must round up into the next minute, not display an invalid '60.00' seconds field."""
    assert format_timestamp(59.999) == "01:00.00"


def test_format_timestamp_rejects_negative_input():
    with pytest.raises(ValueError):
        format_timestamp(-1.0)
