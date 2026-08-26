import pytest

from app.video.export import (
    DEFAULT_BRIDGE_GAP_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    merge_for_export,
)
from app.video.timeline import AppearanceInterval


def intervals(*pairs) -> list[AppearanceInterval]:
    return [AppearanceInterval(start_time=s, end_time=e) for s, e in pairs]


def spans(result) -> list[tuple[float, float]]:
    return [(round(i.start_time, 3), round(i.end_time, 3)) for i in result]


def test_empty_input_produces_no_segments():
    assert merge_for_export([], video_duration=100.0) == []


def test_short_gap_is_bridged_rather_than_cut():
    """A cutaway shorter than the bridge threshold is held through.

    Cutting away and back for under a second reads as a glitch in the
    finished reel rather than as an edit.
    """
    result = merge_for_export(
        intervals((10.0, 14.0), (14.5, 18.0)),
        video_duration=100.0,
        bridge_gap_seconds=1.5,
        padding_seconds=0.0,
    )

    assert spans(result) == [(10.0, 18.0)]


def test_long_gap_stays_as_two_segments():
    """A genuine absence is still cut apart."""
    result = merge_for_export(
        intervals((10.0, 14.0), (40.0, 44.0)),
        video_duration=100.0,
        bridge_gap_seconds=1.5,
        padding_seconds=0.0,
    )

    assert spans(result) == [(10.0, 14.0), (40.0, 44.0)]


def test_short_segment_is_grown_around_its_midpoint():
    """A brief appearance is extended evenly, not just lengthened at the end."""
    result = merge_for_export(
        intervals((50.0, 51.0),),
        video_duration=100.0,
        min_segment_seconds=4.0,
        padding_seconds=0.0,
    )

    # midpoint 50.5, so 48.5 -> 52.5 rather than 50.0 -> 54.0
    assert spans(result) == [(48.5, 52.5)]


def test_growing_can_close_a_gap_and_triggers_a_second_merge():
    """Two segments that only overlap *after* being grown must still merge.

    Growing happens after the first merge pass, so without a second pass this
    returns overlapping segments and the concatenated output repeats footage.
    """
    result = merge_for_export(
        intervals((10.0, 10.5), (13.0, 13.5)),
        video_duration=100.0,
        bridge_gap_seconds=0.5,
        min_segment_seconds=4.0,
        padding_seconds=0.0,
    )

    assert len(result) == 1
    start, end = spans(result)[0]
    assert start <= 10.25 and end >= 13.25


def test_segments_never_overlap_or_run_backwards():
    """Whatever the inputs, the output must be safe to concatenate."""
    result = merge_for_export(
        intervals((5.0, 9.0), (8.0, 12.0), (12.2, 13.0), (30.0, 31.0)),
        video_duration=100.0,
    )

    for earlier, later in zip(result, result[1:]):
        assert earlier.end_time <= later.start_time
    for segment in result:
        assert segment.start_time < segment.end_time


def test_segments_are_clamped_to_the_video():
    """Padding and growth must not run past either end of the source."""
    result = merge_for_export(
        intervals((0.1, 0.4), (99.6, 99.9)),
        video_duration=100.0,
        min_segment_seconds=6.0,
        padding_seconds=1.0,
    )

    assert result[0].start_time >= 0.0
    assert result[-1].end_time <= 100.0


def test_a_video_shorter_than_the_minimum_segment_is_not_padded_past_itself():
    """The video's real bounds win over the minimum-length preference."""
    result = merge_for_export(
        intervals((0.5, 1.0),),
        video_duration=2.0,
        min_segment_seconds=30.0,
        padding_seconds=0.0,
    )

    assert spans(result) == [(0.0, 2.0)]


def test_input_order_does_not_matter():
    out_of_order = merge_for_export(
        intervals((40.0, 44.0), (10.0, 14.0)), video_duration=100.0
    )
    in_order = merge_for_export(
        intervals((10.0, 14.0), (40.0, 44.0)), video_duration=100.0
    )

    assert spans(out_of_order) == spans(in_order)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_duration": -1.0},
        {"video_duration": 10.0, "bridge_gap_seconds": -1.0},
        {"video_duration": 10.0, "min_segment_seconds": -1.0},
        {"video_duration": 10.0, "padding_seconds": -1.0},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        merge_for_export(intervals((1.0, 2.0)), **kwargs)


def test_defaults_are_sane_relative_to_each_other():
    """A bridged gap should be shorter than the shortest segment worth keeping."""
    assert DEFAULT_BRIDGE_GAP_SECONDS < DEFAULT_MIN_SEGMENT_SECONDS
