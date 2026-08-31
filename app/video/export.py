from app.video.timeline import AppearanceInterval, merge_spans

# Two consecutive appearances closer together than this are bridged into one
# segment rather than cut apart and rejoined.
#
# This is an editorial threshold, not a detection one, and the distinction
# matters: build_appearance_intervals already merged detections separated by
# less than its sampling-derived gap tolerance, because a gap that small is
# probably a missed sample rather than the person leaving. What survives that
# pass are real absences -- the person genuinely turned away, or the shot cut
# to someone else and back. Some of those are still too short to cut on. A
# one-second cutaway and return reads as a glitch in the finished reel, not as
# an edit, so it is smoother to hold the shot through it.
DEFAULT_BRIDGE_GAP_SECONDS = 1.5
# Segments shorter than this are extended rather than emitted as-is.
#
# On the 22-minute test footage the lead character's 153 appearances have a
# median duration of 3.0s and 75 of them run under 3s. Cutting those verbatim
# produces a strobing reel that is technically correct and unwatchable, which
# fails the stage-0.4 "final video watchable" criterion on its own.
DEFAULT_MIN_SEGMENT_SECONDS = 2.0
# Extra headroom added to each side before cutting, so a segment does not open
# or close mid-word or mid-gesture.
#
# This is *additional* to whatever padding build_appearance_intervals already
# applied (half a sampling step by default), and the two compound. It is small
# because the detection-side padding has usually done most of the work.
DEFAULT_EXPORT_PADDING_SECONDS = 0.25


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def merge_for_export(
    intervals: list[AppearanceInterval],
    video_duration: float,
    bridge_gap_seconds: float | None = None,
    min_segment_seconds: float | None = None,
    padding_seconds: float | None = None,
) -> list[AppearanceInterval]:
    """
    Turns raw appearance intervals into segments worth actually cutting.

    `build_appearance_intervals` answers "when was this person on screen",
    which is a question about detection. Cutting asks a different question --
    "what should the finished reel contain" -- and the two disagree. On the
    22-minute test footage the lead's appearances come back as 153 intervals
    with a median duration of 3.0s, half of them under 3s. Cut literally, that
    is a strobe rather than a reel.

    So this pass widens each interval for headroom, bridges gaps too short to
    cut across, and extends anything still under the minimum length. It is
    deliberately pure and separate from the cutting itself: the judgement about
    what makes a watchable segment is worth testing without encoding a frame.

    Extending short segments can close gaps that were previously wide enough to
    keep, so merging runs again afterwards; the result is always
    non-overlapping and in chronological order.

    Args:
        intervals: Appearance intervals for one identity, any order.
        video_duration: Source duration in seconds; segments are clamped to it.
        bridge_gap_seconds: Gaps at or below this are held through rather than
            cut. Defaults to DEFAULT_BRIDGE_GAP_SECONDS.
        min_segment_seconds: Shorter segments are grown around their midpoint.
            Defaults to DEFAULT_MIN_SEGMENT_SECONDS.
        padding_seconds: Extra headroom per side, additional to any padding the
            intervals already carry. Defaults to DEFAULT_EXPORT_PADDING_SECONDS.

    Returns:
        Non-overlapping AppearanceInterval objects in chronological order.
        A segment can still be shorter than min_segment_seconds if the video
        itself is: the video's bounds win over the preference.
    """
    if video_duration < 0:
        raise ValueError("video_duration must be non-negative")
    if bridge_gap_seconds is None:
        bridge_gap_seconds = DEFAULT_BRIDGE_GAP_SECONDS
    if bridge_gap_seconds < 0:
        raise ValueError("bridge_gap_seconds must be non-negative")
    if min_segment_seconds is None:
        min_segment_seconds = DEFAULT_MIN_SEGMENT_SECONDS
    if min_segment_seconds < 0:
        raise ValueError("min_segment_seconds must be non-negative")
    if padding_seconds is None:
        padding_seconds = DEFAULT_EXPORT_PADDING_SECONDS
    if padding_seconds < 0:
        raise ValueError("padding_seconds must be non-negative")

    if not intervals:
        return []

    padded = [
        (
            _clamp(interval.start_time - padding_seconds, 0.0, video_duration),
            _clamp(interval.end_time + padding_seconds, 0.0, video_duration),
        )
        for interval in intervals
    ]
    spans = merge_spans(padded, bridge_gap_seconds)

    # Grow anything still too short around its own midpoint, so the extra time
    # is taken evenly from both sides rather than always running late.
    grown: list[tuple[float, float]] = []
    for start, end in spans:
        shortfall = min_segment_seconds - (end - start)
        if shortfall > 0:
            midpoint = (start + end) / 2
            half = min_segment_seconds / 2
            start = _clamp(midpoint - half, 0.0, video_duration)
            end = _clamp(midpoint + half, 0.0, video_duration)
        grown.append((start, end))

    # Growing can close a gap that was wide enough to keep a moment ago.
    final_spans = merge_spans(grown, bridge_gap_seconds)

    return [AppearanceInterval(start_time=start, end_time=end) for start, end in final_spans]


class ExportError(Exception):
    """Raised when a clip cannot be cut or the segments cannot be joined."""
