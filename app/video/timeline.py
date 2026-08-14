from dataclasses import dataclass

from app.faces.grouper import FaceIdentityGroup

# FluxCutter samples frames rather than decoding every one, so a lone gap
# between two detections the size of a single sampling step is expected
# noise (the person may simply have fallen on the "off" frame of a
# sample), not proof they left the shot. Tolerating up to one missed
# sample before splitting into a new appearance is a small, principled
# default tied to the actual sampling rate rather than a fixed constant.
DEFAULT_GAP_TOLERANCE_SAMPLE_MULTIPLIER = 2.0
# A detection at time T only tells us the person was on screen somewhere
# within about half a sampling step of T — the next real appearance could
# have started up to interval/2 earlier, or still be continuing up to
# interval/2 later, and sampling wouldn't have caught it yet. Padding by
# half a sample is a conservative, sampling-derived default; the later
# clip-extraction stage may want to widen this further.
DEFAULT_PADDING_SAMPLE_FRACTION = 0.5


@dataclass(frozen=True)
class AppearanceInterval:
    """A contiguous span of original-video time, in seconds, during which a person appears."""

    start_time: float
    end_time: float


def format_timestamp(seconds: float) -> str:
    """Formats a duration in seconds as MM:SS.ss for display only."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")

    # Round to whole centiseconds first so e.g. 59.999s rolls over to
    # "01:00.00" instead of the seconds field itself rounding up to an
    # invalid "60.00".
    total_centiseconds = round(seconds * 100)
    minutes, remaining_centiseconds = divmod(total_centiseconds, 6000)
    return f"{minutes:02d}:{remaining_centiseconds / 100:05.2f}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_appearance_intervals(
    group: FaceIdentityGroup,
    video_duration: float,
    sample_interval: float,
    gap_tolerance_seconds: float | None = None,
    padding_seconds: float | None = None,
) -> list[AppearanceInterval]:
    """
    Converts one identity group's detection timestamps into appearance intervals.

    Detections already carry their original-video timestamp (see
    FaceObservation.source_timestamp, set from the real decoded frame PTS
    in app/video/frames.py — not reconstructed from a sample index), so
    this only needs to decide which nearby detections belong to the same
    appearance and where that appearance's boundaries should sit.

    Args:
        group: The single selected identity group to compute intervals for.
        video_duration: The source video's duration in seconds, used to
            clamp intervals so they never run past the end of the video.
        sample_interval: The sampling interval (seconds) used when frames
            were extracted for this run. Used to derive sampling-aware
            defaults for gap tolerance and padding.
        gap_tolerance_seconds: Maximum time between two consecutive
            detections before they're treated as separate appearances.
            Defaults to DEFAULT_GAP_TOLERANCE_SAMPLE_MULTIPLIER * sample_interval.
        padding_seconds: Time added before the first and after the last
            detection of each appearance. Defaults to
            DEFAULT_PADDING_SAMPLE_FRACTION * sample_interval.

    Returns:
        Non-overlapping AppearanceInterval objects in chronological order,
        clamped to [0, video_duration].
    """
    if video_duration < 0:
        raise ValueError("video_duration must be non-negative")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if gap_tolerance_seconds is None:
        gap_tolerance_seconds = DEFAULT_GAP_TOLERANCE_SAMPLE_MULTIPLIER * sample_interval
    if gap_tolerance_seconds < 0:
        raise ValueError("gap_tolerance_seconds must be non-negative")
    if padding_seconds is None:
        padding_seconds = DEFAULT_PADDING_SAMPLE_FRACTION * sample_interval
    if padding_seconds < 0:
        raise ValueError("padding_seconds must be non-negative")

    timestamps = sorted(observation.source_timestamp for observation in group.observations)
    if not timestamps:
        return []

    # Chain consecutive detections into raw spans wherever the gap between
    # them is within tolerance.
    raw_spans: list[list[float]] = []
    for timestamp in timestamps:
        if raw_spans and (timestamp - raw_spans[-1][-1]) <= gap_tolerance_seconds:
            raw_spans[-1].append(timestamp)
        else:
            raw_spans.append([timestamp])

    padded_spans = [
        (
            _clamp(span[0] - padding_seconds, 0.0, video_duration),
            _clamp(span[-1] + padding_seconds, 0.0, video_duration),
        )
        for span in raw_spans
    ]

    # Padding can push previously-separate spans into overlapping or
    # touching ranges, so merge again after padding is applied.
    merged_spans: list[list[float]] = []
    for start, end in padded_spans:
        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1][1] = max(merged_spans[-1][1], end)
        else:
            merged_spans.append([start, end])

    return [AppearanceInterval(start_time=start, end_time=end) for start, end in merged_spans]
