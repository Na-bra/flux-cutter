import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.video.timeline import AppearanceInterval, format_timestamp

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


def _merge_close(
    spans: list[tuple[float, float]], bridge_gap_seconds: float
) -> list[tuple[float, float]]:
    """Joins spans that overlap or sit within bridge_gap_seconds of each other."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= bridge_gap_seconds:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


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
    spans = _merge_close(padded, bridge_gap_seconds)

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
    final_spans = _merge_close(grown, bridge_gap_seconds)

    return [AppearanceInterval(start_time=start, end_time=end) for start, end in final_spans]


class ExportError(Exception):
    """Raised when a clip cannot be cut or the segments cannot be joined."""


@dataclass(frozen=True)
class ExportResult:
    """What one export produced."""

    output_path: Path
    segment_count: int
    exported_seconds: float
    encode_seconds: float


def _require_ffmpeg() -> str:
    """Locates the ffmpeg binary, or explains what to install."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise ExportError(
            "ffmpeg was not found on PATH. Clip export shells out to ffmpeg for "
            "cutting and concatenation; install it (e.g. 'brew install ffmpeg') "
            "and try again."
        )
    return binary


def _run(command: list[str], description: str) -> None:
    """Runs one ffmpeg invocation, surfacing its own error text on failure."""
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-8:])
        raise ExportError(f"{description} failed (ffmpeg exit {completed.returncode}):\n{tail}")


def export_segments(
    video_path: Path,
    segments: list[AppearanceInterval],
    output_path: Path,
    video_encoder: str = "libx264",
    audio_encoder: str = "aac",
    quality: int = 20,
    include_audio: bool = True,
    progress: bool = True,
) -> ExportResult:
    """
    Cuts each segment out of the source and joins them into one file.

    Segments are re-encoded rather than stream-copied, which is not an
    efficiency oversight but the only way to get the cut where it was asked
    for. A stream copy can only begin at a keyframe, and on the 22-minute test
    footage keyframes are a median 2.67s apart and up to 7.84s apart, while the
    median segment is 3.0s long -- so a copied cut would routinely open several
    seconds early, on somebody else's face. (The 23s test clip is nearly
    all-intra, keyframes 0.03s apart, and hides this completely: an approach
    validated only there would look perfect and fail on real video.)

    Joining is a second pass over the encoded segments using the concat
    demuxer, which *is* a stream copy and safely so, because every segment was
    just written with identical codec parameters.

    Args:
        video_path: The source video.
        segments: Non-overlapping segments in chronological order, as returned
            by merge_for_export.
        output_path: Where to write the joined result.
        video_encoder: ffmpeg video encoder. 'h264_videotoolbox' is markedly
            faster on Apple silicon; 'libx264' is the portable default.
        audio_encoder: ffmpeg audio encoder, used only when include_audio.
        quality: Constant-quality level (-crf for libx264, -q:v for
            videotoolbox). Lower is better quality and a bigger file.
        include_audio: Whether to carry the source audio through.
        progress: Print per-segment progress, since a long reel takes a while.

    Returns:
        An ExportResult describing what was written.
    """
    if not segments:
        raise ExportError("No segments to export.")
    for earlier, later in zip(segments, segments[1:]):
        if later.start_time < earlier.end_time:
            raise ExportError(
                "Segments overlap; pass them through merge_for_export first so the "
                "joined output does not repeat footage."
            )

    binary = _require_ffmpeg()
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_flag = "-q:v" if "videotoolbox" in video_encoder else "-crf"
    started = time.monotonic()
    exported_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix="fluxcutter-export-") as workspace:
        workspace_path = Path(workspace)
        segment_paths: list[Path] = []

        for index, segment in enumerate(segments):
            duration = segment.end_time - segment.start_time
            segment_path = workspace_path / f"segment_{index:05d}.mp4"

            command = [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                # -ss ahead of -i seeks by index first and then decodes forward
                # to the exact frame, so this stays accurate without scanning
                # the whole file for every one of what may be a hundred cuts.
                "-ss",
                f"{segment.start_time:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-c:v",
                video_encoder,
                quality_flag,
                str(quality),
                # Concatenation later assumes every segment shares a timebase.
                "-video_track_timescale",
                "90000",
            ]
            command += (["-c:a", audio_encoder, "-ac", "2"] if include_audio else ["-an"])
            command += ["-y", str(segment_path)]

            _run(command, f"Cutting segment {index + 1}/{len(segments)}")
            if not segment_path.exists() or segment_path.stat().st_size == 0:
                raise ExportError(
                    f"Segment {index + 1} produced no output; the requested range "
                    f"({segment.start_time:.2f}s-{segment.end_time:.2f}s) may lie "
                    "outside the video."
                )

            segment_paths.append(segment_path)
            exported_seconds += duration
            if progress:
                print(
                    f"  cut {index + 1}/{len(segments)}  "
                    f"{format_timestamp(segment.start_time)} -> "
                    f"{format_timestamp(segment.end_time)}  ({duration:.2f}s)"
                )

        listing = workspace_path / "segments.txt"
        listing.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
            encoding="utf-8",
        )

        _run(
            [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-y",
                str(output_path),
            ],
            "Joining segments",
        )

    return ExportResult(
        output_path=output_path,
        segment_count=len(segments),
        exported_seconds=exported_seconds,
        encode_seconds=time.monotonic() - started,
    )
