"""Long-running work for the desktop UI, kept free of any Tkinter import.

Scanning a 22-minute video takes minutes, so it cannot run on the thread
that draws the window. Everything here is written to run on a worker
thread and report back through plain callbacks, which means it can also
be exercised from a test without a display attached.

Two rules this module exists to enforce:

- Nothing here touches a widget. The UI turns these callbacks into
  widget updates on the main thread; see app/ui/app.py.
- The PyAV container is opened and consumed on the same thread, inside
  one `with` block. Frames stream (7d), so the container has to outlive
  the iteration rather than the call that started it.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.faces.grouper import (
    DEFAULT_CONSOLIDATION_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_MIN_GROUP_EYE_SPAN,
    DEFAULT_SIMILARITY_THRESHOLD,
    FaceIdentityGroup,
    auto_min_detections,
)
from app.main import run_identity_pipeline
from app.ui.gallery import DEFAULT_PADDING_RATIO, build_identity_gallery
from app.video.export import (
    DEFAULT_BRIDGE_GAP_SECONDS,
    DEFAULT_EXPORT_PADDING_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    export_segments,
    merge_for_export,
)
from app.video.frames import extract_frames
from app.video.loader import get_video_info, load_video
from app.video.timeline import build_appearance_intervals


class Cancelled(Exception):
    """Raised inside the worker when the user asks it to stop.

    Deliberately an exception rather than a flag checked by the pipeline:
    it unwinds through run_identity_pipeline's `finally`, so the detector
    and embedder are closed on the way out, and through export_segments'
    temporary-directory context manager, so half-encoded segments are
    cleaned up. Neither of those needed a cancellation concept added to
    it to make that work.
    """


@dataclass(frozen=True)
class ScanSettings:
    """The knobs a scan run needs, defaulted to the CLI's own defaults.

    Mirrors the `group` command's arguments so the UI and the CLI cannot
    drift into producing different results from the same video.
    """

    sample_interval: float = 0.5
    confidence_threshold: float = 0.6
    padding_ratio: float = DEFAULT_PADDING_RATIO
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD
    consolidation_threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_face_size: int = DEFAULT_MIN_FACE_SIZE
    min_group_eye_span: float = DEFAULT_MIN_GROUP_EYE_SPAN
    min_detections: int | None = None


@dataclass(frozen=True)
class ExportSettings:
    """The knobs an export run needs, defaulted to the CLI's own defaults."""

    gap_tolerance_seconds: float | None = None
    appearance_padding_seconds: float | None = None
    bridge_gap_seconds: float = DEFAULT_BRIDGE_GAP_SECONDS
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS
    export_padding_seconds: float = DEFAULT_EXPORT_PADDING_SECONDS
    video_encoder: str = "libx264"
    audio_encoder: str = "aac"
    quality: int = 20
    include_audio: bool = True


@dataclass(frozen=True)
class Person:
    """One identity, in the form the UI needs to draw and then export it."""

    index: int
    thumbnail: Image.Image
    detection_count: int
    first_seen: float
    last_seen: float
    group: FaceIdentityGroup


@dataclass(frozen=True)
class ScanResult:
    """Everything one scan produced, including what export needs later."""

    video_path: Path
    video_duration: float
    sample_interval: float
    people: list[Person] = field(default_factory=list)
    frame_count: int = 0
    detection_count: int = 0
    unassigned_count: int = 0
    min_detections: int = 0
    elapsed_seconds: float = 0.0


def _tracked_frames(frames, total_frames, cancel, report):
    """Wraps the frame stream to report progress and honour cancellation.

    Wrapping the iterator rather than passing a callback into the
    pipeline keeps app/main.py unaware that a UI exists: the pipeline
    consumes an iterator either way.
    """
    for index, item in enumerate(frames):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        timestamp = item[0]
        if report is not None:
            fraction = min(1.0, (index + 1) / total_frames) if total_frames else 0.0
            report(fraction, timestamp)
        yield item


def scan(
    video_path: Path,
    settings: ScanSettings | None = None,
    on_progress=None,
    cancel: threading.Event | None = None,
) -> ScanResult:
    """Finds every distinct person in a video.

    Args:
        video_path: The video to scan.
        settings: Detection and grouping knobs; CLI defaults when omitted.
        on_progress: Called as (fraction, timestamp_seconds) per sampled
            frame. Fraction is 0.0 when the duration is unknown and no
            estimate is possible.
        cancel: Set it to stop the scan at the next sampled frame.

    Raises:
        Cancelled: If `cancel` was set while scanning.
        VideoLoadError: If the video cannot be opened.
    """
    settings = settings or ScanSettings()
    video_path = Path(video_path)
    started = time.monotonic()

    with load_video(video_path) as container:
        duration = get_video_info(container)["duration"]
        resolved_min_detections = (
            max(1, settings.min_detections)
            if settings.min_detections is not None
            else auto_min_detections(duration, settings.sample_interval)
        )

        total_frames = int(duration / settings.sample_interval) if duration else 0
        frames = _tracked_frames(
            extract_frames(container, sample_interval=settings.sample_interval),
            total_frames,
            cancel,
            on_progress,
        )

        result = run_identity_pipeline(
            frames,
            confidence_threshold=settings.confidence_threshold,
            padding_ratio=settings.padding_ratio,
            similarity_threshold=settings.similarity_threshold,
            margin_threshold=settings.margin_threshold,
            consolidation_threshold=settings.consolidation_threshold,
            min_confidence=settings.min_confidence,
            min_face_size=settings.min_face_size,
            min_group_eye_span=settings.min_group_eye_span,
            min_detections=resolved_min_detections,
        )

    # Only knowable after the stream has been consumed, and only needed as
    # a fallback: containers that report no duration still have to yield a
    # number for interval clamping.
    if not duration:
        duration = result.last_timestamp

    gallery = build_identity_gallery(
        result.grouper.groups,
        unassigned_count=len(result.grouper.unassigned),
        padding_ratio=settings.padding_ratio,
    )

    people = [
        Person(
            index=index,
            thumbnail=Image.fromarray(card.representative_thumbnail),
            detection_count=card.detection_count,
            first_seen=card.first_seen_timestamp,
            last_seen=card.last_seen_timestamp,
            group=group,
        )
        for index, (card, group) in enumerate(zip(gallery.cards, gallery.groups))
    ]

    return ScanResult(
        video_path=video_path,
        video_duration=duration,
        sample_interval=settings.sample_interval,
        people=people,
        frame_count=result.frame_count,
        detection_count=result.total_detections,
        unassigned_count=gallery.unassigned_count,
        min_detections=resolved_min_detections,
        elapsed_seconds=time.monotonic() - started,
    )


def plan_export(
    person: Person,
    video_duration: float,
    sample_interval: float,
    settings: ExportSettings | None = None,
):
    """Works out which segments a person's reel would contain.

    Split out from `export` so the UI can tell someone what they are about
    to get -- how many cuts, how long -- before committing them to an
    encode that runs for minutes.
    """
    settings = settings or ExportSettings()

    intervals = build_appearance_intervals(
        person.group,
        video_duration=video_duration,
        sample_interval=sample_interval,
        gap_tolerance_seconds=settings.gap_tolerance_seconds,
        padding_seconds=settings.appearance_padding_seconds,
    )
    segments = merge_for_export(
        intervals,
        video_duration=video_duration,
        bridge_gap_seconds=settings.bridge_gap_seconds,
        min_segment_seconds=settings.min_segment_seconds,
        padding_seconds=settings.export_padding_seconds,
    )
    return intervals, segments


def export(
    scan_result: ScanResult,
    person: Person,
    output_path: Path,
    settings: ExportSettings | None = None,
    on_progress=None,
    cancel: threading.Event | None = None,
):
    """Cuts one person's appearances into a single reel.

    Args:
        scan_result: The scan that produced `person`, for the video path,
            duration and sampling interval the intervals were built at.
        person: Who to cut for.
        output_path: Where to write the reel.
        settings: Editorial and encoding knobs; CLI defaults when omitted.
        on_progress: Called as (fraction, cuts_done, cuts_total) after
            each segment is encoded.
        cancel: Set it to stop after the current segment finishes.

    Raises:
        Cancelled: If `cancel` was set during the encode.
        ExportError: If ffmpeg is missing, or a segment produced nothing.
    """
    settings = settings or ExportSettings()

    _, segments = plan_export(
        person,
        video_duration=scan_result.video_duration,
        sample_interval=scan_result.sample_interval,
        settings=settings,
    )

    def report(index: int, total: int, _segment) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if on_progress is not None:
            on_progress((index + 1) / total, index + 1, total)

    return export_segments(
        scan_result.video_path,
        segments,
        output_path,
        video_encoder=settings.video_encoder,
        audio_encoder=settings.audio_encoder,
        quality=settings.quality,
        include_audio=settings.include_audio,
        progress=False,
        on_segment=report,
    )
