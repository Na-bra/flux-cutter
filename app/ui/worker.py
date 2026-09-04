"""Long-running work for the desktop UI, kept free of any Tkinter import.

Scanning a 22-minute video takes minutes, so it cannot run on the thread
that draws the window. Everything here is written to run on a worker
thread and report back through plain callbacks, which means it can also
be exercised from a test without a display attached.

Two rules this module exists to enforce:

- Nothing here touches a widget. The UI turns these callbacks into
  updates on the main thread; see app/ui/web.py.
- The PyAV container is opened and consumed on the same thread, inside
  one `with` block. Frames stream (7d), so the container has to outlive
  the iteration rather than the call that started it.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.faces.grouper import (
    DEFAULT_COOCCURRENCE_SIMILARITY_CEILING,
    DEFAULT_FORBID_COOCCURRING,
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
from app.models import MODELS, ensure_model, find_model
from app.ui.gallery import DEFAULT_PADDING_RATIO, build_identity_gallery
from app.modes import DEFAULT_MODE
from app.video.cutter import cut_segments
from app.video.source import VideoSource
from app.video.export import (
    DEFAULT_BRIDGE_GAP_SECONDS,
    DEFAULT_EXPORT_PADDING_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    merge_for_export,
)
from app.video.frames import extract_frames
from app.video.loader import get_video_info
from app.video.timeline import build_appearance_intervals


class Cancelled(Exception):
    """Raised inside the worker when the user asks it to stop.

    Deliberately an exception rather than a flag checked by the pipeline:
    it unwinds through run_identity_pipeline's `finally`, so the detector
    and embedder are closed on the way out, and through cut_segments'
    open output container, which is closed in its own `finally`. Neither
    of those needed a cancellation concept added to it to make that work.
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
    # Which pipeline runs. Chosen by the user, never inferred.
    mode: str = DEFAULT_MODE
    forbid_cooccurring: bool = DEFAULT_FORBID_COOCCURRING
    cooccurrence_similarity_ceiling: float = DEFAULT_COOCCURRENCE_SIMILARITY_CEILING
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


# Video encoders worth offering, best first. Hardware encoders are listed
# ahead of libx264 because the gap is not subtle -- videotoolbox cut the
# same 12s reel in 4.7s against libx264's 23.9s -- and each is specific to
# hardware that may not be present: videotoolbox is Apple-only, nvenc is
# NVIDIA, qsv is Intel, amf is AMD. Which of them exist is a question about
# the machine the app is running on, not about the platform it was built
# for, so the list is filtered by asking rather than by guessing from
# sys.platform. libx264 is last and unconditional: it is the one that is
# always there.
ENCODER_PREFERENCE = (
    "h264_videotoolbox",
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
    "libx264",
)


def _encoder_works(name: str) -> bool:
    """Whether this machine can really encode with `name`.

    Constructing the codec is not enough. PyAV's Windows wheel compiles in
    h264_nvenc, h264_qsv and h264_amf unconditionally, so on a PC with no
    NVIDIA card `Codec("h264_nvenc", "w")` still succeeds and the failure
    arrives minutes later, in the middle of an export. Actually opening an
    encoder and pushing one frame through it costs milliseconds and asks
    the question that matters: is the hardware there.
    """
    import av
    import numpy as np

    try:
        with av.open("/dev/null" if os.name != "nt" else "NUL", mode="w", format="mp4") as sink:
            stream = sink.add_stream(name, rate=30)
            stream.width, stream.height, stream.pix_fmt = 160, 128, "yuv420p"
            frame = av.VideoFrame.from_ndarray(
                np.zeros((128, 160, 3), dtype=np.uint8), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = 0
            stream.encode(frame)
            stream.encode()
    except Exception:
        return False
    return True


def available_encoders() -> list[str]:
    """Which of the encoders we offer this machine can actually run.

    Asks PyAV, whose FFmpeg is bundled in the wheel and therefore travels
    with the app rather than having to be installed alongside it.
    """
    found = [name for name in ENCODER_PREFERENCE if _encoder_works(name)]
    return found or ["libx264"]


def default_encoder() -> str:
    """The best encoder this machine actually has."""
    return available_encoders()[0]


# The two encoders take quality on scales that do not merely differ but run
# in opposite directions: -crf is 0-51 and lower is better, -q:v is 0-100
# and higher is better. Handing both the same number silently produced a
# 1.7 Mbps videotoolbox file where libx264 gave 13.9 Mbps on the same clip,
# so callers pick a named level and this translates it. It lives here rather
# than next to the dropdown because it is a decision about encoding, and
# because here it can be tested without a display.
QUALITY_LEVELS = {
    "Standard": {"libx264": 26, "h264_videotoolbox": 45},
    "High": {"libx264": 22, "h264_videotoolbox": 55},
    "Maximum": {"libx264": 18, "h264_videotoolbox": 70},
}
DEFAULT_QUALITY_LEVEL = "High"


def quality_for(encoder: str, level: str) -> int:
    """Translates a named quality level into what this encoder expects."""
    settings = QUALITY_LEVELS.get(level, QUALITY_LEVELS[DEFAULT_QUALITY_LEVEL])
    # An encoder we have no mapping for is far likelier to be crf-based than
    # videotoolbox-like, videotoolbox being the one Apple special case.
    return settings.get(encoder, settings["libx264"])


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
    """Everything one scan produced, including what export needs later.

    `source` is the live handle on the footage and `video_path` is where it
    was when the scan ran. They come apart the moment the user moves the
    file: export reads through `source`, while anything cosmetic -- naming
    the output after the video, say -- reads `video_path`. Callers that
    own a ScanResult own the descriptor inside it, and should close it.
    """

    video_path: Path
    video_duration: float
    sample_interval: float
    people: list[Person] = field(default_factory=list)
    frame_count: int = 0
    detection_count: int = 0
    unassigned_count: int = 0
    min_detections: int = 0
    elapsed_seconds: float = 0.0
    source: VideoSource | None = None

    def close(self) -> None:
        """Releases the footage handle. Safe to call more than once."""
        if self.source is not None:
            self.source.close()


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


def missing_models() -> list:
    """Which models still have to be fetched before a scan can run."""
    return [spec for spec in MODELS.values() if find_model(spec) is None]


def fetch_models(on_progress=None, cancel: threading.Event | None = None) -> None:
    """Downloads any model that is not on disk yet.

    Done here, before the pipeline starts, rather than left to the detector
    and embedder to trigger on construction. Those would fetch it several
    frames deep with nowhere to report to but stdout, which a window does
    not have; pulling it forward means the download is a visible phase with
    a progress bar of its own.

    Args:
        on_progress: Called as (description, fraction, done, total).
        cancel: Checked as the bytes arrive, so a 166 MB download can be
            stopped. The partial file is discarded, never left to look
            like a finished one.
    """
    for spec in missing_models():
        def report(fraction: float, done: int, total: int, spec=spec) -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if on_progress is not None:
                on_progress(spec.description, fraction, done, total)

        ensure_model(spec, on_progress=report)


def scan(
    video_path: Path,
    settings: ScanSettings | None = None,
    on_progress=None,
    cancel: threading.Event | None = None,
    on_download=None,
) -> ScanResult:
    """Finds every distinct person in a video.

    Args:
        video_path: The video to scan.
        settings: Detection and grouping knobs; CLI defaults when omitted.
        on_progress: Called as (fraction, timestamp_seconds) per sampled
            frame. Fraction is 0.0 when the duration is unknown and no
            estimate is possible.
        cancel: Set it to stop the scan at the next sampled frame, or
            during a model download.
        on_download: Called as (description, fraction, done, total) while a
            model is being fetched on first run.

    Raises:
        Cancelled: If `cancel` was set while scanning or downloading.
        ModelDownloadError: If a model cannot be fetched or fails its
            checksum.
        VideoLoadError: If the video cannot be opened.
    """
    settings = settings or ScanSettings()

    # Before the video is even opened: a first run should not decode two
    # minutes of frames and only then discover it has no model to embed
    # them with.
    fetch_models(on_progress=on_download, cancel=cancel)

    video_path = Path(video_path)
    started = time.monotonic()

    # Opened once, held for the life of the result. The descriptor is what
    # lets the export survive the user moving this file while they look
    # through the gallery (app/video/source.py).
    source = VideoSource(video_path)

    try:
        result, duration, resolved_min_detections = _scan_footage(
            source, settings, cancel, on_progress
        )
    except BaseException:
        source.close()
        raise

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
        source=source,
    )


def _scan_footage(source, settings, cancel, on_progress):
    """Runs the pipeline over the footage, returning it with what it needed.

    Split out of `scan` only so the container is closed and the descriptor
    released by one `try` rather than two nested ones.
    """
    with source.open() as container:
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
            forbid_cooccurring=settings.forbid_cooccurring,
            cooccurrence_similarity_ceiling=settings.cooccurrence_similarity_ceiling,
            mode=settings.mode,
            min_detections=resolved_min_detections,
        )

    # Only knowable after the stream has been consumed, and only needed as
    # a fallback: containers that report no duration still have to yield a
    # number for interval clamping.
    if not duration:
        duration = result.last_timestamp

    return result, duration, resolved_min_detections


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
        CutterError: If the source cannot be read or the segments are
            unusable. A source whose file has moved raises this only once
            both the descriptor and the path have failed -- see
            ScanResult.source.
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

    return cut_segments(
        # The held descriptor when there is one, so a video moved since the
        # scan still cuts; the recorded path otherwise.
        scan_result.source or scan_result.video_path,
        segments,
        output_path,
        video_encoder=settings.video_encoder,
        audio_encoder=settings.audio_encoder,
        quality=settings.quality,
        include_audio=settings.include_audio,
        on_segment=report,
    )
