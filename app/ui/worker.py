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

import av
from dataclasses import dataclass, field, replace
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
from app.faces.edits import (
    EditError,
    discard_groups,
    merge_groups,
    rename_group,
    split_group,
)
from app.models import MODELS, ensure_model, find_model
from app.scans import CachedScan
from app.scans import find as find_scan
from app.scans import save as save_scan
from app.ui.gallery import DEFAULT_PADDING_RATIO, build_identity_gallery
from app.modes import DEFAULT_MODE, get_mode
from app.video.cutter import cut_segments
from app.video.source import VideoSource
from app.video.export import (
    DEFAULT_BRIDGE_GAP_SECONDS,
    DEFAULT_EXPORT_PADDING_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    merge_for_export,
)
from app.video.frames import extract_frames
from app.video.loader import get_video_info, use_threaded_decoding
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

    @classmethod
    def for_mode(cls, mode: str = DEFAULT_MODE, sample_interval: float = 0.5) -> "ScanSettings":
        """The settings a scan in this mode should run with.

        The field defaults above are live action's numbers. Building
        settings with only `mode=` changed kept those, so the window scanned
        animation with a live-action similarity floor of 0.35 where
        animation's own is 0.75. On the 7-minute animation sample that found
        2 people where the mode's own thresholds find 6, and 306 faces where
        they find 344.

        Resolved the same way the command line resolves an unset flag
        (`resolve_mode_settings` in app/__main__.py), so the two front ends
        run the same scan -- and share its kept copy, and the names in it.
        """
        spec = get_mode(mode)
        return cls(
            sample_interval=sample_interval,
            mode=spec.id,
            confidence_threshold=spec.detection.confidence_threshold,
            similarity_threshold=spec.grouping.similarity_threshold,
            consolidation_threshold=spec.grouping.consolidation_threshold,
            min_confidence=spec.detection.min_confidence,
            min_face_size=spec.detection.min_face_size,
            min_group_eye_span=spec.grouping.min_group_eye_span,
        )


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
    name: str | None = None

    @property
    def label(self) -> str:
        """What to call this person: their name, or their position."""
        return self.name or f"Person #{self.index + 1}"


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
    # True when this came back from the scan cache rather than being
    # computed. The window says so: an instant result that looks like a
    # fresh scan invites the suspicion that it did not really look.
    reused: bool = False
    # How long the original scan took, when this was reused.
    original_seconds: float = 0.0

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
    use_cache: bool = True,
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
        use_cache: Whether a kept scan of the same video under the same
            settings may be reused, and this one kept (app/scans.py).

    Raises:
        Cancelled: If `cancel` was set while scanning or downloading.
        ModelDownloadError: If a model cannot be fetched or fails its
            checksum.
        VideoLoadError: If the video cannot be opened.
    """
    settings = settings or ScanSettings()
    video_path = Path(video_path)
    started = time.monotonic()

    key, found = find_scan(
        video_path,
        sample_interval=settings.sample_interval,
        confidence_threshold=settings.confidence_threshold,
        padding_ratio=settings.padding_ratio,
        similarity_threshold=settings.similarity_threshold,
        margin_threshold=settings.margin_threshold,
        consolidation_threshold=settings.consolidation_threshold,
        min_confidence=settings.min_confidence,
        min_face_size=settings.min_face_size,
        min_group_eye_span=settings.min_group_eye_span,
        mode=settings.mode,
        forbid_cooccurring=settings.forbid_cooccurring,
        cooccurrence_similarity_ceiling=settings.cooccurrence_similarity_ceiling,
        min_detections=settings.min_detections,
    )
    kept = found if use_cache else None

    # Consulted before the models are fetched, not after. A reused scan
    # detects and embeds nothing, so on a machine that has never run one
    # this is also the difference between opening a known video instantly
    # and waiting on a 174 MB download to do no work with.
    if kept is None:
        fetch_models(on_progress=on_download, cancel=cancel)

    # Opened once, held for the life of the result. The descriptor is what
    # lets the export survive the user moving this file while they look
    # through the gallery (app/video/source.py).
    source = VideoSource(video_path)

    if kept is not None:
        scan = kept
        duration = scan.video_duration or scan.last_timestamp
        resolved_min_detections = scan.min_detections
        if on_progress is not None:
            on_progress(1.0, duration or 0.0)
    else:
        try:
            result, duration, resolved_min_detections = _scan_footage(
                source, settings, cancel, on_progress
            )
        except BaseException:
            source.close()
            raise

        scan = CachedScan(
            groups=result.grouper.groups,
            unassigned_count=len(result.grouper.unassigned),
            total_detections=result.total_detections,
            track_count=result.track_count,
            frame_count=result.frame_count,
            last_timestamp=result.last_timestamp,
            embedding_time=result.embedding_time,
            grouping_time=result.grouping_time,
            video_duration=duration,
            min_detections=resolved_min_detections,
            created_at=time.time(),
            scan_seconds=time.monotonic() - started,
        )
        # A cancelled scan raises out of _scan_footage above, so anything
        # reaching here ran to the end of the footage and is complete.
        if use_cache and scan.frame_count > 0:
            save_scan(key, scan)

    gallery = build_identity_gallery(
        scan.groups,
        unassigned_count=scan.unassigned_count,
        padding_ratio=settings.padding_ratio,
    )

    people = _people_from(gallery)

    return ScanResult(
        video_path=video_path,
        video_duration=duration,
        sample_interval=settings.sample_interval,
        people=people,
        frame_count=scan.frame_count,
        detection_count=scan.total_detections,
        unassigned_count=gallery.unassigned_count,
        min_detections=resolved_min_detections,
        elapsed_seconds=time.monotonic() - started,
        source=source,
        reused=kept is not None,
        original_seconds=scan.scan_seconds,
    )


def _people_from(gallery) -> list["Person"]:
    """The person cards a gallery describes, in its own order.

    Shared by the scan and by every edit: a merged or split gallery has to
    produce cards the same way an unedited one does, or the two would
    disagree about what a person card is.
    """
    return [
        Person(
            index=index,
            thumbnail=Image.fromarray(card.representative_thumbnail),
            detection_count=card.detection_count,
            first_seen=card.first_seen_timestamp,
            last_seen=card.last_seen_timestamp,
            group=group,
            name=card.name,
        )
        for index, (card, group) in enumerate(zip(gallery.cards, gallery.groups))
    ]


def apply_edit(
    scan_result: "ScanResult",
    settings: ScanSettings,
    operation: str,
    indexes: list[int],
    tracks: list[int] | None = None,
    name: str | None = None,
) -> "ScanResult":
    """Corrects the identities, and keeps the correction.

    Grouping is fitted to real footage and still splits one actor across
    two cards, or keeps a logo as a convincing phantom person. Until this
    existed the only recourse was re-tuning thresholds and rescanning --
    minutes of work to fix something visible at a glance.

    The corrected groups are written back to the scan cache under the same
    key, so the fix outlives the window rather than having to be repeated
    every time the video is opened. `--rescan` still discards them, which
    is the way back to what the clustering actually said.

    Returns a new ScanResult sharing this one's footage handle; the caller
    keeps owning that handle and should close it once.

    Raises:
        EditError: If the edit does not describe something that can be done.
    """
    groups = [person.group for person in scan_result.people]

    if operation == "merge":
        edited = merge_groups(groups, indexes)
    elif operation == "split":
        (index,) = indexes
        edited = split_group(groups, index, tracks or [])
    elif operation == "discard":
        edited = discard_groups(groups, indexes)
    elif operation == "rename":
        (index,) = indexes
        edited = rename_group(groups, index, name or "")
    else:
        raise EditError(f"Unknown edit: {operation!r}")

    gallery = build_identity_gallery(
        edited,
        unassigned_count=scan_result.unassigned_count,
        padding_ratio=settings.padding_ratio,
    )
    people = _people_from(gallery)

    updated = replace(
        scan_result,
        people=people,
        # The detections did not change -- they were only regrouped -- but
        # discarding a card removes its own from the gallery, so this is
        # recounted rather than carried over.
        detection_count=sum(person.detection_count for person in people),
        reused=False,
    )

    _keep_edited(updated, settings, gallery)
    return updated


def _keep_edited(scan_result: "ScanResult", settings: ScanSettings, gallery) -> None:
    """Writes corrected identities back over the scan they came from.

    Best effort: an edit the user can see must not fail because the cache
    could not be written. It would simply have to be made again.
    """
    key, existing = find_scan(
        scan_result.video_path,
        sample_interval=settings.sample_interval,
        confidence_threshold=settings.confidence_threshold,
        padding_ratio=settings.padding_ratio,
        similarity_threshold=settings.similarity_threshold,
        margin_threshold=settings.margin_threshold,
        consolidation_threshold=settings.consolidation_threshold,
        min_confidence=settings.min_confidence,
        min_face_size=settings.min_face_size,
        min_group_eye_span=settings.min_group_eye_span,
        mode=settings.mode,
        forbid_cooccurring=settings.forbid_cooccurring,
        cooccurrence_similarity_ceiling=settings.cooccurrence_similarity_ceiling,
        min_detections=settings.min_detections,
    )
    if existing is None:
        return
    try:
        save_scan(
            key,
            replace(
                existing,
                groups=gallery.groups,
                edited=True,
            ),
        )
    except OSError:
        pass


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


def combined_group(people: list[Person]) -> FaceIdentityGroup:
    """One group holding every selected person's detections.

    Two leads' scenes are the union of their appearances, and a union of
    timestamps is all `build_appearance_intervals` reads. So rather than
    building each person's intervals and merging the results, the
    observations are pooled and the existing stage runs once over the
    combined timeline.

    That is not merely less code, it is the more correct answer. Gap
    tolerance and padding then apply to the reel as it will be watched:
    when one lead leaves a scene and the other arrives a second later,
    the pooled timeline sees one continuous appearance, while merging two
    separately-built interval lists would have already cut it in two and
    padded both halves.

    The group is a carrier, not an identity -- it has no centroid and no
    representative, because a group of two people has neither.
    """
    if not people:
        raise ValueError("no people to combine")
    if len(people) == 1:
        return people[0].group

    observations = [
        observation for person in people for observation in person.group.observations
    ]
    return FaceIdentityGroup(group_id=-1, observations=observations)


def plan_export(
    person: Person | list[Person],
    video_duration: float,
    sample_interval: float,
    settings: ExportSettings | None = None,
):
    """Works out which segments a reel would contain.

    Split out from `export` so the UI can tell someone what they are about
    to get -- how many cuts, how long -- before committing them to an
    encode that runs for minutes.

    Takes one person or several; several gives the reel of every scene any
    of them is in.
    """
    settings = settings or ExportSettings()
    people = person if isinstance(person, list) else [person]

    intervals = build_appearance_intervals(
        combined_group(people),
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


# How many frames a preview filmstrip shows. Enough to tell whether the
# reel opens on the right person and holds together, few enough that
# building it is a fraction of a second rather than a wait.
PREVIEW_FRAMES = 6
PREVIEW_WIDTH = 192


def preview_frames(
    scan_result: "ScanResult",
    people: "Person | list[Person]",
    limit: int = PREVIEW_FRAMES,
    width: int = PREVIEW_WIDTH,
    settings: ExportSettings | None = None,
) -> list[tuple[float, Image.Image]]:
    """Frames from the reel that would be cut, spread across its length.

    Selecting a card said "14 cuts, about 4:31" and then asked the user to
    commit minutes of encoding to it on faith. This shows what is in it.

    Seeking is the right tool here and nowhere else in this project. It
    loses badly for sampling, where a frame is wanted every 0.5s and each
    seek decodes the whole GOP in front of it (Instructions 18); for six
    frames spread over twenty minutes it decodes six short GOPs instead of
    the entire video, which is the case the same measurement says it wins.

    Frames come from the middle of each chosen segment rather than its
    start, because a cut's first frame often lands mid-transition and
    shows a face nobody would recognise.

    Returns (timestamp, image) pairs in chronological order, empty when
    there is nothing to cut or the footage can no longer be read. It is a
    preview: failing to draw one must never stop an export that would
    otherwise work.
    """
    chosen = people if isinstance(people, list) else [people]
    if not chosen or scan_result.source is None:
        return []

    _, segments = plan_export(
        chosen,
        video_duration=scan_result.video_duration,
        sample_interval=scan_result.sample_interval,
        settings=settings,
    )
    if not segments:
        return []

    # Spread across the reel rather than taking the first few, so a
    # 100-cut reel is represented by its whole length.
    if len(segments) <= limit:
        picked = list(segments)
    else:
        step = (len(segments) - 1) / (limit - 1) if limit > 1 else 0
        picked = [segments[round(position * step)] for position in range(limit)]

    wanted = [
        (segment.start_time + segment.end_time) / 2.0 for segment in picked
    ]

    try:
        return _frames_at(scan_result, wanted, width)
    except Exception:
        # Any decode failure at all: a moved file, a truncated video, a
        # codec that will not seek. The preview is a convenience.
        return []


def _preview_container(scan_result: "ScanResult"):
    """Opens the footage for a preview, without disturbing an export.

    Deliberately by path rather than through `scan_result.source`. A
    VideoSource hands out readers by duplicating one descriptor, and
    duplicated descriptors share a file offset -- so two readers seek each
    other sideways and both fail with "invalid data". Everything else in
    this app reads the footage one reader at a time; a preview drawn while
    an export is running is the first thing that does not.

    So the preview reads its own open file and leaves the shared
    descriptor to the export, which must have it: the descriptor is what
    lets an export survive the video being moved mid-session, and reading
    by path cannot promise that. The size is checked first for the same
    reason `relocate` checks it -- a different file at the old path would
    show frames from footage the reel is not made of.

    Returns None when the path cannot stand in, and the preview is simply
    not drawn.
    """
    source = scan_result.source
    if source is None:
        return None
    path = source.path
    try:
        if not path.is_file() or path.stat().st_size != source.size:
            return None
    except OSError:
        return None

    container = av.open(str(path))
    use_threaded_decoding(container)
    return container


def _frames_at(scan_result: "ScanResult", timestamps: list[float], width: int):
    """Decodes one frame at each timestamp, by seeking to each in turn."""
    frames = []
    container = _preview_container(scan_result)
    if container is None:
        return []
    with container:
        stream = next(
            (s for s in container.streams if s.type == "video"), None
        )
        if stream is None:
            return []
        time_base = stream.time_base

        for wanted in timestamps:
            container.seek(int(wanted / time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                # The first frame at or after the target. Seeking lands on
                # the keyframe before it, so this decodes forward.
                if frame.time + 1e-6 < wanted:
                    continue
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                height = max(1, round(image.height * width / image.width))
                frames.append(
                    (float(frame.time), image.resize((width, height)))
                )
                break
    return frames


# How many tracks a split picker shows at once. A person on a 22-minute
# episode has a median of 21 tracks and can have hundreds, and a picker
# nobody can read is not a correction tool.
TRACK_PREVIEWS = 24


def track_previews(
    scan_result: "ScanResult",
    person: "Person",
    limit: int = TRACK_PREVIEWS,
    width: int = PREVIEW_WIDTH,
) -> list[tuple[int, float, Image.Image]]:
    """One picture per track in a person's card, for choosing what to split.

    A track is the only unit a group can honestly be split along, so this
    is what a split picker has to show. The pictures are read back from the
    footage rather than stored: keeping a crop for every observation would
    multiply an episode's cached scan from 6 MB to hundreds, to serve a
    correction that is made rarely and looked at once.

    Longest tracks first, because a mistakenly merged card is two
    substantial runs of somebody, not a scattering of single frames -- and
    because with hundreds of tracks the short ones are the ones nobody can
    judge from a thumbnail anyway.

    Returns (track index, timestamp, image), where the track index counts
    into `person.group.tracks`. Empty if the footage cannot be read.
    """
    if scan_result.source is None:
        return []

    tracks = person.group.tracks
    if len(tracks) < 2:
        return []

    ranked = sorted(
        range(len(tracks)), key=lambda i: (-len(tracks[i]), tracks[i][0].source_timestamp)
    )[:limit]
    # Shown in time order, whichever were picked as the longest.
    ranked.sort(key=lambda i: tracks[i][0].source_timestamp)

    # The middle observation of each track: the ends of a track are where
    # a face is entering or leaving, and the middle is where it is seen.
    wanted = [tracks[i][len(tracks[i]) // 2] for i in ranked]

    try:
        frames = _frames_at(
            scan_result,
            [observation.source_timestamp for observation in wanted],
            width,
        )
    except Exception:
        return []

    return [
        (track_index, timestamp, image)
        for track_index, (timestamp, image) in zip(ranked, frames)
    ]


def export(
    scan_result: ScanResult,
    person: Person | list[Person],
    output_path: Path,
    settings: ExportSettings | None = None,
    on_progress=None,
    cancel: threading.Event | None = None,
):
    """Cuts one or more people's appearances into a single reel.

    Args:
        scan_result: The scan that produced `person`, for the video path,
            duration and sampling interval the intervals were built at.
        person: Who to cut for. Several gives every scene any of them is
            in, on one combined timeline.
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
