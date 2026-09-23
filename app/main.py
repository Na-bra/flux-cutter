import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.faces.detector import FaceDetection, FaceDetector
from app.faces.embedder import FaceEmbedder
from app.faces.grouper import (
    DEFAULT_COOCCURRENCE_SIMILARITY_CEILING,
    DEFAULT_FORBID_COOCCURRING,
    FaceIdentityGroup,
    FaceObservation,
    IdentityGrouper,
    auto_min_detections,
)
from app.faces.reference import (
    ReferenceError,
    ReferenceFace,
    match_reference,
    reference_from_groups,
)
from app.faces.tracker import FaceTracker
from app.modes import DEFAULT_MODE, get_mode
from app.scans import CachedScan
from app.scans import find as find_scan
from app.scans import save as save_scan
from app.ui.gallery import (
    build_face_gallery,
    build_identity_gallery,
    crop_face,
    format_person_card,
    format_selected_item,
    save_gallery_montage,
    save_identity_gallery_montage,
)
from app.video.cutter import (
    Clip,
    CutterError,
    cut_clips,
    cut_segments,
    probe_clip,
)
from app.video.export import merge_for_export
from app.video.repeats import find_repeats, fingerprint_kept, without_repeats
from app.video.frames import extract_frames
from app.video.loader import (
    SUPPORTED_EXTENSIONS,
    VideoLoadError,
    get_video_info,
    load_video,
)
from app.video.timeline import build_appearance_intervals, format_timestamp


def run_face_detection(container, output_dir: Path, sample_interval: float):
    """Runs detection and saves annotated frames."""
    print(f"Extracting frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)

    detector = FaceDetector()
    total_faces = 0
    processed_frames = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotated frames to: {output_dir.resolve()}")

    start_time = time.monotonic()

    for i, (timestamp, frame_data) in enumerate(frames):
        detections = detector.detect(frame_data)
        processed_frames += 1
        if not detections:
            continue

        total_faces += len(detections)
        img = Image.fromarray(frame_data)
        draw = ImageDraw.Draw(img)

        for face in detections:
            box = (face.box.x_min, face.box.y_min, face.box.x_max, face.box.y_max)
            draw.rectangle(box, outline="red", width=5)

        output_path = output_dir / f"frame_{i:04d}_ts_{timestamp:.2f}.jpg"
        img.save(output_path)

    end_time = time.monotonic()
    detector.close()

    if processed_frames == 0:
        print("No frames extracted.")
        return

    duration = end_time - start_time
    fps = processed_frames / duration if duration > 0 else 0

    print("\n--- Detection Report ---")
    print(f"Frames processed: {processed_frames}")
    print(f"Total faces detected: {total_faces}")
    print(f"Total processing time: {duration:.2f} seconds")
    print(f"Processing speed: {fps:.2f} frames/sec")
    print("--- End Report ---\n")


def run_face_gallery(
    container,
    output_dir: Path,
    sample_interval: float,
    confidence_threshold: float,
    max_items: int,
    padding_ratio: float,
    select_index: int | None,
):
    """Build and save a face gallery montage from sampled video frames."""
    print(f"Sampling frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)

    detector = FaceDetector(confidence_threshold=confidence_threshold)
    detection_records: list[tuple[float, np.ndarray, list]] = []
    total_detections = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving gallery output to: {output_dir.resolve()}")

    start_time = time.monotonic()

    frame_count = 0
    for timestamp, frame_data in frames:
        frame_count += 1
        detections = detector.detect(frame_data)
        total_detections += len(detections)
        detection_records.append((timestamp, frame_data, detections))

    if frame_count == 0:
        detector.close()
        print("No frames extracted.")
        return

    gallery = build_face_gallery(
        detection_records,
        padding_ratio=padding_ratio,
        max_items=max_items,
    )
    montage_path = output_dir / "face-gallery.jpg"
    save_gallery_montage(gallery.items, montage_path)

    end_time = time.monotonic()
    detector.close()

    duration = end_time - start_time
    fps = frame_count / duration if duration > 0 else 0

    print("\n--- Gallery Report ---")
    print(f"Frames processed: {frame_count}")
    print(f"Detections received: {total_detections}")
    print(f"Gallery candidates: {gallery.candidate_count}")
    print(f"Gallery items: {len(gallery.items)}")
    print(f"Total processing time: {duration:.2f} seconds")
    print(f"Processing speed: {fps:.2f} frames/sec")
    print(f"Gallery montage: {montage_path}")
    print("--- End Report ---\n")

    if select_index is not None:
        selected_item = gallery.select(select_index)
        print(format_selected_item(selected_item, index=select_index))


def _resolve_min_detections(
    requested: int | None, duration_seconds: float | None, sample_interval: float
) -> int:
    """Settles the minimum-detections cutoff and says which way it was decided.

    An explicit value always wins. Otherwise it is derived from the video's
    runtime and the sampling interval, and reported, because a silently
    applied filter that hides identities is the kind of thing someone should
    be told about rather than left to discover from a short gallery.
    """
    if requested is not None:
        print(f"Minimum detections per identity: {max(1, requested)} (set explicitly).")
        return max(1, requested)

    resolved = auto_min_detections(duration_seconds, sample_interval)
    if duration_seconds:
        print(
            f"Minimum detections per identity: {resolved} "
            f"(~{resolved * sample_interval:.1f}s of screen time, from a "
            f"{duration_seconds:.0f}s video at a {sample_interval}s interval)."
        )
    else:
        print(
            f"Minimum detections per identity: {resolved} "
            f"(video duration unavailable; using the floor only)."
        )
    return resolved


def _selection_name(indexes: list[int], groups=None) -> str:
    """How a chosen person, or several, is named in a report.

    A named identity is reported by name. With nobody named this stays the
    compact "People #1 and #2" rather than repeating the word for each.
    """
    named = [
        (groups[index].name if groups and groups[index].name else None)
        for index in indexes
    ]
    if any(named):
        labels = [
            name or f"Person #{index + 1}" for name, index in zip(named, indexes)
        ]
    else:
        numbers = [f"#{index + 1}" for index in indexes]
        if len(numbers) == 1:
            return f"Person {numbers[0]}"
        if len(numbers) == 2:
            return f"People {numbers[0]} and {numbers[1]}"
        return f"People {', '.join(numbers[:-1])} and {numbers[-1]}"

    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def combined_group(groups: list[FaceIdentityGroup]) -> FaceIdentityGroup:
    """One group holding every chosen person's detections.

    A reel of two people is the union of their appearances, and a union of
    timestamps is all `build_appearance_intervals` reads -- so the
    observations are pooled and the existing stage runs once over the
    combined timeline. That is also the more correct answer than merging
    two separately-built interval lists: when one lead leaves a scene and
    the other arrives a second later, the pooled timeline sees one
    continuous appearance rather than two padded halves.

    The result is a carrier, not an identity. It has no centroid and no
    representative, because a group of two people has neither.
    """
    if not groups:
        raise ValueError("no groups to combine")
    if len(groups) == 1:
        return groups[0]
    return FaceIdentityGroup(
        group_id=-1,
        observations=[
            observation for group in groups for observation in group.observations
        ],
    )


class SelectionError(Exception):
    """Raised when the run cannot tell which person it is about.

    Deliberately raised rather than exited on: the single-video commands
    turn this into a failed command, while a batch turns it into one
    skipped episode and carries on to the next.
    """


def _resolve_selection(
    identity_gallery,
    select_index: int | list[int] | None,
    reference: ReferenceFace | None,
    reference_threshold: float | None = None,
    mode: str = DEFAULT_MODE,
    select_name: str | None = None,
) -> list[int]:
    """Which person card the run is about, however the user said it.

    An index, a name and a reference face are ways of naming the same
    thing, and they resolve to the same index here so that everything
    downstream -- interval building, cutting, the report -- stays unaware
    of which one the user reached for.

    A name and a reference together mean "this person, wherever they are":
    a scan where somebody named them uses those cards as named, and any
    other scan is searched by face. That is how a person named once is
    found across a folder of videos.

    Raises:
        SelectionError: If no person was named, the index is out of range,
            or the reference matched nobody clearly enough.
    """
    if select_name:
        wanted = [
            position
            for position, group in enumerate(identity_gallery.groups)
            if group.name and group.name.casefold() == select_name.casefold()
        ]
        if wanted:
            print(f"{select_name} is named in this scan.")
            return wanted
        if reference is None:
            known = sorted(
                group.name for group in identity_gallery.groups if group.name
            )
            raise SelectionError(
                f"Nobody in this scan is called {select_name!r}. "
                + (f"Named so far: {', '.join(known)}." if known else
                   "Nobody has been named yet -- name them in the window.")
            )

    if reference is not None:
        try:
            match = match_reference(
                reference,
                identity_gallery.groups,
                minimum_similarity=reference_threshold,
                mode=mode,
            )
        except ReferenceError as error:
            raise SelectionError(str(error)) from error

        runner_up = (
            "no runner-up"
            if match.runner_up_similarity is None
            else f"next best {match.runner_up_similarity:.2f}"
        )
        print(
            f"{reference.name} matched Person #{match.index + 1} "
            f"at {match.similarity:.2f} ({runner_up})."
        )
        return [match.index]

    if select_index is None:
        raise SelectionError(
            "Choose a person with --select-index, --select-name or --reference."
        )

    wanted = [select_index] if isinstance(select_index, int) else list(select_index)
    if not wanted:
        raise SelectionError("Choose a person with --select-index or --reference.")

    for index in wanted:
        if index < 0 or index >= len(identity_gallery.groups):
            raise SelectionError(
                f"--select-index {index} is out of range "
                f"(0-{len(identity_gallery.groups) - 1})."
            )

    # Deduplicated and ordered, so `--select-index 2 0 2` is the same reel
    # as `--select-index 0 2` rather than counting anybody twice.
    return sorted(set(wanted))


@dataclass(frozen=True)
class PipelineResult:
    """What one identity-grouping pass produced, including stream tallies."""

    grouper: IdentityGrouper
    total_detections: int
    track_count: int
    embedding_time: float
    grouping_time: float
    frame_count: int
    last_timestamp: float


def run_identity_pipeline(
    frames: Iterator[tuple[float, np.ndarray]],
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    consolidation_threshold: float,
    min_confidence: float,
    min_face_size: int,
    min_group_eye_span: float,
    min_detections: int,
    forbid_cooccurring: bool = DEFAULT_FORBID_COOCCURRING,
    cooccurrence_similarity_ceiling: float = DEFAULT_COOCCURRENCE_SIMILARITY_CEILING,
    mode: str = DEFAULT_MODE,
) -> "PipelineResult":
    """Runs detect -> crop -> embed -> track -> group over sampled frames.

    Shared by the `group`, `timestamps` and `export` commands and by the
    desktop UI, so every caller drives the same identity-grouping pipeline
    instead of maintaining several copies of it.

    Detections are linked into tracks by spatial continuity before any
    identity matching happens, so the grouper compares track-averaged
    embeddings rather than individual noisy frames.

    `frames` is consumed once as it streams, so this also tallies the
    frame count and the last timestamp seen: callers used to read those
    off a materialized list, which is exactly the thing that made memory
    scale with video length.
    """
    # The mode decides which models run. Built here, one job at a time, so
    # that selecting Live Action never loads the animation models and vice
    # versa -- the two sets together are most of a gigabyte resident.
    spec = get_mode(mode)
    detector = spec.build_detector(
        confidence_threshold=confidence_threshold,
        min_face_size=spec.detection.min_face_size,
    )
    embedder = spec.build_embedder()
    # The tracker's contradiction floor is a similarity, so it belongs to
    # the embedding space, not to the tracker: 0.25 separates a shot cut
    # from a continuing face under ArcFace and never fires at all under
    # CCIP, where two different characters already sit near 0.57.
    tracker = FaceTracker(contradiction_floor=spec.grouping.contradiction_floor)
    grouper = IdentityGrouper(
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        consolidation_threshold=consolidation_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        min_group_eye_span=min_group_eye_span,
        min_detections=min_detections,
        forbid_cooccurring=forbid_cooccurring,
        cooccurrence_similarity_ceiling=cooccurrence_similarity_ceiling,
    )

    total_detections = 0
    embedding_time = 0.0
    grouping_time = 0.0
    frame_count = 0
    last_timestamp = 0.0

    try:
        for frame_index, (timestamp, frame_data) in enumerate(frames):
            frame_count += 1
            last_timestamp = timestamp
            detections = detector.detect(frame_data)
            total_detections += len(detections)

            # Faces the grouper would throw away need never be embedded.
            # Embedding is the dominant cost and 12.3% of detections on the
            # test footage were below the confidence or size floor, so this
            # is work with no consumer. The grouper is asked rather than the
            # thresholds re-tested here, so there is one definition of what
            # counts. Dropping them also shortens the tracker's input, which
            # is why this was measured end to end rather than assumed: the
            # resulting partition is identical -- same 43 identities, 0
            # disagreements over 2.44 million co-membership pairs.
            gradeable = [d for d in detections if grouper.accepts_detection(d)]

            # Crop first, so faces that cannot be cropped never reach the
            # embedder and the batch stays aligned with what survived.
            croppable: list[tuple[FaceDetection, np.ndarray]] = []
            for detection in gradeable:
                try:
                    croppable.append(
                        (detection, crop_face(frame_data, detection, padding_ratio=padding_ratio))
                    )
                except ValueError:
                    continue

            # One forward pass for every face in this frame rather than one
            # per face. Embedding is the pipeline's dominant compute cost,
            # though the win is modest: batch size is faces-per-frame, ~2.2
            # on the test footage regardless of sampling interval (7i).
            embed_start = time.monotonic()
            embeddings = embedder.embed_batch(
                frame_data, [detection for detection, _ in croppable]
            )
            embedding_time += time.monotonic() - embed_start

            observations = [
                FaceObservation(
                    embedding=embedded.embedding,
                    embedding_space=embedded.embedding_space,
                    detection=detection,
                    face_crop=face_crop,
                    source_timestamp=timestamp,
                    frame_index=frame_index,
                    sharpness=embedded.sharpness,
                )
                for (detection, face_crop), embedded in zip(croppable, embeddings)
                if embedded is not None
            ]

            track_start = time.monotonic()
            tracker.add_frame(frame_index, observations)
            grouping_time += time.monotonic() - track_start
    finally:
        detector.close()
        embedder.close()

    group_start = time.monotonic()
    tracks = tracker.finish()
    for track in tracks:
        grouper.add_track(track)
    # add_track only buffers; clustering is lazy and used to run on the
    # caller's first access to .groups, which is outside this timer. The
    # reported grouping time was therefore the buffering alone -- 0.04s
    # against the 13.88s the clustering actually costs on test_3.mp4, a
    # figure low enough to have been quoted as a reason not to optimise
    # clustering at all. Forcing it here makes the number mean what it says.
    grouper.finish()
    grouping_time += time.monotonic() - group_start

    return PipelineResult(
        grouper=grouper,
        total_detections=total_detections,
        track_count=len(tracks),
        embedding_time=embedding_time,
        grouping_time=grouping_time,
        frame_count=frame_count,
        last_timestamp=last_timestamp,
    )


def scan_or_reuse(
    container,
    video_path: Path,
    *,
    sample_interval: float,
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    consolidation_threshold: float,
    min_confidence: float,
    min_face_size: int,
    min_group_eye_span: float,
    forbid_cooccurring: bool,
    cooccurrence_similarity_ceiling: float,
    mode: str,
    min_detections: int | None,
    use_cache: bool = True,
) -> CachedScan:
    """The scan every grouping command needs, run once and then kept.

    All three commands did the same four things -- sample, resolve the
    screen-time cutoff, run the pipeline, check something came back -- and
    then threw the answer away. The documented workflow made that visible:
    `group` to see the montage, then `export --select-index 0`, which
    scanned the same footage a second time to reach the same identities.

    A hit here is the same answer, not a similar one: the key covers the
    file's identity and every setting that can change what comes out
    (app/scans.py), so there is nothing to verify and nothing to go stale.
    """
    video_duration = get_video_info(container)["duration"]

    key, kept = find_scan(
        video_path,
        sample_interval=sample_interval,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        consolidation_threshold=consolidation_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        min_group_eye_span=min_group_eye_span,
        mode=mode,
        forbid_cooccurring=forbid_cooccurring,
        cooccurrence_similarity_ceiling=cooccurrence_similarity_ceiling,
        min_detections=min_detections,
    )

    if use_cache:
        if kept is not None:
            age = time.time() - kept.created_at
            print(
                f"Reusing the scan of {video_path.name} from "
                f"{_ago(age)} ({kept.scan_seconds:.0f}s of work skipped). "
                "Pass --rescan to run it again."
            )
            return kept

    print(f"Sampling frames at a {sample_interval}-second interval...")
    resolved_min_detections = _resolve_min_detections(
        min_detections, video_duration, sample_interval
    )

    started = time.monotonic()
    result = run_identity_pipeline(
        extract_frames(container, sample_interval=sample_interval),
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        consolidation_threshold=consolidation_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        min_group_eye_span=min_group_eye_span,
        forbid_cooccurring=forbid_cooccurring,
        cooccurrence_similarity_ceiling=cooccurrence_similarity_ceiling,
        mode=mode,
        min_detections=resolved_min_detections,
    )
    scan_seconds = time.monotonic() - started

    scan = CachedScan(
        groups=result.grouper.groups,
        unassigned_count=len(result.grouper.unassigned),
        total_detections=result.total_detections,
        track_count=result.track_count,
        frame_count=result.frame_count,
        last_timestamp=result.last_timestamp,
        embedding_time=result.embedding_time,
        grouping_time=result.grouping_time,
        video_duration=video_duration,
        min_detections=resolved_min_detections,
        created_at=time.time(),
        scan_seconds=scan_seconds,
        video=Path(video_path).name,
    )

    # Only worth keeping if there is something in it. A scan that found no
    # frames is a video that could not be read, and storing that would
    # cache the failure rather than the work.
    if scan.frame_count > 0:
        save_scan(key, scan)

    return scan


def _ago(seconds: float) -> str:
    """A rough age, for saying when a reused scan was made."""
    if seconds < 90:
        return "moments ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f} hours ago"
    return f"{seconds / 86400:.0f} days ago"


def run_face_grouping(
    container,
    output_dir: Path,
    sample_interval: float,
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    consolidation_threshold: float,
    min_confidence: float,
    min_face_size: int,
    min_group_eye_span: float,
    forbid_cooccurring: bool,
    cooccurrence_similarity_ceiling: float,
    mode: str,
    min_detections: int | None,
    select_index: int | None,
    video_path: Path | None = None,
    use_cache: bool = True,
):
    """Detect, embed, and group faces into per-identity clusters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving identity gallery output to: {output_dir.resolve()}")

    start_time = time.monotonic()
    result = scan_or_reuse(
        container,
        video_path if video_path is not None else Path("unknown"),
        sample_interval=sample_interval,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        consolidation_threshold=consolidation_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        min_group_eye_span=min_group_eye_span,
        forbid_cooccurring=forbid_cooccurring,
        cooccurrence_similarity_ceiling=cooccurrence_similarity_ceiling,
        mode=mode,
        min_detections=min_detections,
        use_cache=use_cache and video_path is not None,
    )

    if result.frame_count == 0:
        print("No frames extracted.")
        return

    identity_gallery = build_identity_gallery(
        result.groups,
        unassigned_count=result.unassigned_count,
        padding_ratio=padding_ratio,
    )
    montage_path = output_dir / "identity-gallery.jpg"
    save_identity_gallery_montage(identity_gallery.cards, montage_path)

    duration = time.monotonic() - start_time
    fps = result.frame_count / duration if duration > 0 else 0

    print("\n--- Grouping Report ---")
    print(f"Frames processed: {result.frame_count}")
    print(f"Detections received: {result.total_detections}")
    print(f"Face tracks built: {result.track_count}")
    print(f"Identity groups found: {len(identity_gallery.cards)}")
    print(f"Observations grouped: {identity_gallery.total_observations - identity_gallery.unassigned_count}")
    print(f"Unassigned detections: {identity_gallery.unassigned_count}")
    print(f"Embedding time: {result.embedding_time:.2f} seconds")
    print(f"Grouping time: {result.grouping_time:.2f} seconds")
    print(f"Total processing time: {duration:.2f} seconds")
    print(f"Processing speed: {fps:.2f} frames/sec")
    print(f"Identity gallery montage: {montage_path}")
    print("--- End Report ---\n")

    if select_index is not None:
        selected_card = identity_gallery.cards[select_index]
        print(format_person_card(selected_card, index=select_index))


def run_appearance_timestamps(
    container,
    sample_interval: float,
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    consolidation_threshold: float,
    min_confidence: float,
    min_face_size: int,
    min_group_eye_span: float,
    forbid_cooccurring: bool,
    cooccurrence_similarity_ceiling: float,
    mode: str,
    min_detections: int | None,
    gap_tolerance_seconds: float | None,
    appearance_padding_seconds: float | None,
    select_index: int | None = None,
    reference: ReferenceFace | None = None,
    reference_threshold: float | None = None,
    video_path: Path | None = None,
    use_cache: bool = True,
    select_name: str | None = None,
):
    """Group faces, then compute appearance intervals for one selected person."""
    start_time = time.monotonic()
    result = scan_or_reuse(
        container,
        video_path if video_path is not None else Path("unknown"),
        sample_interval=sample_interval,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        consolidation_threshold=consolidation_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        min_group_eye_span=min_group_eye_span,
        forbid_cooccurring=forbid_cooccurring,
        cooccurrence_similarity_ceiling=cooccurrence_similarity_ceiling,
        mode=mode,
        min_detections=min_detections,
        use_cache=use_cache and video_path is not None,
    )

    if result.frame_count == 0:
        print("No frames extracted.")
        return

    # The fallback needs the last sampled timestamp, which with streaming
    # is not known until the frames have been consumed -- and is stored
    # with the scan so a reused one answers it too.
    video_duration = result.video_duration
    if video_duration is None:
        video_duration = result.last_timestamp
        print(f"Warning: video duration unavailable; using last sampled timestamp ({video_duration:.2f}s) instead.")

    identity_gallery = build_identity_gallery(
        result.groups,
        unassigned_count=result.unassigned_count,
        padding_ratio=padding_ratio,
    )

    if not identity_gallery.groups:
        print("No identity groups found; nothing to compute appearance intervals for.")
        return

    chosen = _resolve_selection(
        identity_gallery, select_index, reference, reference_threshold, mode,
        select_name,
    )
    selected_group = combined_group(
        [identity_gallery.groups[index] for index in chosen]
    )
    selection_name = _selection_name(chosen, identity_gallery.groups)

    timeline_start = time.monotonic()
    intervals = build_appearance_intervals(
        selected_group,
        video_duration=video_duration,
        sample_interval=sample_interval,
        gap_tolerance_seconds=gap_tolerance_seconds,
        padding_seconds=appearance_padding_seconds,
    )
    timeline_duration = time.monotonic() - timeline_start
    total_duration = time.monotonic() - start_time

    print(f"\n--- Appearance Timestamps: {selection_name} ---")
    print(f"Detections for this selection: {selected_group.size}")
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Appearance intervals: {len(intervals)}")
    for index, interval in enumerate(intervals, start=1):
        print(f"\nAppearance {index}:")
        print(f"  Start: {format_timestamp(interval.start_time)}  ({interval.start_time:.2f}s)")
        print(f"  End:   {format_timestamp(interval.end_time)}  ({interval.end_time:.2f}s)")
    print(f"\nTotal detections processed: {result.total_detections}")
    print(f"Face tracks built: {result.track_count}")
    print(f"Embedding time: {result.embedding_time:.2f} seconds")
    print(f"Grouping time: {result.grouping_time:.2f} seconds")
    print(f"Timestamp-generation time: {timeline_duration:.4f} seconds")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print("--- End Report ---\n")


@dataclass(frozen=True)
class ExportPlan:
    """What would be cut from one video, decided before any encoding."""

    video_path: Path
    selection_name: str
    selected_group: FaceIdentityGroup
    intervals: list
    segments: list

    @property
    def seconds(self) -> float:
        return sum(s.end_time - s.start_time for s in self.segments)


def plan_export(
    scan: CachedScan,
    video_path: Path,
    *,
    sample_interval: float,
    padding_ratio: float,
    mode: str,
    gap_tolerance_seconds: float | None,
    appearance_padding_seconds: float | None,
    bridge_gap_seconds: float | None,
    min_segment_seconds: float | None,
    export_padding_seconds: float | None,
    select_index: int | list[int] | None = None,
    reference: ReferenceFace | None = None,
    reference_threshold: float | None = None,
    select_name: str | None = None,
) -> ExportPlan | None:
    """Chooses the person in a scan and works out the segments to cut.

    Separate from the cut so a folder of videos can be planned in full
    before a reel joining them starts encoding. Returns None, having said
    why, when there is nothing to cut.

    Raises:
        SelectionError: If the person cannot be picked out of this scan.
    """
    if scan.frame_count == 0:
        print("No frames extracted.")
        return None

    video_duration = scan.video_duration
    if video_duration is None:
        video_duration = scan.last_timestamp
        print(f"Warning: video duration unavailable; using last sampled timestamp ({video_duration:.2f}s) instead.")

    identity_gallery = build_identity_gallery(
        scan.groups,
        unassigned_count=scan.unassigned_count,
        padding_ratio=padding_ratio,
    )

    if not identity_gallery.groups:
        print("No identity groups found; nothing to export.")
        return None

    chosen = _resolve_selection(
        identity_gallery, select_index, reference, reference_threshold, mode,
        select_name,
    )
    selected_group = combined_group(
        [identity_gallery.groups[index] for index in chosen]
    )
    selection_name = _selection_name(chosen, identity_gallery.groups)
    intervals = build_appearance_intervals(
        selected_group,
        video_duration=video_duration,
        sample_interval=sample_interval,
        gap_tolerance_seconds=gap_tolerance_seconds,
        padding_seconds=appearance_padding_seconds,
    )
    segments = merge_for_export(
        intervals,
        video_duration=video_duration,
        bridge_gap_seconds=bridge_gap_seconds,
        min_segment_seconds=min_segment_seconds,
        padding_seconds=export_padding_seconds,
    )

    if not segments:
        print("No segments to export for this person.")
        return None

    return ExportPlan(
        video_path=video_path,
        selection_name=selection_name,
        selected_group=selected_group,
        intervals=intervals,
        segments=segments,
    )


def run_export(
    container,
    video_path: Path,
    output_path: Path,
    sample_interval: float,
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    consolidation_threshold: float,
    min_confidence: float,
    min_face_size: int,
    min_group_eye_span: float,
    forbid_cooccurring: bool,
    cooccurrence_similarity_ceiling: float,
    mode: str,
    min_detections: int | None,
    gap_tolerance_seconds: float | None,
    appearance_padding_seconds: float | None,
    bridge_gap_seconds: float | None,
    min_segment_seconds: float | None,
    export_padding_seconds: float | None,
    video_encoder: str,
    audio_encoder: str,
    quality: int,
    include_audio: bool,
    select_index: int | None = None,
    reference: ReferenceFace | None = None,
    reference_threshold: float | None = None,
    use_cache: bool = True,
    select_name: str | None = None,
):
    """Groups faces, then cuts one person's appearances into a single reel."""
    start_time = time.monotonic()
    result = scan_or_reuse(
        container,
        video_path,
        sample_interval=sample_interval,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        consolidation_threshold=consolidation_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        min_group_eye_span=min_group_eye_span,
        forbid_cooccurring=forbid_cooccurring,
        cooccurrence_similarity_ceiling=cooccurrence_similarity_ceiling,
        mode=mode,
        min_detections=min_detections,
        use_cache=use_cache,
    )

    plan = plan_export(
        result,
        video_path,
        sample_interval=sample_interval,
        padding_ratio=padding_ratio,
        mode=mode,
        gap_tolerance_seconds=gap_tolerance_seconds,
        appearance_padding_seconds=appearance_padding_seconds,
        bridge_gap_seconds=bridge_gap_seconds,
        min_segment_seconds=min_segment_seconds,
        export_padding_seconds=export_padding_seconds,
        select_index=select_index,
        reference=reference,
        reference_threshold=reference_threshold,
        select_name=select_name,
    )
    if plan is None:
        return
    selection_name = plan.selection_name
    selected_group = plan.selected_group
    intervals = plan.intervals
    segments = plan.segments

    appearance_seconds = sum(i.end_time - i.start_time for i in intervals)
    segment_seconds = sum(s.end_time - s.start_time for s in segments)

    print(f"\n--- Export: {selection_name} ---")
    print(f"Detections for this selection: {selected_group.size}")
    print(f"Appearance intervals: {len(intervals)} ({appearance_seconds:.1f}s on screen)")
    print(
        f"Segments to cut: {len(segments)} ({segment_seconds:.1f}s) "
        f"after bridging short gaps and enforcing a minimum length"
    )
    print(f"Encoding with {video_encoder}...")

    def report(index: int, total: int, segment) -> None:
        print(
            f"  cut {index + 1}/{total}  "
            f"{format_timestamp(segment.start_time)} -> "
            f"{format_timestamp(segment.end_time)}  "
            f"({segment.end_time - segment.start_time:.2f}s)"
        )

    export = cut_segments(
        video_path,
        segments,
        output_path,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        quality=quality,
        include_audio=include_audio,
        on_segment=report,
    )

    total_duration = time.monotonic() - start_time
    speed = export.exported_seconds / export.encode_seconds if export.encode_seconds > 0 else 0

    print(f"\nWrote: {export.output_path}")
    print(f"Reel duration: {export.exported_seconds:.1f} seconds from {export.segment_count} segments")
    print(f"Encoding time: {export.encode_seconds:.1f} seconds ({speed:.2f}x realtime)")
    print(f"Total processing time: {total_duration:.1f} seconds")
    print("--- End Report ---\n")

    return export


# ------------------------------------------------------------------- batch


def _without_repeats(plans: list["ExportPlan"]) -> tuple[list["ExportPlan"], dict[Path, float]]:
    """Leaves out what a later video repeats of footage already in the reel.

    See app/video/repeats.py. Fingerprinting costs one more read of each
    video the first time, and is kept beside the scans after that.
    """
    from app.scans import scan_cache_dir

    directory = scan_cache_dir() / "fingerprints"
    prints = {}
    for position, plan in enumerate(plans):
        print(f"  checking {plan.video_path.name} for repeated footage...")
        try:
            prints[position] = fingerprint_kept(plan.video_path, directory)
        except Exception:
            continue
    repeats = {
        (later, earlier): found
        for later in prints
        for earlier in prints
        if earlier < later and (found := find_repeats(prints[later], prints[earlier]))
    }
    trimmed, removed = without_repeats(
        [(position, plan.segments) for position, plan in enumerate(plans)], repeats
    )
    return (
        [replace(plan, segments=segments) for plan, (_, segments) in zip(plans, trimmed)],
        {plans[position].video_path: seconds for position, seconds in removed.items()},
    )


def collect_videos(paths: list[Path], recursive: bool = False) -> list[Path]:
    """Expands a mix of files and folders into a sorted list of videos.

    Sorted because a season is watched in order and a report that jumps
    around is harder to read than one that does not. Duplicates are
    dropped: naming a folder and one file inside it is a reasonable thing
    to type and should not cut the same reel twice.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            found.extend(
                child
                for child in sorted(path.glob(pattern))
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            found.append(path)

    unique: list[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


@dataclass(frozen=True)
class BatchOutcome:
    """What happened to one video in a batch."""

    video_path: Path
    output_path: Path | None = None
    reel_seconds: float = 0.0
    similarity: float | None = None
    # Why nothing was written, when nothing was. None on success.
    skipped_because: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.output_path is not None


def batch_output_path(video_path: Path, output_dir: Path, suffix: str = "reel") -> Path:
    """Names one episode's reel after the episode it came from."""
    return output_dir / f"{video_path.stem}-{suffix}.mp4"


# The settings a scan's cache key is built from, and the ones that turn a
# scan into segments. Batch takes one dictionary of everything and hands
# each stage its share.
SCAN_SETTINGS = (
    "sample_interval",
    "confidence_threshold",
    "padding_ratio",
    "similarity_threshold",
    "margin_threshold",
    "consolidation_threshold",
    "min_confidence",
    "min_face_size",
    "min_group_eye_span",
    "forbid_cooccurring",
    "cooccurrence_similarity_ceiling",
    "mode",
    "min_detections",
)
PLAN_SETTINGS = (
    "sample_interval",
    "padding_ratio",
    "mode",
    "gap_tolerance_seconds",
    "appearance_padding_seconds",
    "bridge_gap_seconds",
    "min_segment_seconds",
    "export_padding_seconds",
)
CUT_SETTINGS = ("video_encoder", "audio_encoder", "quality", "include_audio")


def _share(settings: dict, names: tuple[str, ...]) -> dict:
    return {name: settings[name] for name in names if name in settings}


def find_person(videos: list[Path], name: str, settings: dict) -> ReferenceFace:
    """The face of someone named on a card, from the kept scans of these videos.

    Nothing is scanned or decoded: a name only exists in a scan somebody
    has already opened and named people in, so the kept scans are the whole
    of what there is to search. Every card with the name counts, across
    every video, and their faces are averaged -- so naming the person in
    two episodes gives a steadier target than naming them in one.

    Raises:
        SelectionError: If nobody in these videos' kept scans has the name,
            which is said before anything long starts, with the names that
            do exist.
    """
    named: list[FaceIdentityGroup] = []
    named_in: list[Path] = []
    known: set[str] = set()
    for path in videos:
        try:
            _, kept = find_scan(path, **_share(settings, SCAN_SETTINGS))
        except OSError:
            continue
        if kept is None:
            continue
        here = [
            group
            for group in kept.groups
            if group.name and group.name.casefold() == name.casefold()
        ]
        known.update(group.name for group in kept.groups if group.name)
        if here:
            named.extend(here)
            named_in.append(path)

    if not named:
        # Named somewhere else -- another season, another folder -- counts:
        # the people library keeps every face a name was given on.
        from app.faces import library

        mode = settings.get("mode") or DEFAULT_MODE
        remembered = library.find(name, get_mode(mode).embedding_space)
        if remembered is not None:
            print(
                f"{remembered.name} is not named in these videos; finding them "
                f"by the face saved from {remembered.videos} earlier "
                f"video{'s' if remembered.videos != 1 else ''}."
            )
            average = remembered.faces.mean(axis=0)
            return ReferenceFace(
                embedding=(average / np.linalg.norm(average)).astype(np.float32),
                embedding_space=remembered.space,
                detection=None,
                source=None,
                face_count=1,
                label=remembered.name,
            )
        everyone = sorted(known | {p.name for p in library.known()})
        raise SelectionError(
            f"Nobody is named {name!r} in these videos or any other. Open one "
            "in the window, click their card and give it that name, then run "
            "this again."
            + (f" Names so far: {', '.join(everyone)}." if everyone else "")
        )

    # Spelled the way it was saved, not the way it was typed.
    name = named[0].name
    try:
        reference = reference_from_groups(named, label=name)
    except ReferenceError as error:
        raise SelectionError(str(error)) from error

    elsewhere = len(videos) - len(named_in)
    print(
        f"{name} is named in {len(named_in)} of {len(videos)} "
        f"video{'s' if len(videos) != 1 else ''}"
        + (f"; finding them in the other {elsewhere} by face." if elsewhere else ".")
    )
    return reference


def run_batch(
    video_paths: list[Path],
    reference: ReferenceFace | None,
    output_dir: Path,
    export_settings: dict,
    recursive: bool = False,
    reference_threshold: float | None = None,
    person: str | None = None,
    combine_path: Path | None = None,
    use_cache: bool = True,
    keep_repeats: bool = False,
) -> list[BatchOutcome]:
    """Cuts one person out of every video in a folder.

    The person is named one of two ways, and neither needs a picture of
    them from anywhere but the footage:

    - `person`, a name given to their card in any of these videos. Scans
      where they are named use those cards; every other video is searched
      for the same face (`find_person`).
    - `reference`, a photograph, for when nobody has been named yet.

    Either way every video is matched against the *same* face, so nothing
    needs to decide that a stranger in episode 3 is the person from
    episode 1 by comparing the two episodes to each other -- a comparison
    that would have to survive a change of lighting, camera and costume.

    With `combine_path`, the videos become one reel rather than one each.
    Every video is scanned and planned first and the reel is cut once, at
    the end, so a video that cannot contribute is found out before minutes
    of encoding depend on it.

    Nothing here aborts the run. A season is twenty scans of several
    minutes each, and losing the other nineteen because episode three is
    a different mode, or holds nobody who matches, or was never a readable
    video, is the one failure mode that would make this unusable. Every
    video's outcome is recorded and the summary says which produced a reel.

    Raises:
        SelectionError: Only before any video is touched: when `person`
            is named nowhere, or no way of naming someone was given.
    """
    videos = collect_videos(video_paths, recursive=recursive)
    if not videos:
        print("No videos found to process.", file=sys.stderr)
        return []

    if person:
        target = find_person(videos, person, export_settings)
    elif reference is not None:
        target = reference
    else:
        raise SelectionError("Choose a person with a name or a reference photo.")

    if combine_path is not None:
        return _run_combined(
            videos, target, person, combine_path, export_settings,
            reference_threshold, use_cache, keep_repeats,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Cutting {target.name} out of {len(videos)} "
        f"video{'s' if len(videos) != 1 else ''} into {output_dir}.\n"
    )

    outcomes: list[BatchOutcome] = []
    for position, video_path in enumerate(videos, start=1):
        print(f"=== [{position}/{len(videos)}] {video_path.name}")
        destination = batch_output_path(video_path, output_dir)
        try:
            with load_video(video_path) as container:
                export = run_export(
                    container,
                    video_path=video_path,
                    output_path=destination,
                    reference=target,
                    select_name=person,
                    reference_threshold=reference_threshold,
                    use_cache=use_cache,
                    **export_settings,
                )
        except (VideoLoadError, SelectionError, CutterError, ReferenceError) as error:
            print(f"  skipped: {error}\n", file=sys.stderr)
            outcomes.append(
                BatchOutcome(video_path=video_path, skipped_because=str(error))
            )
            continue

        if export is None:
            # run_export returns nothing when there was no footage worth
            # cutting -- no frames, no identities, no segments. It has
            # already said which on the way past.
            outcomes.append(
                BatchOutcome(
                    video_path=video_path,
                    skipped_because="nothing to cut for this person",
                )
            )
            continue

        outcomes.append(
            BatchOutcome(
                video_path=video_path,
                output_path=export.output_path,
                reel_seconds=export.exported_seconds,
            )
        )

    print("--- Batch summary ---")
    written = [outcome for outcome in outcomes if outcome.succeeded]
    for outcome in outcomes:
        if outcome.succeeded:
            print(
                f"  {outcome.video_path.name} -> {Path(outcome.output_path).name} "
                f"({outcome.reel_seconds:.1f}s)"
            )
        else:
            print(f"  {outcome.video_path.name} -- {outcome.skipped_because}")
    total = sum(outcome.reel_seconds for outcome in written)
    print(
        f"\n{len(written)}/{len(outcomes)} produced a reel, "
        f"{total:.1f}s of footage in total."
    )
    print("--- End Batch ---\n")
    return outcomes


def _run_combined(
    videos: list[Path],
    target: ReferenceFace,
    person: str | None,
    combine_path: Path,
    export_settings: dict,
    reference_threshold: float | None,
    use_cache: bool,
    keep_repeats: bool = False,
) -> list[BatchOutcome]:
    """Plans every video, then cuts what they hold into one reel."""
    print(
        f"Cutting {target.name} out of {len(videos)} "
        f"video{'s' if len(videos) != 1 else ''} into one reel, {combine_path}.\n"
    )
    outcomes: dict[Path, BatchOutcome] = {}
    plans: list[ExportPlan] = []

    for position, video_path in enumerate(videos, start=1):
        print(f"=== [{position}/{len(videos)}] {video_path.name}")
        try:
            with load_video(video_path) as container:
                scan = scan_or_reuse(
                    container,
                    video_path,
                    use_cache=use_cache,
                    **_share(export_settings, SCAN_SETTINGS),
                )
            plan = plan_export(
                scan,
                video_path,
                reference=target,
                select_name=person,
                reference_threshold=reference_threshold,
                **_share(export_settings, PLAN_SETTINGS),
            )
        except (VideoLoadError, SelectionError, CutterError, ReferenceError) as error:
            print(f"  skipped: {error}\n", file=sys.stderr)
            outcomes[video_path] = BatchOutcome(video_path, skipped_because=str(error))
            continue

        if plan is None:
            outcomes[video_path] = BatchOutcome(
                video_path, skipped_because="nothing to cut for this person"
            )
            continue

        print(f"  {len(plan.segments)} segment(s), {plan.seconds:.1f}s\n")
        plans.append(plan)

    # Every video is read once more before the cut, so one that cannot be
    # is left out by name rather than taking the whole reel down with it.
    # A different frame rate is no longer a reason: the cut converts it.
    usable: list[ExportPlan] = []
    for plan in plans:
        try:
            probe_clip(plan.video_path)
        except CutterError as error:
            outcomes[plan.video_path] = BatchOutcome(
                plan.video_path, skipped_because=str(error)
            )
            continue
        usable.append(plan)

    repeated: dict[Path, float] = {}
    if not keep_repeats and len(usable) > 1:
        usable, repeated = _without_repeats(usable)
        for plan in [p for p in usable if not p.segments]:
            outcomes[plan.video_path] = BatchOutcome(
                plan.video_path,
                skipped_because="everything in it is already in the reel from an earlier video",
            )
        usable = [plan for plan in usable if plan.segments]

    if usable:
        cut_settings = _share(export_settings, CUT_SETTINGS)
        print(f"Encoding {sum(p.seconds for p in usable):.1f}s from {len(usable)} video(s)...")
        try:
            export = cut_clips(
                [Clip(plan.video_path, plan.segments) for plan in usable],
                combine_path,
                on_clip=lambda index, total, path: print(f"  [{index + 1}/{total}] {path.name}"),
                **cut_settings,
            )
        except CutterError as error:
            print(f"  the reel could not be cut: {error}\n", file=sys.stderr)
            for plan in usable:
                outcomes[plan.video_path] = BatchOutcome(
                    plan.video_path,
                    skipped_because=f"the reel could not be cut: {error}",
                )
        else:
            for plan, seconds in zip(usable, export.clip_seconds):
                outcomes[plan.video_path] = BatchOutcome(
                    plan.video_path,
                    output_path=export.output_path,
                    reel_seconds=seconds,
                )

    ordered = [outcomes[path] for path in videos]
    print("--- Batch summary ---")
    for outcome in ordered:
        if outcome.succeeded:
            left = repeated.get(outcome.video_path, 0.0)
            note = f"  (left out {left:.1f}s already in the reel)" if left >= 0.5 else ""
            print(f"  {outcome.video_path.name}  {outcome.reel_seconds:.1f}s{note}")
        else:
            print(f"  {outcome.video_path.name} -- {outcome.skipped_because}")
    contributed = [outcome for outcome in ordered if outcome.succeeded]
    if contributed:
        print(
            f"\n{len(contributed)}/{len(ordered)} videos went into {combine_path}, "
            f"{sum(o.reel_seconds for o in contributed):.1f}s in total."
        )
    else:
        print(f"\nNo video had anything to cut; {combine_path} was not written.")
    print("--- End Batch ---\n")
    return ordered
