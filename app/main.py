import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.faces.detector import FaceDetection, FaceDetector
from app.faces.embedder import FaceEmbedder
from app.faces.grouper import (
    DEFAULT_COOCCURRENCE_SIMILARITY_CEILING,
    DEFAULT_FORBID_COOCCURRING,
    FaceObservation,
    IdentityGrouper,
    auto_min_detections,
)
from app.faces.tracker import FaceTracker
from app.modes import DEFAULT_MODE, get_mode
from app.ui.gallery import (
    build_face_gallery,
    build_identity_gallery,
    crop_face,
    format_person_card,
    format_selected_item,
    save_gallery_montage,
    save_identity_gallery_montage,
)
from app.video.cutter import CutterError, cut_segments
from app.video.export import merge_for_export
from app.video.frames import extract_frames
from app.video.loader import get_video_info
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
):
    """Detect, embed, and group faces into per-identity clusters."""
    print(f"Sampling frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)

    video_duration = get_video_info(container)["duration"]
    resolved_min_detections = _resolve_min_detections(
        min_detections, video_duration, sample_interval
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving identity gallery output to: {output_dir.resolve()}")

    start_time = time.monotonic()
    result = run_identity_pipeline(
        frames,
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

    if result.frame_count == 0:
        print("No frames extracted.")
        return

    identity_gallery = build_identity_gallery(
        result.grouper.groups,
        unassigned_count=len(result.grouper.unassigned),
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
    select_index: int,
):
    """Group faces, then compute appearance intervals for one selected person."""
    print(f"Sampling frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)
    video_duration = get_video_info(container)["duration"]
    resolved_min_detections = _resolve_min_detections(
        min_detections, video_duration, sample_interval
    )

    start_time = time.monotonic()
    result = run_identity_pipeline(
        frames,
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

    if result.frame_count == 0:
        print("No frames extracted.")
        return

    # Resolved only now: the fallback needs the last sampled timestamp, and
    # with streaming that is not known until the frames have been consumed.
    if video_duration is None:
        video_duration = result.last_timestamp
        print(f"Warning: video duration unavailable; using last sampled timestamp ({video_duration:.2f}s) instead.")

    identity_gallery = build_identity_gallery(
        result.grouper.groups,
        unassigned_count=len(result.grouper.unassigned),
        padding_ratio=padding_ratio,
    )

    if not identity_gallery.groups:
        print("No identity groups found; nothing to compute appearance intervals for.")
        return

    if select_index < 0 or select_index >= len(identity_gallery.groups):
        print(
            f"Error: --select-index {select_index} is out of range "
            f"(0-{len(identity_gallery.groups) - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_group = identity_gallery.groups[select_index]

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

    print(f"\n--- Appearance Timestamps: Person #{select_index + 1} ---")
    print(f"Detections for this person: {selected_group.size}")
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
    select_index: int,
):
    """Groups faces, then cuts one person's appearances into a single reel."""
    print(f"Sampling frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)
    video_duration = get_video_info(container)["duration"]
    resolved_min_detections = _resolve_min_detections(
        min_detections, video_duration, sample_interval
    )

    start_time = time.monotonic()
    result = run_identity_pipeline(
        frames,
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

    if result.frame_count == 0:
        print("No frames extracted.")
        return

    if video_duration is None:
        video_duration = result.last_timestamp
        print(f"Warning: video duration unavailable; using last sampled timestamp ({video_duration:.2f}s) instead.")

    identity_gallery = build_identity_gallery(
        result.grouper.groups,
        unassigned_count=len(result.grouper.unassigned),
        padding_ratio=padding_ratio,
    )

    if not identity_gallery.groups:
        print("No identity groups found; nothing to export.")
        return

    if select_index < 0 or select_index >= len(identity_gallery.groups):
        print(
            f"Error: --select-index {select_index} is out of range "
            f"(0-{len(identity_gallery.groups) - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_group = identity_gallery.groups[select_index]
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
        return

    appearance_seconds = sum(i.end_time - i.start_time for i in intervals)
    segment_seconds = sum(s.end_time - s.start_time for s in segments)

    print(f"\n--- Export: Person #{select_index + 1} ---")
    print(f"Detections for this person: {selected_group.size}")
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

    try:
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
    except CutterError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    total_duration = time.monotonic() - start_time
    speed = export.exported_seconds / export.encode_seconds if export.encode_seconds > 0 else 0

    print(f"\nWrote: {export.output_path}")
    print(f"Reel duration: {export.exported_seconds:.1f} seconds from {export.segment_count} segments")
    print(f"Encoding time: {export.encode_seconds:.1f} seconds ({speed:.2f}x realtime)")
    print(f"Total processing time: {total_duration:.1f} seconds")
    print("--- End Report ---\n")
