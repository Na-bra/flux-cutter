import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.faces.detector import FaceDetector
from app.faces.embedder import FaceEmbedder
from app.faces.grouper import FaceObservation, IdentityGrouper
from app.faces.tracker import FaceTracker
from app.ui.gallery import (
    build_face_gallery,
    build_identity_gallery,
    crop_face,
    format_person_card,
    format_selected_item,
    save_gallery_montage,
    save_identity_gallery_montage,
)
from app.video.frames import extract_frames
from app.video.loader import get_video_info
from app.video.timeline import build_appearance_intervals, format_timestamp


def run_face_detection(container, output_dir: Path, sample_interval: float):
    """Runs detection and saves annotated frames."""
    print(f"Extracting frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)
    if not frames:
        print("No frames extracted.")
        return

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
    if not frames:
        print("No frames extracted.")
        return

    detector = FaceDetector(confidence_threshold=confidence_threshold)
    detection_records: list[tuple[float, np.ndarray, list]] = []
    total_detections = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving gallery output to: {output_dir.resolve()}")

    start_time = time.monotonic()

    for timestamp, frame_data in frames:
        detections = detector.detect(frame_data)
        total_detections += len(detections)
        detection_records.append((timestamp, frame_data, detections))

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
    fps = len(frames) / duration if duration > 0 else 0

    print("\n--- Gallery Report ---")
    print(f"Frames processed: {len(frames)}")
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


def _run_identity_pipeline(
    frames: list[tuple[float, np.ndarray]],
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    min_confidence: float,
    min_face_size: int,
) -> tuple[IdentityGrouper, int, int, float, float]:
    """Runs detect -> crop -> embed -> track -> group over sampled frames.

    Shared by the `group` and `timestamps` commands so both drive the same
    identity-grouping pipeline instead of maintaining two copies of it.

    Detections are linked into tracks by spatial continuity before any
    identity matching happens, so the grouper compares track-averaged
    embeddings rather than individual noisy frames.

    Returns:
        (grouper, total_detections, track_count, embedding_time, grouping_time)
    """
    detector = FaceDetector(confidence_threshold=confidence_threshold)
    embedder = FaceEmbedder()
    tracker = FaceTracker()
    grouper = IdentityGrouper(
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
    )

    total_detections = 0
    embedding_time = 0.0
    grouping_time = 0.0

    try:
        for frame_index, (timestamp, frame_data) in enumerate(frames):
            detections = detector.detect(frame_data)
            total_detections += len(detections)

            observations = []
            for detection in detections:
                try:
                    face_crop = crop_face(frame_data, detection, padding_ratio=padding_ratio)
                except ValueError:
                    continue

                embed_start = time.monotonic()
                try:
                    embedding = embedder.embed(frame_data, detection)
                except ValueError:
                    continue
                embedding_time += time.monotonic() - embed_start

                observations.append(
                    FaceObservation(
                        embedding=embedding,
                        detection=detection,
                        face_crop=face_crop,
                        source_timestamp=timestamp,
                        frame_index=frame_index,
                    )
                )

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
    grouping_time += time.monotonic() - group_start

    return grouper, total_detections, len(tracks), embedding_time, grouping_time


def run_face_grouping(
    container,
    output_dir: Path,
    sample_interval: float,
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    min_confidence: float,
    min_face_size: int,
    select_index: int | None,
):
    """Detect, embed, and group faces into per-identity clusters."""
    print(f"Sampling frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)
    if not frames:
        print("No frames extracted.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving identity gallery output to: {output_dir.resolve()}")

    start_time = time.monotonic()
    grouper, total_detections, track_count, embedding_time, grouping_time = _run_identity_pipeline(
        frames,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
    )

    identity_gallery = build_identity_gallery(
        grouper.groups,
        unassigned_count=len(grouper.unassigned),
        padding_ratio=padding_ratio,
    )
    montage_path = output_dir / "identity-gallery.jpg"
    save_identity_gallery_montage(identity_gallery.cards, montage_path)

    duration = time.monotonic() - start_time
    fps = len(frames) / duration if duration > 0 else 0

    print("\n--- Grouping Report ---")
    print(f"Frames processed: {len(frames)}")
    print(f"Detections received: {total_detections}")
    print(f"Face tracks built: {track_count}")
    print(f"Identity groups found: {len(identity_gallery.cards)}")
    print(f"Observations grouped: {identity_gallery.total_observations - identity_gallery.unassigned_count}")
    print(f"Unassigned detections: {identity_gallery.unassigned_count}")
    print(f"Embedding time: {embedding_time:.2f} seconds")
    print(f"Grouping time: {grouping_time:.2f} seconds")
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
    min_confidence: float,
    min_face_size: int,
    gap_tolerance_seconds: float | None,
    appearance_padding_seconds: float | None,
    select_index: int,
):
    """Group faces, then compute appearance intervals for one selected person."""
    print(f"Sampling frames at a {sample_interval}-second interval...")
    frames = extract_frames(container, sample_interval=sample_interval)
    if not frames:
        print("No frames extracted.")
        return

    video_duration = get_video_info(container)["duration"]
    if video_duration is None:
        video_duration = frames[-1][0]
        print(f"Warning: video duration unavailable; using last sampled timestamp ({video_duration:.2f}s) instead.")

    start_time = time.monotonic()
    grouper, total_detections, track_count, embedding_time, grouping_time = _run_identity_pipeline(
        frames,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
    )

    identity_gallery = build_identity_gallery(
        grouper.groups,
        unassigned_count=len(grouper.unassigned),
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
    print(f"\nTotal detections processed: {total_detections}")
    print(f"Face tracks built: {track_count}")
    print(f"Embedding time: {embedding_time:.2f} seconds")
    print(f"Grouping time: {grouping_time:.2f} seconds")
    print(f"Timestamp-generation time: {timeline_duration:.4f} seconds")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print("--- End Report ---\n")
