import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from app.faces.detector import FaceDetector
from app.faces.embedder import FaceEmbedder
from app.faces.grouper import (
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    FaceObservation,
    IdentityGrouper,
)
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
from app.video.loader import VideoLoadError, get_video_info, load_video
from app.video.timeline import build_appearance_intervals, format_timestamp


def main():
    """Main entry point for the command-line utility."""
    parser = argparse.ArgumentParser(description="FluxCutter video processing utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'info' command
    info_parser = subparsers.add_parser("info", help="Get metadata for a video file.")
    info_parser.add_argument("video_path", type=Path, help="Path to the video file.")

    # 'extract' command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract frames from a video file."
    )
    extract_parser.add_argument(
        "video_path", type=Path, help="Path to the video file."
    )
    extract_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between extracted frames.",
    )

    # 'detect' command
    detect_parser = subparsers.add_parser(
        "detect", help="Detect faces in a video and save annotated frames."
    )
    detect_parser.add_argument("video_path", type=Path, help="Path to the video file.")
    detect_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/detected-faces"),
        help="Directory to save annotated frames.",
    )
    detect_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between frames to process.",
    )

    # 'gallery' command
    gallery_parser = subparsers.add_parser(
        "gallery", help="Generate a face gallery from sampled video frames."
    )
    gallery_parser.add_argument("video_path", type=Path, help="Path to the video file.")
    gallery_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/face-gallery"),
        help="Directory to save the gallery montage.",
    )
    gallery_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between sampled frames.",
    )
    gallery_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum detection confidence for gallery candidates.",
    )
    gallery_parser.add_argument(
        "--max-items",
        type=int,
        default=24,
        help="Maximum number of gallery thumbnails to keep.",
    )
    gallery_parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Padding ratio around each detected face crop.",
    )
    gallery_parser.add_argument(
        "--select-index",
        type=int,
        default=None,
        help="Optional gallery item index to print after generation.",
    )

    # 'group' command
    group_parser = subparsers.add_parser(
        "group", help="Group detected faces into per-identity clusters."
    )
    group_parser.add_argument("video_path", type=Path, help="Path to the video file.")
    group_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/face-groups"),
        help="Directory to save the identity gallery montage.",
    )
    group_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between sampled frames.",
    )
    group_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum detection confidence to consider a face at all.",
    )
    group_parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Padding ratio around each detected face crop.",
    )
    group_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Minimum cosine similarity to an existing group's centroid to assign a match.",
    )
    group_parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
        help="Minimum similarity gap over the second-best group before assigning a match.",
    )
    group_parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum detection confidence for a face to be used in grouping.",
    )
    group_parser.add_argument(
        "--min-face-size",
        type=int,
        default=DEFAULT_MIN_FACE_SIZE,
        help="Minimum face box side length (pixels) for a face to be used in grouping.",
    )
    group_parser.add_argument(
        "--select-index",
        type=int,
        default=None,
        help="Optional person card index to print after generation.",
    )

    # 'timestamps' command
    timestamps_parser = subparsers.add_parser(
        "timestamps", help="Compute appearance intervals for one selected identity group."
    )
    timestamps_parser.add_argument("video_path", type=Path, help="Path to the video file.")
    timestamps_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between sampled frames.",
    )
    timestamps_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum detection confidence to consider a face at all.",
    )
    timestamps_parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Padding ratio around each detected face crop (affects representative thumbnails only).",
    )
    timestamps_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Minimum cosine similarity to an existing group's centroid to assign a match.",
    )
    timestamps_parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
        help="Minimum similarity gap over the second-best group before assigning a match.",
    )
    timestamps_parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum detection confidence for a face to be used in grouping.",
    )
    timestamps_parser.add_argument(
        "--min-face-size",
        type=int,
        default=DEFAULT_MIN_FACE_SIZE,
        help="Minimum face box side length (pixels) for a face to be used in grouping.",
    )
    timestamps_parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=None,
        help="Seconds between detections before starting a new appearance interval. "
        "Defaults to 2x --interval.",
    )
    timestamps_parser.add_argument(
        "--appearance-padding",
        type=float,
        default=None,
        help="Seconds of padding added before/after each appearance interval. Defaults to 0.5x --interval.",
    )
    timestamps_parser.add_argument(
        "--select-index",
        type=int,
        required=True,
        help="Person card index (as shown by the 'group' command) to compute appearance intervals for.",
    )

    args = parser.parse_args()

    try:
        with load_video(args.video_path) as container:
            if args.command == "info":
                info = get_video_info(container)
                print(info)
            elif args.command == "extract":
                frames = extract_frames(container, sample_interval=args.interval)
                print(f"Extracted {len(frames)} frames.")
            elif args.command == "detect":
                run_face_detection(
                    container,
                    output_dir=args.output_dir,
                    sample_interval=args.interval,
                )
            elif args.command == "gallery":
                run_face_gallery(
                    container,
                    output_dir=args.output_dir,
                    sample_interval=args.interval,
                    confidence_threshold=args.confidence_threshold,
                    max_items=args.max_items,
                    padding_ratio=args.padding,
                    select_index=args.select_index,
                )
            elif args.command == "group":
                run_face_grouping(
                    container,
                    output_dir=args.output_dir,
                    sample_interval=args.interval,
                    confidence_threshold=args.confidence_threshold,
                    padding_ratio=args.padding,
                    similarity_threshold=args.similarity_threshold,
                    margin_threshold=args.margin_threshold,
                    min_confidence=args.min_confidence,
                    min_face_size=args.min_face_size,
                    select_index=args.select_index,
                )
            elif args.command == "timestamps":
                run_appearance_timestamps(
                    container,
                    sample_interval=args.interval,
                    confidence_threshold=args.confidence_threshold,
                    padding_ratio=args.padding,
                    similarity_threshold=args.similarity_threshold,
                    margin_threshold=args.margin_threshold,
                    min_confidence=args.min_confidence,
                    min_face_size=args.min_face_size,
                    gap_tolerance_seconds=args.gap_tolerance,
                    appearance_padding_seconds=args.appearance_padding,
                    select_index=args.select_index,
                )

    except VideoLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


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
) -> tuple[IdentityGrouper, int, float, float]:
    """Runs detect -> crop -> embed -> group over already-sampled frames.

    Shared by the `group` and `timestamps` commands so both drive the same
    identity-grouping pipeline instead of maintaining two copies of it.

    Returns:
        (grouper, total_detections, embedding_time, grouping_time)
    """
    detector = FaceDetector(confidence_threshold=confidence_threshold)
    embedder = FaceEmbedder()
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

                observation = FaceObservation(
                    embedding=embedding,
                    detection=detection,
                    face_crop=face_crop,
                    source_timestamp=timestamp,
                    frame_index=frame_index,
                )

                group_start = time.monotonic()
                grouper.add(observation)
                grouping_time += time.monotonic() - group_start
    finally:
        detector.close()
        embedder.close()

    return grouper, total_detections, embedding_time, grouping_time


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
    grouper, total_detections, embedding_time, grouping_time = _run_identity_pipeline(
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
    grouper, total_detections, embedding_time, grouping_time = _run_identity_pipeline(
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
    print(f"Embedding time: {embedding_time:.2f} seconds")
    print(f"Grouping time: {grouping_time:.2f} seconds")
    print(f"Timestamp-generation time: {timeline_duration:.4f} seconds")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print("--- End Report ---\n")


if __name__ == "__main__":
    main()