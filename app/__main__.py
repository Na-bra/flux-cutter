import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.faces.grouper import (
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from app.main import (
    run_appearance_timestamps,
    run_face_detection,
    run_face_gallery,
    run_face_grouping,
)
from app.video.frames import extract_frames
from app.video.loader import VideoLoadError, get_video_info, load_video


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


if __name__ == "__main__":
    main()
