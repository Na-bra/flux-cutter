from pathlib import Path

import av


SUPPORTED_EXTENSIONS = {".mp4", ".mov"}


class VideoLoadError(Exception):
    """Raised when a video cannot be loaded or inspected."""


def load_video(video_path: str | Path) -> av.container.InputContainer:
    """
    Open a video file and return the PyAV container.

    Raises:
        VideoLoadError: If the file does not exist, has an unsupported
                        extension, or cannot be opened.
    """
    path = Path(video_path)

    if not path.exists():
        raise VideoLoadError(f"Video file does not exist: {path}")

    if not path.is_file():
        raise VideoLoadError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoLoadError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        return av.open(str(path))
    except av.AVError as exc:
        raise VideoLoadError(f"Could not open video: {path}") from exc


def get_video_info(container: av.container.InputContainer) -> dict:
    """Return basic information about the first video stream."""

    video_stream = next(
        (stream for stream in container.streams if stream.type == "video"),
        None,
    )

    if video_stream is None:
        raise VideoLoadError("The file does not contain a video stream.")

    return {
        "width": video_stream.width,
        "height": video_stream.height,
        "codec": video_stream.codec_context.name,
        "frames": video_stream.frames,
        "duration": (
            float(video_stream.duration * video_stream.time_base)
            if video_stream.duration is not None
            else None
        ),
        "fps": float(video_stream.average_rate)
        if video_stream.average_rate
        else None,
    }