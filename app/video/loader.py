from pathlib import Path

import av


SUPPORTED_EXTENSIONS = {".mp4", ".mov"}


class VideoLoadError(Exception):
    """Raised when a video cannot be loaded or inspected."""


def validate_video_path(video_path: str | Path) -> Path:
    """
    Check that a path is a video this app will accept, and return it.

    Separated from load_video because VideoSource needs the same answer
    without opening anything: it validates a replacement file before
    committing to it.

    Raises:
        VideoLoadError: If the file does not exist, is not a file, or has
                        an unsupported extension.
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

    return path


def use_threaded_decoding(container: av.container.InputContainer) -> bool:
    """Ask the video stream to decode on more than one core.

    Decoding was the pipeline's largest single cost: 85 ms per sampled
    frame against 9 ms to detect a face and 29 ms to embed one. Nearly all
    of it is waste -- at a 1-second interval on 24 fps footage 23 of every
    24 decoded frames are dropped -- but they cannot be skipped, because a
    frame is only decodable from the ones before it. Seeking to each sample
    point instead is *slower*: it lands on the previous keyframe and
    decodes forward, which measured 1.9x the work.

    So decode the same frames, on more cores. Over 180 seconds of the 720p
    test footage: 15.4s -> 3.8s, which takes the whole pipeline down about
    40%. The frames that come out are byte-identical -- this changes how
    they are decoded, not which.

    This lives at open time because FFmpeg only accepts the setting while
    the codec is closed, and a container that has already decoded anything
    raises. Returns whether it was applied, so a caller that cares can tell
    a real speedup from a silent no-op.
    """
    video_stream = next(
        (stream for stream in container.streams if stream.type == "video"),
        None,
    )
    if video_stream is None:
        return False
    try:
        video_stream.thread_type = "AUTO"
    except (RuntimeError, ValueError):
        return False
    return True


def load_video(video_path: str | Path) -> av.container.InputContainer:
    """
    Open a video file and return the PyAV container.

    Raises:
        VideoLoadError: If the file does not exist, has an unsupported
                        extension, or cannot be opened.
    """
    path = validate_video_path(video_path)

    try:
        container = av.open(str(path))
    except av.FFmpegError as exc:
        # av.AVError, which this used to catch, no longer exists in PyAV 18 --
        # so the except clause itself raised AttributeError and a corrupt file
        # surfaced as "module 'av' has no attribute 'AVError'" instead of a
        # readable message. FFmpegError is the base every open() failure
        # derives from (InvalidDataError for a file that is not a video,
        # FileNotFoundError for a missing one).
        raise VideoLoadError(f"Could not open video: {path} ({exc})") from exc

    use_threaded_decoding(container)
    return container


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