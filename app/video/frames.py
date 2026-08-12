import av
import numpy as np


class FrameExtractionError(Exception):
    """Raised when frames cannot be extracted from a video."""


def extract_frames(
    container: av.container.InputContainer,
    sample_interval: float = 1.0,
) -> list[tuple[float, np.ndarray]]:
    """
    Extracts frames from a video at approximately regular time intervals.

    Args:
        container: The PyAV container for the opened video file.
        sample_interval: The approximate time in seconds between frames to extract.

    Returns:
        A list of (timestamp, frame) tuples, where:
        - timestamp is in seconds
        - frame is a NumPy array in RGB format

    Raises:
        ValueError: If `sample_interval` is not a positive number.
        FrameExtractionError: If the video contains no video stream.
    """
    if sample_interval <= 0:
        raise ValueError("sample_interval must be greater than 0")

    video_stream = next(
        (stream for stream in container.streams if stream.type == "video"),
        None,
    )
    if video_stream is None:
        raise FrameExtractionError("The container does not contain a video stream.")

    frames = []
    next_sample_time = 0.0

    container.seek(0)
    for frame in container.decode(video_stream):
        timestamp = float(frame.time)

        if timestamp >= next_sample_time:
            image = frame.to_ndarray(format="rgb24")
            frames.append((timestamp, image))
            next_sample_time += sample_interval

    return frames