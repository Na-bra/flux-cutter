from collections.abc import Iterator

import av
import numpy as np


class FrameExtractionError(Exception):
    """Raised when frames cannot be extracted from a video."""


def extract_frames(
    container: av.container.InputContainer,
    sample_interval: float = 1.0,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Yields frames from a video at approximately regular time intervals.

    This streams: exactly one decoded frame is held at a time, so memory is
    flat in the length of the video. It used to return a list, which made
    the peak footprint the whole sampled video at once -- roughly
    width x height x 3 x (duration / sample_interval) bytes. On a 22-minute
    720p episode that was ~4.9 GB at a 1.0s interval and ~15 GB at 0.25s,
    so the denser sampling that identity grouping actually wants was the
    sampling the machine could not afford.

    Two consequences for callers, both of which bite quietly:

    - The result can only be iterated once, and has no length. Callers that
      need a count should tally as they go rather than reach for len().
    - Decoding happens while iterating, not when this is called, so the
      iteration must finish *inside* the `with load_video(...)` block that
      owns the container. Consuming it after the container closes reads
      from a closed file.

    Argument validation stays eager -- bad arguments raise here, at the
    call, rather than being deferred to the first item.

    Args:
        container: The PyAV container for the opened video file.
        sample_interval: The approximate time in seconds between frames.

    Yields:
        (timestamp, frame) tuples, where timestamp is in seconds and frame
        is a NumPy array in RGB format.

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

    return _iter_sampled_frames(container, video_stream, sample_interval)


def _iter_sampled_frames(
    container: av.container.InputContainer,
    video_stream,
    sample_interval: float,
) -> Iterator[tuple[float, np.ndarray]]:
    """The decode loop itself, split out so validation above can stay eager."""
    next_sample_time = 0.0

    container.seek(0)
    for frame in container.decode(video_stream):
        timestamp = float(frame.time)

        if timestamp >= next_sample_time:
            yield timestamp, frame.to_ndarray(format="rgb24")

            # Advance past this frame, not by a single interval. The
            # schedule starts at zero, so footage that starts later --
            # an MP4 with a start offset, a clip cut from the middle of a
            # longer file -- leaves the target far behind the first frame,
            # and a single step still leaves it behind. Every following
            # frame then qualifies until the target catches up, so a clip
            # beginning at 5s returned six frames spanning 0.2 seconds
            # before settling into the interval it was asked for. Gaps
            # mid-file do the same thing: dropped frames and variable
            # frame rates both produce a burst of near-duplicates where
            # one sample was wanted.
            #
            # The loop keeps the samples on multiples of the interval
            # rather than spacing them from whatever frame happened to be
            # yielded, so the timing cannot drift over a long video.
            while next_sample_time <= timestamp:
                next_sample_time += sample_interval
