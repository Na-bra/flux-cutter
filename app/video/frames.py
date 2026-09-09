from collections.abc import Iterator

import av
import numpy as np


class FrameExtractionError(Exception):
    """Raised when frames cannot be extracted from a video."""


def skip_nonreference_frames(video_stream) -> bool:
    """Ask the decoder not to fully decode frames nothing depends on.

    An H.264 stream is mostly B-frames that no other frame is predicted
    from. Sampling reads one frame every 0.5-1.0s and discards the rest,
    so the ones that are neither sampled nor referenced are decoded purely
    to be thrown away. `skip_frame = NONREF` tells FFmpeg to stop short on
    exactly those.

    Measured over the first 180s of the 720p test episode, at a 0.5s
    interval: 4316 frames decoded -> 2287, and 3.46s -> 3.00s. The same on
    1080p animation footage (5.56s -> 4.63s). All-intra footage such as
    test.mp4 has no non-reference frames at all, so this is a no-op there
    rather than a loss.

    The cost is that a sample can land on the next reference frame instead
    of the exact one, which moved the sampled timestamps by a median 0.035s
    and at most 0.083s -- two frames at 24 fps, against a sampling interval
    500 to 1000 times that. Detection, tracking and grouping all work off
    the timestamps that come back, so a slightly different neighbouring
    frame is a different sample, not a wrong one.

    This is deliberately not in `load_video`, which the export path also
    uses: cutting re-encodes every frame it reads, so a container opened
    for cutting must decode all of them.

    Returns whether it was applied, so a caller can tell a real saving from
    a silent no-op.
    """
    try:
        video_stream.codec_context.skip_frame = "NONREF"
    except (AttributeError, RuntimeError, ValueError):
        return False
    return True


def extract_frames(
    container: av.container.InputContainer,
    sample_interval: float = 1.0,
    skip_nonreference: bool = True,
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
        skip_nonreference: Skip decoding frames nothing is predicted
            from, which is faster and moves sampled timestamps by up
            to a couple of frames. See `skip_nonreference_frames`.
            Pass False to sample the exact frames on the schedule.

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

    return _iter_sampled_frames(
        container, video_stream, sample_interval, skip_nonreference
    )


def _iter_sampled_frames(
    container: av.container.InputContainer,
    video_stream,
    sample_interval: float,
    skip_nonreference: bool = True,
) -> Iterator[tuple[float, np.ndarray]]:
    """The decode loop itself, split out so validation above can stay eager."""
    next_sample_time = 0.0

    # The container belongs to the caller, who may well decode it again for
    # something that does want every frame, so the decoder is put back the
    # way it was found rather than left narrowed by a call that has ended.
    previous_skip = None
    applied = False
    if skip_nonreference:
        previous_skip = getattr(
            getattr(video_stream, "codec_context", None), "skip_frame", None
        )
        applied = skip_nonreference_frames(video_stream)

    try:
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

    finally:
        if applied and previous_skip is not None:
            try:
                video_stream.codec_context.skip_frame = previous_skip
            except (AttributeError, RuntimeError, ValueError):
                pass
