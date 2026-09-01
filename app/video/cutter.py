"""Cutting a reel in-process, through PyAV, with no external ffmpeg.

This replaces shelling out to the `ffmpeg` binary. The reason is not
elegance, it is that the binary is not reliably there: a Finder-launched
.app inherits `/usr/bin:/bin:/usr/sbin:/sbin`, so a Homebrew ffmpeg is
invisible to it and export failed in the packaging trial while every other
stage worked (Instructions.md 9). PyAV is already a dependency, already
ships FFmpeg inside its wheel, and already carries every encoder the
exporter asks for -- so doing the work in-process removes a dependency
rather than adding one.

It is also structurally simpler than what it replaces. The subprocess
version wrote one temporary file per segment and joined them with the
concat demuxer, because that is how you do it with a command-line tool.
Holding the output container open means each segment's frames can be
encoded straight into the finished reel: no temporary files, no second
pass, no assumption that the segments share codec parameters.

Two things it must get right, both invisible until played:

- **Seeking is approximate; cutting must not be.** A seek lands on the
  keyframe at or before the target, which on real footage can be seconds
  early (7h). Frames are therefore decoded forward from there and dropped
  until the segment's true start -- the same thing `-ss` before `-i` does.
- **Timestamps must be continuous across the joins.** Source frames carry
  their original PTS, which jumps backwards between segments -- muxing those
  straight through gives a file whose timeline goes backwards twice a
  minute. Each stream therefore keeps its own running count and stamps
  frames against it, video by frame number and audio by sample number.

  Neither shortcut works. Computing the PTS from the source's own timestamps
  fails to mux at all (EINVAL), because an encoder's time_base is not what
  the stream reports before it is opened. Handing frames over with
  `pts = None` and letting the encoder assign them muxes happily and writes
  a file whose video duration is `0.033333` -- one frame -- with an
  `avg_frame_rate` of `10800/1`. Playback looks fine because the audio
  stream carries a sane duration, which is exactly why that one is worth
  spelling out: it is wrong in a way nothing but a probe will tell you.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

from app.video.loader import VideoLoadError
from app.video.source import VideoSource
from app.video.timeline import AppearanceInterval


class CutterError(Exception):
    """Raised when a reel cannot be cut."""


@dataclass(frozen=True)
class CutResult:
    """What one cut produced."""

    output_path: Path
    segment_count: int
    exported_seconds: float
    encode_seconds: float


# videotoolbox takes -q:v on a 0-100 scale where higher is better; x264 and
# its relatives take -crf on 0-51 where lower is better. The callers already
# translate between the two (app/ui/worker.quality_for); this only has to
# know which option name to hand each encoder.
def _quality_options(encoder: str, quality: int) -> dict[str, str]:
    if "videotoolbox" in encoder:
        return {"q:v": str(quality)}
    return {"crf": str(quality)}


# The output timeline's unit. 90kHz is the MP4 convention and is what the
# subprocess version asked ffmpeg for with -video_track_timescale.
VIDEO_TIME_BASE = Fraction(1, 90000)


def _open_output(
    output_path: Path,
    source_video,
    source_audio,
    video_encoder: str,
    audio_encoder: str,
    quality: int,
):
    """Opens the reel and configures its streams to match the source."""
    output = av.open(str(output_path), mode="w")

    video = output.add_stream(
        video_encoder,
        rate=source_video.average_rate or Fraction(30, 1),
        options=_quality_options(video_encoder, quality),
    )
    video.width = source_video.width
    video.height = source_video.height
    # yuv420p rather than the source's own format: it is what every player
    # can decode, and the test footage is yuv444p, which many cannot.
    video.pix_fmt = "yuv420p"
    video.codec_context.time_base = VIDEO_TIME_BASE

    audio = None
    if source_audio is not None:
        audio = output.add_stream(audio_encoder, rate=source_audio.rate)
        audio.layout = source_audio.layout
        # An encoder's codec_context.time_base is not populated until it is
        # opened, so timestamps are computed against the sample rate, which
        # is what an audio timebase is anyway.
        audio.time_base = Fraction(1, source_audio.rate)

    return output, video, audio


@dataclass
class _Counters:
    """How much of the reel has been written, per stream.

    Kept across segments: this is what makes the output one continuous
    timeline rather than three that each restart at zero.
    """

    video: int = 0
    audio: int = 0


class _AudioState:
    """Resampling and re-framing for the reel's audio.

    AAC encodes fixed-size frames (1024 samples), while decoded frames come
    in whatever size the source used, so they have to be rebuffered. One
    state object spans every segment rather than one per segment, which is
    also what keeps the audio continuous across the joins.
    """

    def __init__(self, out_audio):
        self._encoder = out_audio
        self._resampler = av.AudioResampler(
            format=out_audio.format, layout=out_audio.layout, rate=out_audio.rate
        )
        self._fifo = av.AudioFifo()
        self._frame_size = out_audio.codec_context.frame_size or 1024
        self._time_base = Fraction(1, out_audio.rate)

    def write(self, frame, output, counters) -> None:
        for resampled in self._resampler.resample(frame):
            resampled.pts = None
            self._fifo.write(resampled)
            self._drain(output, counters)

    def _drain(self, output, counters, partial: bool = False) -> None:
        while True:
            chunk = self._fifo.read(self._frame_size, partial=partial)
            if chunk is None:
                return
            chunk.pts = counters.audio
            chunk.time_base = self._time_base
            counters.audio += chunk.samples
            for packet in self._encoder.encode(chunk):
                output.mux(packet)

    def flush(self, output, counters) -> None:
        """Encodes whatever is left, including a final short frame."""
        self._drain(output, counters, partial=True)


def cut_segments(
    video: Path | VideoSource,
    segments: list[AppearanceInterval],
    output_path: Path,
    video_encoder: str = "libx264",
    audio_encoder: str = "aac",
    quality: int = 20,
    include_audio: bool = True,
    on_segment: Callable[[int, int, AppearanceInterval], None] | None = None,
) -> CutResult:
    """
    Cuts each segment out of the source and writes them as one reel.

    Args:
        video: The source video, as a path or as a VideoSource. A
            VideoSource keeps cutting after the file has been moved or
            renamed since the scan.
        segments: Non-overlapping segments in chronological order, as
            returned by merge_for_export.
        output_path: Where to write the joined result.
        video_encoder: Any encoder PyAV can construct. See
            app.ui.worker.available_encoders for what this machine has.
        audio_encoder: Used only when include_audio and the source has audio.
        quality: Constant-quality level, on whichever scale the encoder
            uses.
        include_audio: Whether to carry the source audio through.
        on_segment: Called as (index, total, segment) after each segment is
            written. Raising from it aborts the cut.

    Returns:
        A CutResult describing what was written.

    Raises:
        CutterError: If the segments are unusable or the source cannot be read.
    """
    if not segments:
        raise CutterError("No segments to export.")
    for earlier, later in zip(segments, segments[1:]):
        if later.start_time < earlier.end_time:
            raise CutterError(
                "Segments overlap; pass them through merge_for_export first so "
                "the joined output does not repeat footage."
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    exported_seconds = 0.0

    if isinstance(video, VideoSource):
        video_path = video.path
        try:
            source = video.open()
        except VideoLoadError as error:
            raise CutterError(str(error)) from error
    else:
        video_path = Path(video)
        try:
            source = av.open(str(video_path))
        except av.FFmpegError as error:
            raise CutterError(f"Could not open {video_path}: {error}") from error

    with source:
        if not source.streams.video:
            raise CutterError(f"{video_path} has no video stream.")
        source_video = source.streams.video[0]
        source_audio = (
            source.streams.audio[0]
            if include_audio and source.streams.audio
            else None
        )
        source_video.thread_type = "AUTO"

        output, out_video, out_audio = _open_output(
            output_path, source_video, source_audio, video_encoder, audio_encoder, quality
        )

        audio_state = _AudioState(out_audio) if out_audio is not None else None
        counters = _Counters()
        frame_rate = source_video.average_rate or Fraction(30, 1)
        elapsed = 0.0

        try:
            for index, segment in enumerate(segments):
                written = _write_segment(
                    source,
                    source_video,
                    source_audio,
                    output,
                    out_video,
                    out_audio,
                    audio_state,
                    counters,
                    frame_rate,
                    segment,
                )
                elapsed += written
                exported_seconds += written

                if on_segment is not None:
                    on_segment(index, len(segments), segment)

            # Encoders buffer; without a flush the reel loses its tail.
            if audio_state is not None:
                audio_state.flush(output, counters)
            for packet in out_video.encode():
                output.mux(packet)
            if out_audio is not None:
                for packet in out_audio.encode():
                    output.mux(packet)
        finally:
            output.close()

    return CutResult(
        output_path=output_path,
        segment_count=len(segments),
        exported_seconds=exported_seconds,
        encode_seconds=time.monotonic() - started,
    )


def _write_segment(
    source,
    source_video,
    source_audio,
    output,
    out_video,
    out_audio,
    audio_state,
    counters,
    frame_rate,
    segment: AppearanceInterval,
) -> float:
    """Encodes one segment into the open reel, returning its real duration."""
    start = segment.start_time
    end = segment.end_time

    # Seeks land on a keyframe at or before the target, so this rewinds to
    # somewhere safely before the cut and decodes forward to the exact frame.
    offset = int(start / source_video.time_base)
    source.seek(offset, stream=source_video)

    streams = [source_video] + ([source_audio] if source_audio is not None else [])
    last_video_time = start

    # Decoding video and audio together yields them interleaved, and audio
    # runs ahead of video. Breaking the loop on the first frame to pass
    # `end` therefore ended the segment on an audio frame and threw away the
    # video still to come -- 7 frames a segment, 339 of an expected 360
    # across the test reel. Each stream is finished independently instead,
    # and the loop stops only once both are done.
    video_done = False
    audio_done = source_audio is None

    for frame in source.decode(*streams):
        if frame.time is None:
            continue

        is_video = isinstance(frame, av.VideoFrame)
        if frame.time >= end:
            if is_video:
                video_done = True
            else:
                audio_done = True
            if video_done and audio_done:
                break
            continue

        if frame.time < start:
            # Decoded only to get here; this is the part of the keyframe
            # gap that the viewer must not see.
            continue
        if (is_video and video_done) or (not is_video and audio_done):
            continue

        if is_video:
            last_video_time = frame.time
            # The source here is yuv444p, which many players cannot decode;
            # converting explicitly rather than relying on the encoder makes
            # the output format a decision rather than a coincidence.
            converted = frame.reformat(
                width=out_video.width, height=out_video.height, format="yuv420p"
            )
            converted.pts = int(
                round(counters.video / frame_rate / VIDEO_TIME_BASE)
            )
            converted.time_base = VIDEO_TIME_BASE
            counters.video += 1
            for packet in out_video.encode(converted):
                output.mux(packet)
        elif out_audio is not None and isinstance(frame, av.AudioFrame):
            audio_state.write(frame, output, counters)

    # What was actually written, which is a frame or so short of the request
    # whenever the segment's end falls between frames.
    return max(0.0, last_video_time - start)
