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
import numpy as np

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
    # Seconds written from each clip, in the order the clips were given --
    # including 0.0 for any that had nothing to take.
    clip_seconds: tuple[float, ...] = ()


@dataclass(frozen=True)
class Clip:
    """One video's part of a reel: the file and the spans to take from it."""

    video: Path | VideoSource
    segments: list[AppearanceInterval]


@dataclass(frozen=True)
class ClipProfile:
    """What a video is made of, as far as joining it to others is concerned."""

    path: Path
    frame_rate: Fraction
    width: int
    height: int
    sample_rate: int | None
    layout: str | None
    # True when the frames are not evenly spaced (see `_frame_spacing`).
    # Such a video is placed on the reel by its frames' own timestamps
    # rather than by counting them.
    variable: bool = False
    # The longest gap between two of its frames, in seconds: how long one
    # of its frames can stay on screen, and so how far past a cut its
    # sound has to be read.
    longest_frame: float = 0.0


# How far apart two frame rates may be and still count as one.
#
# Frames are stamped by counting them against the reel's one frame rate,
# and each segment's sound is measured from how many frames it kept, so a
# source at another rate cannot simply be copied across: 25fps footage in a
# 23.976 reel would play 4% slow and take the wrong length of sound with it.
# Such a source is converted instead (`_write_segment`). One part in ten
# thousand is what separates the same nominal rate written two ways
# (24000/1001 against 23.976) from different ones, and within it frames are
# copied one for one: a minute-long segment ends at most 6ms adrift, and
# the next segment starts exact again.
FRAME_RATE_TOLERANCE = 1e-4


# Slivers of another shot at a segment's edges.
#
# A segment is an appearance padded at both ends, and the padding often
# reaches across a camera cut: on the test episode 63-75% of segment edges
# sat within half a second of one, leaving a median 7 frames of the
# neighbouring shot -- a flash of somebody else as each clip opens or
# closes. Up to this many frames past a cut at either edge are left out.
SLIVER_FRAMES = 12
# A cut is a spike: two frames' colour thumbnails (64x36) differ by at least
# CUT_DIFFERENCE on average (0-255), and by CUT_RATIO times more than the
# frames either side of them do. Relative, because a cut between two angles
# of one set can change little -- 20 here, against 0-3 within a shot -- and
# fast motion can change a lot without being one. The frames either side
# must also be steady, which is what keeps a one-frame flash, white and
# straight back, from reading as two cuts. Colour, because two shots of
# similar brightness can differ by 24 in grey and 91 in colour.
CUT_DIFFERENCE = 12.0
CUT_RATIO = 4.0
STEADY_DIFFERENCE = 12.0
# A segment is never trimmed below this many frames.
MIN_KEPT_FRAMES = 12


def _thumb(frame) -> np.ndarray:
    return frame.reformat(width=64, height=36, format="rgb24").to_ndarray().astype(np.float32)


def _cuts(thumbs: list[np.ndarray]) -> list[int]:
    """Indexes where a new shot starts: a big change, with steady frames around it."""
    differences = [0.0] + [
        float(np.abs(thumbs[i] - thumbs[i - 1]).mean()) for i in range(1, len(thumbs))
    ]
    cuts = []
    for i in range(1, len(thumbs)):
        if differences[i] < CUT_DIFFERENCE:
            continue
        before = differences[i - 1] if i >= 2 else 0.0
        after = differences[i + 1] if i + 1 < len(thumbs) else 0.0
        steady = max(before, after)
        if steady <= STEADY_DIFFERENCE and differences[i] >= CUT_RATIO * max(steady, 1.0):
            cuts.append(i)
    return cuts


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

# What a reel can be saved as, by extension. The reel is H.264 picture and
# AAC sound whatever it is saved as, and all three hold both, so choosing
# one changes only the wrapper -- no second encode, nothing slower. WebM
# is not among them: it takes neither, and would mean encoding VP9 and
# Opus instead, which is a different and much slower export. AVI can
# carry H.264 only awkwardly, and nothing plays that better than MP4.
EXPORT_FORMATS = {"mp4": "MP4", "mkv": "MKV", "mov": "MOV"}
DEFAULT_EXPORT_FORMAT = "mp4"


def export_format_for(video: Path | str) -> str:
    """What a reel of this video is saved as unless someone says otherwise.

    Its own format where a reel can be one -- an MKV episode makes an MKV
    reel, so a library kept in MKV stays in MKV -- and MP4 otherwise.
    """
    extension = Path(video).suffix.lower().lstrip(".")
    if extension == "m4v":
        return "mp4"
    return extension if extension in EXPORT_FORMATS else DEFAULT_EXPORT_FORMAT


def _open_output(
    output_path: Path,
    picture: ClipProfile,
    sound: ClipProfile | None,
    video_encoder: str,
    audio_encoder: str,
    quality: int,
):
    """Opens the reel and configures its streams.

    The picture takes its size and rate from `picture`, the sound its rate
    and layout from `sound` -- for a single video both are that video, so
    the reel matches its source exactly as it always has.
    """
    output = av.open(str(output_path), mode="w")

    video = output.add_stream(
        video_encoder,
        rate=picture.frame_rate,
        options=_quality_options(video_encoder, quality),
    )
    video.width = picture.width
    video.height = picture.height
    # yuv420p rather than the source's own format: it is what every player
    # can decode, and the test footage is yuv444p, which many cannot.
    video.pix_fmt = "yuv420p"
    video.codec_context.time_base = VIDEO_TIME_BASE

    audio = None
    if sound is not None:
        audio = output.add_stream(audio_encoder, rate=sound.sample_rate)
        audio.layout = sound.layout
        # An encoder's codec_context.time_base is not populated until it is
        # opened, so timestamps are computed against the sample rate, which
        # is what an audio timebase is anyway.
        audio.time_base = Fraction(1, sound.sample_rate)

    return output, video, audio


# How long each cut's sound takes to ease in and out.
#
# A join puts two unrelated stretches of sound side by side, and wherever
# the waveform is not near zero at that instant it jumps -- which is heard
# as a click. On a steady tone cut ten times, five of the nine joins jumped
# by 6 to 13 times the largest step the tone itself ever takes. Easing each
# side to zero over a few milliseconds removes the jump. 5ms is shorter than
# anything a listener hears as a fade, and it changes no lengths, so the
# picture and sound stay exactly as aligned as before.
FADE_SECONDS = 0.005


@dataclass
class _Counters:
    """How much of the reel has been written, per stream.

    Kept across segments: this is what makes the output one continuous
    timeline rather than three that each restart at zero.
    """

    video: int = 0
    audio: int = 0


class _AudioState:
    """The reel's audio, cut to exactly the span of the picture it goes with.

    AAC encodes fixed-size frames (1024 samples), while decoded frames come
    in whatever size the source used, so they have to be rebuffered. That
    is the easy half.

    The hard half is that video and audio are not cut in the same units.
    The picture a segment keeps is a run of whole video frames, and it is
    on screen from the first of them to the moment the last one ends. Audio
    frames are shorter -- 21.3ms against 41.7ms at 24fps -- and begin and
    end in different places. Two earlier attempts cut the audio in its own
    units and then patched the difference, and both went wrong audibly:

    - Letting each cut keep whole audio frames and levelling the totals
      afterwards kept the timelines in sync but filled every shortfall with
      manufactured silence -- on the 22-minute footage, 21ms of dead air at
      98 of 100 joins.
    - Before that, letting the difference run put a 102-cut reel 3.3
      seconds out.

    So a segment's audio is not cut in audio frames at all. It is buffered
    around the cut, and then exactly the picture's span is taken out of it
    by sample offset: from the first video frame kept, for as many samples
    as the video frames kept last. Both streams then describe the same
    stretch of the source, to the sample, and silence is only ever written
    where the source genuinely has no sound.

    The length is counted from the running video total rather than per
    segment, so rounding to whole samples cannot accumulate either. What
    is taken goes into one buffer that spans the whole reel and is encoded
    in whole frames; a part-frame left over is simply the start of the
    next frame. That does not move anything -- the output is one contiguous
    run of samples, and framing it into 1024s is invisible to the timing.
    """

    def __init__(self, out_audio, rate: int, frame_rate: Fraction):
        self._encoder = out_audio
        self._resampler = None
        self._frame_size = out_audio.codec_context.frame_size or 1024
        self._time_base = Fraction(1, out_audio.rate)
        self._rate = rate
        self._frame_rate = frame_rate
        # One for the whole reel, holding what is due to be encoded.
        self._output = av.AudioFifo()
        # Samples handed to that buffer so far: the running total the next
        # segment's length is measured against.
        self._pushed = 0
        # One per segment, holding everything decoded around the cut, and
        # the source time its first sample came from.
        self._staging: av.AudioFifo | None = None
        self._origin: float | None = None
        # Samples of silence written because the source had no sound there.
        # Kept so a test can tell genuine gaps from manufactured ones.
        self.silence_samples = 0

    def begin(self) -> None:
        """Starts buffering a new segment's audio.

        With a fresh resampler each time. One fixes its input format on the
        first frame it sees, so a reel drawing on two videos with different
        sample rates cannot share one; and one converting between rates
        holds a few samples back, which carried into the next segment would
        be sound from the moment the reel just cut away from.
        """
        self._staging = av.AudioFifo()
        self._origin = None
        self._resampler = av.AudioResampler(
            format=self._encoder.format,
            layout=self._encoder.layout,
            rate=self._encoder.rate,
        )

    def write(self, frame) -> None:
        """Buffers one decoded frame from around the cut.

        Nothing is encoded yet: which of these samples belong to the reel
        depends on which video frames the segment keeps, and that is only
        known once it has been read.
        """
        if self._origin is None and frame.time is not None:
            self._origin = frame.time
        for resampled in self._resampler.resample(frame):
            resampled.pts = None
            self._staging.write(resampled)

    def _silence(self, samples: int):
        frame = av.AudioFrame(
            format=self._encoder.format,
            layout=self._encoder.layout,
            samples=samples,
        )
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.rate = self._rate
        frame.pts = None
        self.silence_samples += samples
        return frame

    def _push(self, frame) -> None:
        self._output.write(frame)
        self._pushed += frame.samples

    def _faded(self, parts):
        """One segment's sound as a single frame, eased in and out at its ends.

        The fade is applied to the whole segment, silence included, so it
        lands at the join whatever the segment is made of. A segment too
        short for two full fades gets two that meet in its middle.
        """
        planar = self._encoder.format.is_planar
        channels = len(self._encoder.layout.channels)
        arrays = []
        for part in parts:
            data = part.to_ndarray()
            arrays.append(data if planar else data.reshape(-1, channels).T)
        data = np.concatenate(arrays, axis=1)
        dtype = data.dtype

        total = data.shape[1]
        length = min(int(round(FADE_SECONDS * self._rate)), total // 2)
        if length > 0:
            # Raised cosine: starts and ends flat, so the fade adds no
            # corner of its own for the ear to catch.
            ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(length) + 0.5) / length)
            gain = np.ones(total)
            gain[:length] = ramp
            gain[total - length :] = ramp[::-1]
            faded = data * gain
            if np.issubdtype(dtype, np.integer):
                faded = np.round(faded)
            data = faded.astype(dtype)

        if not planar:
            data = data.T.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(data),
            format=self._encoder.format.name,
            layout=self._encoder.layout.name,
        )
        frame.rate = self._rate
        frame.pts = None
        return frame

    def finish(self, first_video_time: float | None, output, counters) -> None:
        """Takes the picture's span out of the segment's audio and encodes it.

        Args:
            first_video_time: Source time of the first video frame this
                segment kept, or None if it kept none.
        """
        staging, origin = self._staging, self._origin
        self._staging = None

        wanted = int(round(counters.video * self._rate / float(self._frame_rate)))
        need = wanted - self._pushed

        # Everything this segment contributes, in order, so it can be faded
        # as one piece before it joins the reel.
        parts = []
        if need > 0:
            if first_video_time is None or origin is None or staging is None:
                # A segment with picture and no sound at all.
                parts.append(self._silence(need))
            else:
                offset = int(round((first_video_time - origin) * self._rate))
                if offset > 0:
                    # Audio decoded ahead of the first frame kept.
                    staging.read(offset, partial=True)
                elif offset < 0:
                    # The source's sound starts after its picture does.
                    gap = min(-offset, need)
                    parts.append(self._silence(gap))
                    need -= gap
                if need > 0:
                    taken = staging.read(need, partial=True)
                    got = 0
                    if taken is not None:
                        parts.append(taken)
                        got = taken.samples
                    if got < need:
                        # The source's sound ends before its picture does.
                        parts.append(self._silence(need - got))
        if parts:
            self._push(self._faded(parts))

        self._drain(output, counters)

    def _drain(self, output, counters) -> None:
        while self._output.samples >= self._frame_size:
            self._emit(self._output.read(self._frame_size), output, counters)

    def _emit(self, chunk, output, counters) -> None:
        chunk.pts = counters.audio
        chunk.time_base = self._time_base
        counters.audio += chunk.samples
        for packet in self._encoder.encode(chunk):
            output.mux(packet)

    def flush(self, output, counters) -> None:
        """Encodes the last part-frame, once no more segments are coming."""
        chunk = self._output.read(self._output.samples, partial=True) if self._output.samples else None
        if chunk is not None:
            self._emit(chunk, output, counters)


def cut_segments(
    video: Path | VideoSource,
    segments: list[AppearanceInterval],
    output_path: Path,
    video_encoder: str = "libx264",
    audio_encoder: str = "aac",
    quality: int = 20,
    include_audio: bool = True,
    on_segment: Callable[[int, int, AppearanceInterval], None] | None = None,
    trim_slivers: bool = True,
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
    return cut_clips(
        [Clip(video, segments)],
        output_path,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        quality=quality,
        include_audio=include_audio,
        on_segment=on_segment,
        trim_slivers=trim_slivers,
    )


def _open_source(video: Path | VideoSource):
    """The path to report and an open container, however the video was given."""
    if isinstance(video, VideoSource):
        try:
            return video.path, video.open()
        except VideoLoadError as error:
            raise CutterError(str(error)) from error
    path = Path(video)
    try:
        return path, av.open(str(path))
    except av.FFmpegError as error:
        raise CutterError(f"Could not open {path}: {error}") from error


def probe_clip(video: Path | VideoSource, include_audio: bool = True) -> ClipProfile:
    """Reads what joining this video to others depends on, without decoding.

    Raises:
        CutterError: If the video cannot be opened or has no picture.
    """
    path, source = _open_source(video)
    with source:
        if not source.streams.video:
            raise CutterError(f"{path} has no video stream.")
        picture = source.streams.video[0]
        sound = (
            source.streams.audio[0]
            if include_audio and source.streams.audio
            else None
        )
        # Read before the stream is demuxed: the header's own figures.
        width, height = picture.width, picture.height
        declared = picture.average_rate
        sample_rate = sound.rate if sound is not None else None
        layout = sound.layout.name if sound is not None else None
        variable, typical, longest = _frame_spacing(source, picture)
        frame_rate = declared or Fraction(30, 1)
        measured = _usual_rate(typical) if typical else None
        if measured is not None and (
            variable or abs(float(frame_rate) - float(measured)) > 0.05 * float(measured)
        ):
            # A variable video's declared rate is whatever its writer
            # guessed, and a header can simply be wrong -- an AVI written
            # with a millisecond clock declared 1000 fps. Where the header
            # and the frames disagree, the frames win: the reel takes the
            # rate the video mostly runs at.
            frame_rate = measured
        return ClipProfile(
            path=path,
            frame_rate=frame_rate,
            width=width,
            height=height,
            sample_rate=sample_rate,
            layout=layout,
            variable=variable,
            longest_frame=longest,
        )


# A millisecond either way. Timestamps rounded to the millisecond are often
# written into a far finer clock (the 22-minute test episode's are), so the
# stream's own clock says nothing about how exact they are.
ROUNDING_SECONDS = 0.0011

# The rates video is made at. A variable video's usual rate is snapped to
# the nearest of these when it is within half a percent of one: measured
# from millisecond timestamps, 30 fps comes out a hair either side of it,
# and a reel at 30.3 fps plays each cut 1% short.
STANDARD_RATES = (
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
    Fraction(30), Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60),
)


def _usual_rate(gap: float) -> Fraction:
    rate = 1.0 / gap
    nearest = min(STANDARD_RATES, key=lambda standard: abs(float(standard) - rate))
    if abs(float(nearest) - rate) <= 0.005 * rate:
        return nearest
    return Fraction(rate).limit_denominator(1001)


# Variable frame rate.
#
# The cutter copies frames one for one, or converts a constant rate by
# counting (see `_write_segment`); both assume the frames are evenly
# spaced. WebM from a browser or a screen recorder usually is not: thirty
# frames a second while something moves, a handful while nothing does.
# Counted as if even, the test file -- 30 fps and 10 fps in turns -- came
# out at 9.6s of picture instead of 12, with its sound up to 1.5s adrift.
#
# So the spacing is read from the file first, which is cheap because it
# only demuxes: every frame's timestamp, no pictures decoded -- 0.18s for
# the 32,508 frames of the 22-minute test episode.
#
# The question is not whether the gaps are all equal but whether counting
# would put a frame in the wrong place: a video is variable when some frame
# sits more than half a frame from where even spacing, first frame to last,
# says it should be. Gaps that merely wobble do not qualify. Millisecond
# timestamps put 23.976 fps footage 41 and 42ms apart in turn (the test
# episode, Matroska, WebM); the iPhone clip has one gap 1.7ms short in 981
# frames. Neither is ever more than a millisecond or two off the grid, and
# both keep the path they have always taken.
def _frame_spacing(source, picture) -> tuple[bool, float, float]:
    """(variable, usual gap, longest gap) between frames, in seconds.

    The usual gap is the average of the gaps near the most common one --
    not the most common gap itself, which millisecond rounding biases: at
    30 fps the gaps are 33 and 34ms, and 33ms alone reads as 30.3 fps.
    """
    try:
        stamps = sorted(
            packet.pts for packet in source.demux(picture) if packet.pts is not None
        )
    except av.FFmpegError:
        return False, 0.0, 0.0
    finally:
        source.seek(0)
    if len(stamps) < 3:
        return False, 0.0, 0.0
    tick = float(picture.time_base)
    times = [(stamp - stamps[0]) * tick for stamp in stamps]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    even = times[-1] / (len(times) - 1)
    if even <= 0:
        return False, 0.0, 0.0
    off_grid = max(abs(moment - index * even) for index, moment in enumerate(times))
    usual = sorted(gaps)[len(gaps) // 2]
    regular = [gap for gap in gaps if abs(gap - usual) <= ROUNDING_SECONDS + usual / 100]
    typical = sum(regular) / len(regular) if regular and usual > 0 else even
    return off_grid > even / 2, typical, max(gaps)


def same_frame_rate(first: Fraction, second: Fraction) -> bool:
    """Whether two videos can share a reel's timeline. See FRAME_RATE_TOLERANCE."""
    return abs(float(first) - float(second)) <= FRAME_RATE_TOLERANCE * float(first)


def cut_clips(
    clips: list[Clip],
    output_path: Path,
    video_encoder: str = "libx264",
    audio_encoder: str = "aac",
    quality: int = 20,
    include_audio: bool = True,
    on_segment: Callable[[int, int, AppearanceInterval], None] | None = None,
    on_clip: Callable[[int, int, Path], None] | None = None,
    trim_slivers: bool = True,
) -> CutResult:
    """Cuts spans out of one or more videos into a single joined reel.

    Every video is read before anything is written, because a reel that
    fails halfway through a season is minutes of encoding thrown away:

    - **One frame rate.** The reel takes the first video's. A video at
      another rate is converted: each of its frames is shown for as many of
      the reel's frames as it covers, repeated or dropped, so it plays at
      its own speed with its own sound. See FRAME_RATE_TOLERANCE.
    - **One picture size.** The reel takes the first video's size. A video
      of another shape is scaled to fit inside it with black bars, never
      stretched; one of the same shape is simply scaled.
    - **One sound format.** The reel takes the first video that has sound;
      every other is converted to it. A video with no sound contributes
      silence of exactly its picture's length, so nothing after it moves.

    Timestamps, the sync of each segment's sound to its picture, and the
    fade at each join are the same across videos as within one: the
    counters run from the first segment of the first video to the last of
    the last.

    Args:
        clips: The videos, in reel order, each with its segments in order.
        on_segment: Called as (index, total, segment) after each segment,
            counting across the whole reel. Raising from it aborts the cut.
        on_clip: Called as (index, total, path) as each video is started.

    Raises:
        CutterError: If there is nothing to cut, segments overlap, a video
            cannot be read, or the videos cannot share one reel.
    """
    given = list(clips)
    clips = [clip for clip in given if clip.segments]
    if not clips:
        raise CutterError("No segments to export.")
    for clip in clips:
        for earlier, later in zip(clip.segments, clip.segments[1:]):
            if later.start_time < earlier.end_time:
                raise CutterError(
                    "Segments overlap; pass them through merge_for_export first so "
                    "the joined output does not repeat footage."
                )

    profiles = [probe_clip(clip.video, include_audio) for clip in clips]
    picture = profiles[0]
    sound = next((p for p in profiles if p.sample_rate is not None), None)

    output_path = Path(output_path)
    # The wrapper is chosen from the extension, so an extension that cannot
    # hold H.264 and AAC would fail only once the encode had started -- or,
    # for a typo, write a file nothing opens.
    if output_path.suffix.lower().lstrip(".") not in EXPORT_FORMATS:
        raise CutterError(
            f"A reel can be saved as {', '.join('.' + f for f in EXPORT_FORMATS)}, "
            f"not {output_path.suffix or 'a file with no extension'}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    exported_seconds = 0.0
    total = sum(len(clip.segments) for clip in clips)
    index = 0
    per_clip = {id(clip): 0.0 for clip in given}

    output, out_video, out_audio = _open_output(
        output_path, picture, sound, video_encoder, audio_encoder, quality
    )
    counters = _Counters()
    frame_rate = picture.frame_rate
    audio_state = (
        _AudioState(out_audio, out_audio.rate, frame_rate)
        if out_audio is not None
        else None
    )

    try:
        for clip_index, (clip, profile) in enumerate(zip(clips, profiles)):
            if on_clip is not None:
                on_clip(clip_index, len(clips), profile.path)
            path, source = _open_source(clip.video)
            with source:
                source_video = source.streams.video[0]
                source_audio = (
                    source.streams.audio[0]
                    if out_audio is not None and source.streams.audio
                    else None
                )
                source_video.thread_type = "AUTO"
                # None when this video's frames can be copied one for one,
                # or when they are placed by their timestamps instead.
                source_rate = (
                    None
                    if profile.variable or same_frame_rate(frame_rate, profile.frame_rate)
                    else profile.frame_rate
                )

                for segment in clip.segments:
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
                        source_rate,
                        trim_slivers,
                        profile.longest_frame if profile.variable else None,
                    )
                    exported_seconds += written
                    per_clip[id(clip)] += written
                    if on_segment is not None:
                        on_segment(index, total, segment)
                    index += 1

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
        segment_count=total,
        exported_seconds=exported_seconds,
        encode_seconds=time.monotonic() - started,
        clip_seconds=tuple(per_clip[id(clip)] for clip in given),
    )


def _fit_frame(frame, width: int, height: int):
    """A decoded frame in the reel's size and pixel format.

    The source here is yuv444p, which many players cannot decode; converting
    explicitly rather than relying on the encoder makes the output format a
    decision rather than a coincidence.

    A frame of another shape -- a 4:3 episode in a 16:9 reel -- is scaled to
    fit and centred on black rather than stretched.
    """
    if frame.width * height == frame.height * width:
        return frame.reformat(width=width, height=height, format="yuv420p")

    scale = min(width / frame.width, height / frame.height)
    fit_width = max(2, min(width, int(round(frame.width * scale))))
    fit_height = max(2, min(height, int(round(frame.height * scale))))
    scaled = frame.reformat(width=fit_width, height=fit_height, format="rgb24")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - fit_height) // 2
    left = (width - fit_width) // 2
    canvas[top : top + fit_height, left : left + fit_width] = scaled.to_ndarray()
    return av.VideoFrame.from_ndarray(canvas, format="rgb24").reformat(
        format="yuv420p"
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
    source_rate: Fraction | None = None,
    trim_slivers: bool = True,
    timed: float | None = None,
) -> float:
    """Encodes one segment into the open reel, returning its real duration.

    Args:
        frame_rate: The reel's frame rate.
        source_rate: This video's frame rate when it differs from the
            reel's, in which case its frames are repeated or dropped to fit.
        timed: Set for a video whose frames are unevenly spaced, to the
            longest gap between two of them. Each frame then stays on
            screen until the next one's timestamp, as a player shows it,
            rather than for a counted share of the reel.
        trim_slivers: Leave out up to SLIVER_FRAMES of another shot at
            either edge. Frames are held back that many deep to decide the
            end; the sound follows the frames kept, since it is taken from
            the first one written for as long as they last.
    """
    start = segment.start_time
    end = segment.end_time

    # Seeks land on a keyframe at or before the target, so this rewinds to
    # somewhere safely before the cut and decodes forward to the exact frame.
    offset = int(start / source_video.time_base)
    source.seek(offset, stream=source_video)

    streams = [source_video] + ([source_audio] if source_audio is not None else [])
    last_video_time = start
    first_video_time = None
    if audio_state is not None:
        audio_state.begin()

    # The picture a segment keeps is on screen until its last frame ends,
    # which can be up to one frame past `end`. Audio is read that far so the
    # span can be taken exactly, rather than stopping at `end` and coming
    # up short of the picture.
    # With a converted source it is the longer of the two frames: the last
    # source frame can cover more of the reel than one reel frame does, and
    # sound read short of it would be padded with silence.
    longest_frame = 1.0 / float(
        min(frame_rate, source_rate) if source_rate is not None else frame_rate
    )
    if timed is not None:
        longest_frame = max(longest_frame, timed)
    audio_stop = end + longest_frame
    # For a converted source: frames read this segment, and reel frames
    # written for them.
    read_here = 0
    written_here = 0
    # Frames decoded but not yet written, with their thumbnails, held back
    # so a sliver of another shot can still be left out at either edge.
    pending: list = []
    start_settled = not trim_slivers
    written_frames = 0

    # Decoding video and audio together yields them interleaved, and audio
    # runs ahead of video. Breaking the loop on the first frame to pass
    # `end` therefore ended the segment on an audio frame and threw away the
    # video still to come -- 7 frames a segment, 339 of an expected 360
    # across the test reel. Each stream is finished independently instead,
    # and the loop stops only once both are done.
    video_done = False
    audio_done = source_audio is None

    # For a timed source: the frame decoded last, which cannot be written
    # until the next one says how long it stays on screen.
    held = None
    last_gap = 1.0 / float(frame_rate)

    def put(converted, copies: int) -> None:
        nonlocal written_here
        written_here += copies
        for _ in range(copies):
            converted.pts = int(round(counters.video / frame_rate / VIDEO_TIME_BASE))
            converted.time_base = VIDEO_TIME_BASE
            counters.video += 1
            for packet in out_video.encode(converted):
                output.mux(packet)

    def slots_before(moment: float) -> int:
        """Reel frames that start before `moment`, counted from the first."""
        return int(round((moment - first_video_time) * float(frame_rate)))

    def emit(frame) -> int:
        """Writes one source frame to the reel, as many times as it covers."""
        nonlocal first_video_time, last_video_time, read_here, written_here, held, last_gap
        if first_video_time is None:
            first_video_time = frame.time
        if timed is not None:
            if frame.time > last_video_time:
                last_gap = frame.time - last_video_time
            last_video_time = frame.time
            converted = _fit_frame(frame, out_video.width, out_video.height)
            # The frame before this one was on screen until now. A frame
            # due in the same reel slot as the one after it is dropped.
            if held is not None:
                put(held, max(0, slots_before(frame.time) - written_here))
            held = converted
            return 1
        last_video_time = frame.time
        converted = _fit_frame(frame, out_video.width, out_video.height)
        if source_rate is None:
            copies = 1
        else:
            # The reel frames that start while this source frame is on
            # screen: every k with k/reel < read/source. Counted in whole
            # frames with exact fractions, so a long segment cannot
            # drift the way summing float timestamps would.
            read_here += 1
            due = -(-(read_here * frame_rate) // source_rate)
            copies = int(due) - written_here
            written_here += copies
        for _ in range(copies):
            converted.pts = int(round(counters.video / frame_rate / VIDEO_TIME_BASE))
            converted.time_base = VIDEO_TIME_BASE
            counters.video += 1
            for packet in out_video.encode(converted):
                output.mux(packet)
        return 1

    for frame in source.decode(*streams):
        if frame.time is None:
            continue

        is_video = isinstance(frame, av.VideoFrame)
        stop = end if is_video else audio_stop
        if frame.time >= stop:
            if is_video:
                video_done = True
            else:
                audio_done = True
            if video_done and audio_done:
                break
            continue

        if is_video and frame.time < start:
            # Decoded only to get here; this is the part of the keyframe
            # gap that the viewer must not see.
            continue
        if not is_video and frame.time + frame.samples / float(frame.rate) <= start:
            # Audio that ends before the cut. A frame straddling the start
            # is kept, because part of it belongs to the first picture.
            continue
        if (is_video and video_done) or (not is_video and audio_done):
            continue

        if is_video:
            if not trim_slivers:
                written_frames += emit(frame)
                continue
            pending.append((frame, _thumb(frame)))
            if not start_settled:
                if len(pending) <= SLIVER_FRAMES + 1:
                    continue
                # The last cut among the opening frames: everything before it
                # is the tail of the shot before.
                opening = [c for c in _cuts([t for _, t in pending]) if c <= SLIVER_FRAMES]
                if opening:
                    del pending[: opening[-1]]
                start_settled = True
            while len(pending) > SLIVER_FRAMES + 1:
                written_frames += emit(pending.pop(0)[0])
        elif out_audio is not None and isinstance(frame, av.AudioFrame):
            audio_state.write(frame)

    if pending:
        thumbs = [t for _, t in pending]
        cuts = _cuts(thumbs)
        if not start_settled:
            opening = [c for c in cuts if c <= SLIVER_FRAMES]
            if opening and len(pending) - opening[-1] >= MIN_KEPT_FRAMES:
                del pending[: opening[-1]]
                cuts = [c - opening[-1] for c in cuts if c > opening[-1]]
        # The first cut among the closing frames: everything from it on is
        # the head of the shot after.
        closing = [c for c in cuts if len(pending) - c <= SLIVER_FRAMES]
        if closing and written_frames + closing[0] >= MIN_KEPT_FRAMES:
            del pending[closing[0] :]
        for kept, _ in pending:
            written_frames += emit(kept)

    # A timed source's last frame lasts as long as the gap before it did:
    # nothing after it in this segment says otherwise.
    if held is not None:
        put(held, max(0 if written_here else 1, slots_before(last_video_time + last_gap) - written_here))

    if audio_state is not None:
        audio_state.finish(first_video_time, output, counters)

    # What was actually written, which is a frame or so short of the request
    # whenever the segment's end falls between frames.
    return max(0.0, last_video_time - (first_video_time if first_video_time is not None else start))
