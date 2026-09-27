"""Does the sound in a reel still line up with the picture?

The cutter's other tests count frames and measure durations, which is what
let the audio drift behind the picture for three releases: the totals were
plausible at every step while the two timelines slid apart. These check
the thing a viewer actually notices, by cutting footage whose sound and
picture are marked at the same instants and measuring how far apart the
marks end up.

The footage is generated here rather than committed, so unlike the rest of
the cutter's tests these need no sample video and run anywhere.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app.video.cutter import cut_segments
from app.video.timeline import AppearanceInterval

av = pytest.importorskip("av")

FPS = 24
RATE = 48000
BURST_SAMPLES = 2400  # 50ms, long enough not to be confused with a fragment


def write_marked_video(
    path: Path,
    seconds: int,
    fps: int = FPS,
    rate: int | None = RATE,
    width: int = 160,
    height: int = 90,
) -> None:
    """Footage with a white frame and a tone burst on every whole second.

    `rate=None` writes picture only. The other parameters exist so a reel
    can be cut from videos that differ the ways real episodes do.
    """
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    video.options = {"crf": "18", "g": "12"}
    audio = None
    if rate is not None:
        audio = container.add_stream("aac", rate=rate)
        audio.layout = "mono"

    for n in range(seconds * fps):
        shade = 255 if n % fps == 0 else 0
        frame = av.VideoFrame.from_ndarray(
            np.full((height, width, 3), shade, dtype=np.uint8), format="rgb24"
        )
        frame.pts = n
        frame.time_base = Fraction(1, fps)
        for packet in video.encode(frame):
            container.mux(packet)

    if audio is not None:
        burst = int(BURST_SAMPLES * rate / RATE)
        samples = np.zeros(seconds * rate, dtype=np.float32)
        for second in range(seconds):
            tone = np.sin(2 * np.pi * 1000 * np.arange(burst) / rate)
            samples[second * rate : second * rate + burst] = 0.9 * tone

        for offset in range(0, len(samples), 1024):
            block = samples[offset : offset + 1024]
            if len(block) < 1024:
                block = np.pad(block, (0, 1024 - len(block)))
            frame = av.AudioFrame.from_ndarray(
                (block * 32767).astype(np.int16).reshape(1, -1),
                format="s16",
                layout="mono",
            )
            frame.rate = rate
            frame.pts = offset
            frame.time_base = Fraction(1, rate)
            for packet in audio.encode(frame):
                container.mux(packet)

    for packet in video.encode():
        container.mux(packet)
    if audio is not None:
        for packet in audio.encode():
            container.mux(packet)
    container.close()


def _runs(times, gap=0.15):
    runs = []
    for moment in times:
        if runs and moment - runs[-1][-1] <= gap:
            runs[-1].append(moment)
        else:
            runs.append([moment])
    return runs


def read_marks(path: Path):
    """Where the flashes and the tone bursts ended up.

    Returns (flash times, burst times, fragment count). A fragment is a
    single loud audio frame with no burst around it -- audio from beyond a
    cut, carried in because a frame that begins before the cut is written
    whole.
    """
    container = av.open(str(path))
    try:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        flashes, loud = [], []
        for frame in container.decode(video_stream, audio_stream):
            if frame.time is None:
                continue
            if isinstance(frame, av.VideoFrame):
                if float(frame.to_ndarray(format="gray").mean()) > 128:
                    flashes.append(frame.time)
            else:
                raw = frame.to_ndarray()
                # AAC decodes to float, some sources to int16.
                scale = 32767.0 if np.issubdtype(raw.dtype, np.integer) else 1.0
                if np.abs(raw.astype(np.float32)).max() / scale > 0.3:
                    loud.append(frame.time)
    finally:
        container.close()

    bursts = _runs(loud)
    return (
        [run[0] for run in _runs(flashes)],
        [run[0] for run in bursts if run[-1] - run[0] >= 0.03],
        len([run for run in bursts if run[-1] - run[0] < 0.03]),
    )


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    path = tmp_path_factory.mktemp("sync") / "marked.mp4"
    write_marked_video(path, seconds=24)
    return path


def test_the_marked_footage_is_itself_in_sync(marked):
    """The measurement has to be trusted before the cut can be judged by it."""
    flashes, bursts, fragments = read_marks(marked)

    assert len(flashes) == 24
    assert len(bursts) == 24
    assert fragments == 0
    offsets = np.array([b - f for b, f in zip(bursts, flashes)])
    # A burst is timed from the audio frame containing it, so it can read up
    # to one frame early. What matters is that it does not slide.
    assert abs(offsets.mean()) < 0.025


def test_sound_and_picture_do_not_slide_apart_across_cuts(marked, tmp_path):
    """The property the duration tests could not see.

    Each cut keeps whole frames of each kind, and video and audio frames
    are not the same length, so every cut lands slightly differently on
    each. Letting that difference accumulate put a 102-cut reel 3.3
    seconds out. Here the marks must stay together from the first cut to
    the last.
    """
    output = tmp_path / "cut.mp4"
    # Each segment holds exactly one flash and its burst, half a second in.
    segments = [AppearanceInterval(k * 3 + 0.5, k * 3 + 2.0) for k in range(8)]

    cut_segments(marked, segments, output)

    flashes, bursts, _ = read_marks(output)
    assert len(flashes) == 8
    assert len(bursts) == 8

    offsets = np.array([b - f for b, f in zip(bursts, flashes)])

    # Asserted as a spread rather than as a fitted slope. A burst is timed
    # from the audio frame that contains it, so each offset is quantised to
    # 21ms; over eight cuts a single step of that size fits a slope of
    # 2.6ms per cut out of pure quantisation, which would make a slope
    # threshold measure the sampling rather than the sync. If the two
    # timelines were sliding, the offsets would fan out instead.
    assert offsets.max() - offsets.min() < 0.05
    assert abs(offsets[-1] - offsets[0]) < 0.05


def test_every_cut_keeps_its_own_marks(marked, tmp_path):
    """A reel of eight cuts holds eight moments, not seven or nine."""
    output = tmp_path / "cut.mp4"
    segments = [AppearanceInterval(k * 3 + 0.5, k * 3 + 2.0) for k in range(8)]

    cut_segments(marked, segments, output)

    flashes, bursts, _ = read_marks(output)

    assert len(flashes) == len(bursts) == len(segments)


def test_the_slide_does_not_grow_with_the_number_of_cuts(marked, tmp_path):
    """The property that actually failed. Two reels over the same footage,
    one cut four times and one sixteen: if the error accumulated, the
    second would be four times worse."""
    few, many = tmp_path / "few.mp4", tmp_path / "many.mp4"

    cut_segments(
        marked, [AppearanceInterval(k * 6 + 0.5, k * 6 + 5.0) for k in range(4)], few
    )
    cut_segments(
        marked,
        [AppearanceInterval(k * 1.5 + 0.4, k * 1.5 + 1.4) for k in range(16)],
        many,
    )

    def spread(path):
        flashes, bursts, _ = read_marks(path)
        pairs = min(len(flashes), len(bursts))
        offsets = np.array([bursts[i] - flashes[i] for i in range(pairs)])
        return offsets.max() - offsets.min()

    assert spread(many) < spread(few) + 0.05


def test_no_audio_is_carried_across_a_cut(marked, tmp_path):
    """A decoded audio frame that begins before the cut used to be written
    whole, so up to 21ms of the moment the reel had cut away arrived at the
    join as a fragment of the wrong sound. Fifteen of twenty cuts carried
    one. Video has no equivalent: it is cut on frame boundaries, which are
    the timeline's own units."""
    output = tmp_path / "cut.mp4"
    segments = [AppearanceInterval(k * 3 + 0.5, k * 3 + 2.0) for k in range(8)]

    cut_segments(marked, segments, output)

    _, _, fragments = read_marks(output)

    assert fragments == 0


# ------------------------------------------------------- nothing is made up


def write_tone_video(path: Path, seconds: int, rate: int = RATE) -> None:
    """Footage whose sound never stops: a steady tone under every frame.

    Any dropout in a reel cut from it is manufactured, because there is no
    quiet moment in the source for it to have come from.
    """
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=FPS)
    video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
    video.options = {"crf": "23", "g": "12"}
    audio = container.add_stream("aac", rate=rate)
    audio.layout = "mono"

    for n in range(seconds * FPS):
        frame = av.VideoFrame.from_ndarray(
            np.full((90, 160, 3), (n * 7) % 256, dtype=np.uint8), format="rgb24"
        )
        frame.pts = n
        frame.time_base = Fraction(1, FPS)
        for packet in video.encode(frame):
            container.mux(packet)

    total = seconds * rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(total) / rate)
    for offset in range(0, total, 1024):
        block = tone[offset : offset + 1024]
        if len(block) < 1024:
            block = np.pad(block, (0, 1024 - len(block)))
        frame = av.AudioFrame.from_ndarray(
            (block * 32767).astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.rate = rate
        frame.pts = offset
        frame.time_base = Fraction(1, rate)
        for packet in audio.encode(frame):
            container.mux(packet)

    for packet in video.encode():
        container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()


@pytest.fixture(scope="module")
def tone(tmp_path_factory):
    path = tmp_path_factory.mktemp("tone") / "tone.mp4"
    write_tone_video(path, seconds=40)
    return path


# Cuts that end mid-frame, the way real appearance intervals do. Joins that
# happen to fall on frame boundaries hide the fault this guards against.
IRREGULAR = [
    AppearanceInterval(1.013, 3.271),
    AppearanceInterval(4.502, 6.918),
    AppearanceInterval(8.130, 9.407),
    AppearanceInterval(11.66, 14.02),
    AppearanceInterval(15.31, 17.95),
    AppearanceInterval(19.07, 20.33),
    AppearanceInterval(22.84, 25.61),
    AppearanceInterval(27.19, 29.03),
    AppearanceInterval(30.55, 33.88),
    AppearanceInterval(35.02, 37.49),
]


def test_no_silence_is_manufactured_at_the_joins(tone, tmp_path, monkeypatch):
    """Levelling the sound with the picture once meant cutting audio in its
    own frames and filling the shortfall with zeros -- 21ms of dead air at
    98 of 100 joins on the 22-minute footage, heard as a dropout at every
    cut. Every duration and sync check passed, because silence is exactly
    the right length. This counts what was written instead."""
    from app.video import cutter

    made_up = []
    original = cutter._AudioState._silence

    def counting(self, samples):
        made_up.append(samples)
        return original(self, samples)

    monkeypatch.setattr(cutter._AudioState, "_silence", counting)

    cut_segments(tone, IRREGULAR, tmp_path / "cut.mp4")

    assert sum(made_up) == 0


def test_the_sound_does_not_dip_at_any_join(tone, tmp_path):
    """The same fault, measured the way it is heard. The source tone never
    falls quiet, so no stretch of the reel may either."""
    output = tmp_path / "cut.mp4"
    cut_segments(tone, IRREGULAR, output)

    container = av.open(str(output))
    try:
        samples = np.concatenate(
            [
                frame.to_ndarray().astype(np.float32).reshape(-1)
                for frame in container.decode(container.streams.audio[0])
            ]
        )
    finally:
        container.close()

    blocks = samples[: len(samples) // 1024 * 1024].reshape(-1, 1024)
    loudness = np.sqrt((blocks**2).mean(axis=1))
    # The codec's own ramp at the very start and end of the file is not a join.
    interior = loudness[3:-3]

    assert interior.min() > 0.5 * np.median(interior)


def _decoded_audio(path: Path) -> np.ndarray:
    container = av.open(str(path))
    try:
        return np.concatenate(
            [
                frame.to_ndarray().astype(np.float64).reshape(-1)
                for frame in container.decode(container.streams.audio[0])
            ]
        )
    finally:
        container.close()


def test_no_join_clicks(tone, tmp_path):
    """A join set two unrelated moments of sound side by side, and wherever
    the waveform was not near zero at that instant it jumped -- a click.
    On this reel five of the nine joins jumped by 6 to 13 times the largest
    step the tone ever takes, and the encoder smeared each jump into the
    samples around it.

    Measured over the whole reel rather than at predicted join positions:
    an earlier probe guessed where the joins were from the cutter's running
    audio count, which trails the true join by up to a frame's worth of
    buffered samples, and found nothing there.

    The other four joins did not click only by coincidence of this signal:
    440Hz advances exactly a third of a cycle per 24fps frame, so a gap of
    a multiple of three frames lines the phase back up. Real sound has no
    such luck.
    """
    output = tmp_path / "cut.mp4"
    cut_segments(tone, IRREGULAR, output)

    source_steps = np.abs(np.diff(_decoded_audio(tone)))
    reel = _decoded_audio(output)
    # The codec's own ramp at the very start and end of the file is not a join.
    reel_steps = np.abs(np.diff(reel[2048:-2048]))

    assert reel_steps.max() < 2 * source_steps.max()


class _Encoder:
    """Only what _AudioState reads from an output stream."""

    def __init__(self, format_name, layout_name):
        self.format = av.AudioFormat(format_name)
        self.layout = av.AudioLayout(layout_name)
        self.rate = RATE

        class _Context:
            frame_size = 1024

        self.codec_context = _Context()


def _flat(frame, planar, channels):
    data = frame.to_ndarray()
    return data if planar else data.reshape(-1, channels).T


@pytest.mark.parametrize(
    "format_name, layout_name, planar, channels",
    [("fltp", "mono", True, 1), ("s16", "stereo", False, 2)],
)
def test_the_fade_eases_both_ends_and_keeps_every_sample(
    format_name, layout_name, planar, channels
):
    from app.video.cutter import FADE_SECONDS, _AudioState

    state = _AudioState(_Encoder(format_name, layout_name), RATE, Fraction(FPS))
    dtype = np.float32 if planar else np.int16
    full = 0.5 if planar else 16000
    samples = 4800
    shape = (channels, samples) if planar else (1, samples * channels)
    part = av.AudioFrame.from_ndarray(
        np.full(shape, full, dtype=dtype), format=format_name, layout=layout_name
    )

    # Two chunks of one segment, eased where they sit in it.
    first, second = part, av.AudioFrame.from_ndarray(
        np.full(shape, full, dtype=dtype), format=format_name, layout=layout_name
    )
    state._ease(first, 0, 2 * samples)
    state._ease(second, samples, 2 * samples)
    data = np.concatenate(
        [_flat(first, planar, channels), _flat(second, planar, channels)], axis=1
    ).astype(np.float64)
    fade = int(round(FADE_SECONDS * RATE))

    assert data.shape[1] == 2 * samples
    # Each channel eased to near zero at both ends, untouched in between.
    assert np.all(np.abs(data[:, 0]) < 0.01 * full)
    assert np.all(np.abs(data[:, -1]) < 0.01 * full)
    assert np.all(data[:, fade:-fade] == full)


def test_a_segment_shorter_than_two_fades_still_comes_out_whole():
    from app.video.cutter import _AudioState

    state = _AudioState(_Encoder("fltp", "mono"), RATE, Fraction(FPS))
    part = av.AudioFrame.from_ndarray(
        np.full((1, 101), 0.5, dtype=np.float32), format="fltp", layout="mono"
    )

    state._ease(part, 0, 101)
    data = part.to_ndarray()[0]

    assert part.samples == 101
    assert np.isfinite(data).all()
    assert data.max() <= 0.5


def test_a_fade_that_spans_two_chunks_is_one_smooth_fade():
    """A segment's sound is written a second at a time, so a fade can
    straddle two chunks; eased chunk by chunk it must match the fade of the
    whole."""
    from app.video.cutter import FADE_SECONDS, _AudioState

    state = _AudioState(_Encoder("fltp", "mono"), RATE, Fraction(FPS))
    total = 4800
    whole = av.AudioFrame.from_ndarray(np.full((1, total), 0.5, np.float32), format="fltp", layout="mono")
    state._ease(whole, 0, total)
    fade = int(round(FADE_SECONDS * RATE))
    split = fade // 3  # the first chunk ends partway into the opening fade
    pieces = [
        av.AudioFrame.from_ndarray(np.full((1, n), 0.5, np.float32), format="fltp", layout="mono")
        for n in (split, total - split)
    ]
    state._ease(pieces[0], 0, total)
    state._ease(pieces[1], split, total)

    joined = np.concatenate([piece.to_ndarray()[0] for piece in pieces])
    assert np.allclose(joined, whole.to_ndarray()[0])


def test_a_long_cut_s_sound_is_written_a_second_at_a_time(tmp_path, monkeypatch):
    """A 3.5-minute cut of 5.1 sound once became about 1.3 GB of copies, and
    a 27-minute reel of a 43-minute MKV crashed when memory ran out. The
    sound now goes to the encoder in chunks of at most one second."""
    from app.video import cutter

    source = tmp_path / "long.mp4"
    write_tone_video(source, seconds=20)
    largest = []
    original = cutter._AudioState._push

    def recording(self, frame):
        largest.append(frame.samples)
        return original(self, frame)

    monkeypatch.setattr(cutter._AudioState, "_push", recording)

    cut_segments(source, [AppearanceInterval(1.0, 18.0)], tmp_path / "cut.mp4")

    assert max(largest) <= RATE
    assert sum(largest) >= 16 * RATE


def test_sound_that_changes_format_partway_is_buffered_whole():
    """Broadcast AC3 can switch from 5.1 to stereo for a moment and back --
    a 43-minute HDTV MKV does it for 0.128s at 973s. A resampler fixes its
    input on the first frame, and handed the other layout it produced a
    frame with nothing behind it; buffering that crashed the app outright
    (a segmentation fault in AudioFifo.write) and left a half-written reel.
    """
    from app.video.cutter import _AudioState

    state = _AudioState(_Encoder("fltp", "5.1(side)"), RATE, Fraction(FPS))
    state.begin()
    layouts = ["5.1(side)"] * 3 + ["stereo"] * 4 + ["5.1(side)"] * 3
    for k, layout in enumerate(layouts):
        channels = len(av.AudioLayout(layout).channels)
        frame = av.AudioFrame.from_ndarray(
            np.full((channels, 1536), 0.1, np.float32), format="fltp", layout=layout
        )
        # Matroska keeps sound in milliseconds, as the HDTV episode did.
        frame.rate, frame.pts, frame.time_base = RATE, k * 32, Fraction(1, 1000)
        state.write(frame)

    assert state._staging.samples == len(layouts) * 1536
