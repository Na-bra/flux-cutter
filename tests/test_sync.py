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


def write_marked_video(path: Path, seconds: int) -> None:
    """Footage with a white frame and a tone burst on every whole second."""
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=FPS)
    video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
    video.options = {"crf": "18", "g": "12"}
    audio = container.add_stream("aac", rate=RATE)
    audio.layout = "mono"

    for n in range(seconds * FPS):
        shade = 255 if n % FPS == 0 else 0
        frame = av.VideoFrame.from_ndarray(
            np.full((90, 160, 3), shade, dtype=np.uint8), format="rgb24"
        )
        frame.pts = n
        frame.time_base = Fraction(1, FPS)
        for packet in video.encode(frame):
            container.mux(packet)

    samples = np.zeros(seconds * RATE, dtype=np.float32)
    for second in range(seconds):
        tone = np.sin(2 * np.pi * 1000 * np.arange(BURST_SAMPLES) / RATE)
        samples[second * RATE : second * RATE + BURST_SAMPLES] = 0.9 * tone

    for offset in range(0, len(samples), 1024):
        block = samples[offset : offset + 1024]
        if len(block) < 1024:
            block = np.pad(block, (0, 1024 - len(block)))
        frame = av.AudioFrame.from_ndarray(
            (block * 32767).astype(np.int16).reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.rate = RATE
        frame.pts = offset
        frame.time_base = Fraction(1, RATE)
        for packet in audio.encode(frame):
            container.mux(packet)

    for packet in video.encode():
        container.mux(packet)
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


def write_tone_video(path: Path, seconds: int) -> None:
    """Footage whose sound never stops: a steady tone under every frame.

    Any dropout in a reel cut from it is manufactured, because there is no
    quiet moment in the source for it to have come from.
    """
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=FPS)
    video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
    video.options = {"crf": "23", "g": "12"}
    audio = container.add_stream("aac", rate=RATE)
    audio.layout = "mono"

    for n in range(seconds * FPS):
        frame = av.VideoFrame.from_ndarray(
            np.full((90, 160, 3), (n * 7) % 256, dtype=np.uint8), format="rgb24"
        )
        frame.pts = n
        frame.time_base = Fraction(1, FPS)
        for packet in video.encode(frame):
            container.mux(packet)

    total = seconds * RATE
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(total) / RATE)
    for offset in range(0, total, 1024):
        block = tone[offset : offset + 1024]
        if len(block) < 1024:
            block = np.pad(block, (0, 1024 - len(block)))
        frame = av.AudioFrame.from_ndarray(
            (block * 32767).astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.rate = RATE
        frame.pts = offset
        frame.time_base = Fraction(1, RATE)
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
