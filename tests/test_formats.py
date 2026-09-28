"""MKV, WebM and AVI: cut with the sound kept in step, like MP4.

Each container gets the sync tests' marked footage -- a white frame and a
tone burst on every whole second -- and eight one-flash segments are cut
out of it. The reel must hold eight flashes and eight bursts, together, in
exactly the time asked for.

WebM gets two files, because the backlog item this closes named its real
risk: WebM from a browser or a screen recorder has a variable frame rate.
Before the cutter placed such frames by their timestamps, the variable file
here came out at 9.6s of picture instead of 12, its sound up to 1.5s adrift.

Footage is made with PyAV, whose wheels carry every encoder used here, so
none of this needs ffmpeg on the machine.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app.video.cutter import _frame_spacing, cut_clips, cut_segments, probe_clip, Clip
from app.video.loader import SUPPORTED_EXTENSIONS, validate_video_path
from app.video.timeline import AppearanceInterval

av = pytest.importorskip("av")

from tests.test_sync import read_marks, write_marked_video  # noqa: E402

SECONDS = 24
RATE = 48000
# One flash and its burst in each, half a second in.
SEGMENTS = [AppearanceInterval(k * 3 + 0.5, k * 3 + 2.0) for k in range(8)]


def write_container(
    path: Path,
    video_codec: str,
    audio_codec: str,
    frame_times: list[float],
    audio_frame: int = 1024,
    options: dict | None = None,
    clock: Fraction = Fraction(1, 1000),
) -> None:
    """Marked footage with a frame at each of `frame_times`, in seconds.

    Timestamps are written in milliseconds by default, as Matroska and WebM
    keep them, so constant-rate footage here carries the same 41/42ms
    rounding a real MKV of 23.976 fps footage does. AVI counts in frames.
    """
    container = av.open(str(path), mode="w")
    video = container.add_stream(video_codec, rate=24)
    video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
    video.time_base = clock
    # The encoder's own clock too, or it rounds every timestamp to its
    # declared rate and a variable file comes out with frames on top of
    # one another.
    video.codec_context.time_base = clock
    video.options = options or {}
    audio = container.add_stream(audio_codec, rate=RATE)
    audio.layout = "mono"

    for when in frame_times:
        shade = 255 if abs(when - round(when)) < 1e-6 else 0
        frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), shade, np.uint8), format="rgb24")
        frame.pts = int(round(when / clock))
        frame.time_base = clock
        for packet in video.encode(frame):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)

    samples = np.zeros(SECONDS * RATE, np.float32)
    for second in range(SECONDS):
        tone = np.sin(2 * np.pi * 1000 * np.arange(2400) / RATE)
        samples[second * RATE : second * RATE + 2400] = 0.9 * tone
    for offset in range(0, len(samples), audio_frame):
        block = np.pad(samples[offset : offset + audio_frame], (0, max(0, offset + audio_frame - len(samples))))
        frame = av.AudioFrame.from_ndarray(
            (block * 32767).astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.rate, frame.pts, frame.time_base = RATE, offset, Fraction(1, RATE)
        for packet in audio.encode(frame):
            container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()


def even(fps: float) -> list[float]:
    return [n / fps for n in range(int(SECONDS * fps))]


def uneven() -> list[float]:
    """Four seconds at 30 fps, two at 10, in turns -- a screen recording
    that slows down whenever nothing moves. A frame on every whole second."""
    times, moment = [], 0.0
    while moment < SECONDS - 1e-9:
        times.append(round(moment, 6))
        step = 1 / 30 if int(moment) % 6 < 4 else 1 / 10
        following = moment + step
        if int(following + 1e-9) != int(moment + 1e-9):
            following = float(int(moment + 1e-9) + 1)
        moment = following
    return times


VP9 = {"deadline": "realtime", "cpu-used": "8"}

MADE = {
    "mkv, h264 and aac": ("clip.mkv", "libx264", "aac", even(24), 1024, {"g": "12"}),
    "mkv, h264 and opus": ("clip-opus.mkv", "libx264", "libopus", even(24), 960, {"g": "12"}),
    "webm, constant": ("clip.webm", "libvpx-vp9", "libopus", even(24), 960, VP9),
    "webm, variable": ("clip-variable.webm", "libvpx-vp9", "libopus", uneven(), 960, VP9),
    "avi, mpeg4 and mp3": ("clip.avi", "mpeg4", "libmp3lame", even(24), 1152, {"qscale": "3"}, Fraction(1, 24)),
}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    folder = tmp_path_factory.mktemp("formats")
    paths = {}
    for label, (name, video_codec, audio_codec, times, audio_frame, options, *clock) in MADE.items():
        paths[label] = folder / name
        write_container(paths[label], video_codec, audio_codec, times, audio_frame, options, *clock)
    return paths


def test_every_format_this_covers_is_accepted(made):
    for path in made.values():
        assert path.suffix in SUPPORTED_EXTENSIONS
        assert validate_video_path(path) == path


@pytest.mark.parametrize("label", list(MADE))
def test_each_format_cuts_in_step(made, tmp_path, label):
    output = tmp_path / "reel.mp4"

    cut_segments(made[label], SEGMENTS, output)

    flashes, bursts, fragments = read_marks(output)
    assert len(flashes) == len(bursts) == len(SEGMENTS)
    assert fragments == 0
    offsets = np.array([b - f for b, f in zip(bursts, flashes)])
    # As in the sync tests: a burst is timed from the audio frame holding it,
    # so offsets are quantised; sliding apart would fan them out.
    assert offsets.max() - offsets.min() < 0.05
    with av.open(str(output)) as container:
        picture = container.streams.video[0]
        frames = sum(1 for _ in container.decode(picture))
        rate = float(picture.average_rate)
    assert frames / rate == pytest.approx(12.0, abs=1.01 / rate)


def test_a_variable_webm_is_recognised_and_a_constant_one_is_not(made):
    variable = probe_clip(made["webm, variable"])
    constant = probe_clip(made["webm, constant"])

    assert variable.variable is True
    # The rate it mostly runs at, not whatever the file declares.
    assert variable.frame_rate == 30
    assert constant.variable is False


def test_a_reel_can_join_a_variable_webm_to_an_mp4(made, tmp_path):
    """Placed by its timestamps on a 24 fps reel: every mark still lands."""
    mp4 = tmp_path / "episode.mp4"
    write_marked_video(mp4, seconds=SECONDS)
    output = tmp_path / "reel.mp4"

    cut_clips(
        [Clip(mp4, SEGMENTS[:4]), Clip(made["webm, variable"], SEGMENTS[4:])],
        output,
    )

    flashes, bursts, _ = read_marks(output)
    assert len(flashes) == len(bursts) == len(SEGMENTS)
    offsets = np.array([b - f for b, f in zip(bursts, flashes)])
    assert offsets.max() - offsets.min() < 0.05


# ------------------------------------------------- telling uneven from rounded


class _Packet:
    def __init__(self, pts):
        self.pts = pts


class _Source:
    """Just enough of a container for `_frame_spacing` to read timestamps."""

    def __init__(self, stamps):
        self._stamps = stamps

    def demux(self, _stream):
        return [_Packet(stamp) for stamp in self._stamps]

    def seek(self, _offset):
        pass


class _Stream:
    def __init__(self, time_base):
        self.time_base = time_base


def spacing(stamps, time_base=Fraction(1, 1000)):
    return _frame_spacing(_Source(stamps), _Stream(time_base))


def test_millisecond_rounding_is_not_a_variable_rate():
    """23.976 fps in milliseconds is 41 and 42ms in turns -- in the
    22-minute test episode's own finer clock, too, where it was first
    mistaken for unevenness."""
    stamps = [round(n * 1001 / 24) for n in range(2000)]
    variable, typical, _ = spacing(stamps)
    assert variable is False
    fine = [stamp * 90 for stamp in stamps]
    assert spacing(fine, Fraction(1, 90000))[0] is False
    assert typical == pytest.approx(1001 / 24000, rel=1e-3)


def test_one_short_gap_is_not_a_variable_rate():
    """The iPhone clip: one gap 1.7ms short in 981 frames."""
    stamps = [n * 1000 for n in range(981)]
    stamps[230:] = [stamp - 50 for stamp in stamps[230:]]
    assert spacing(stamps, Fraction(1, 30000))[0] is False


def test_a_frame_rate_that_changes_is_variable():
    stamps = [round(moment * 1000) for moment in uneven()]
    variable, typical, longest = spacing(stamps)
    assert variable is True
    assert 1 / typical == pytest.approx(30, rel=0.005)
    assert longest == pytest.approx(0.1)


def test_a_header_that_lies_about_the_rate_is_not_believed(tmp_path):
    """An AVI written with a millisecond clock declares 1000 fps. Trusted,
    that would have made a 1000 fps reel; the frames say 24."""
    path = tmp_path / "lying.avi"
    write_container(path, "mpeg4", "libmp3lame", even(24), 1152, {"qscale": "3"})

    profile = probe_clip(path)

    assert profile.frame_rate == 24


# ------------------------------------------------------ what a reel is saved as


@pytest.mark.parametrize("extension, container", [("mkv", "matroska"), ("mov", "mov")])
def test_a_reel_can_be_saved_as_mkv_or_mov_and_keeps_its_sound_in_step(made, tmp_path, extension, container):
    """The same H.264 and AAC, in another wrapper: nothing else changes."""
    output = tmp_path / f"reel.{extension}"

    cut_segments(made["mkv, h264 and aac"], SEGMENTS, output)

    with av.open(str(output)) as opened:
        assert container in opened.format.name
        assert opened.streams.video[0].codec_context.name == "h264"
        assert opened.streams.audio[0].codec_context.name == "aac"
    flashes, bursts, _ = read_marks(output)
    assert len(flashes) == len(bursts) == len(SEGMENTS)
    offsets = np.array([b - f for b, f in zip(bursts, flashes)])
    assert offsets.max() - offsets.min() < 0.05


def test_a_reel_cannot_be_saved_as_something_that_cannot_hold_it(made, tmp_path):
    """WebM takes neither H.264 nor AAC; better said before the encode than
    after minutes of it."""
    from app.video.cutter import CutterError

    for name in ("reel.webm", "reel.avi", "reel"):
        with pytest.raises(CutterError, match="can be saved as"):
            cut_segments(made["mkv, h264 and aac"], SEGMENTS, tmp_path / name)


def test_a_reel_takes_its_video_s_own_format_where_it_can():
    from app.video.cutter import export_format_for

    assert export_format_for("episode.mkv") == "mkv"
    assert export_format_for("clip.MOV") == "mov"
    assert export_format_for("clip.m4v") == "mp4"
    # A reel cannot be WebM or AVI, so those make MP4s.
    assert export_format_for("recording.webm") == "mp4"
    assert export_format_for("old.avi") == "mp4"


# ------------------------------------------------------------ stress-test finds


def write_wide_pixel_video(path: Path, seconds: int = 4) -> None:
    """720x480 with 32:27 pixels, as a DVD rip is stored, holding a circle:
    squeezed in the stored pixels, round once shown at 16:9."""
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=24)
    video.width, video.height, video.pix_fmt = 720, 480, "yuv420p"
    video.codec_context.sample_aspect_ratio = Fraction(32, 27)
    audio = container.add_stream("aac", rate=RATE)
    audio.layout = "stereo"
    yy, xx = np.mgrid[0:480, 0:720]
    # A circle of radius 150 shown pixels: 150 * 27/32 stored pixels across.
    inside = ((xx - 360) / (150 * 27 / 32)) ** 2 + ((yy - 240) / 150) ** 2 <= 1
    picture = np.where(inside[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
    for n in range(seconds * 24):
        frame = av.VideoFrame.from_ndarray(picture, format="rgb24")
        frame.pts, frame.time_base = n, Fraction(1, 24)
        for packet in video.encode(frame):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    silence = np.zeros((1, seconds * RATE * 2), np.int16)
    for offset in range(0, seconds * RATE, 1024):
        frame = av.AudioFrame.from_ndarray(silence[:, 2 * offset : 2 * (offset + 1024)], format="s16", layout="stereo")
        frame.rate, frame.pts, frame.time_base = RATE, offset, Fraction(1, RATE)
        for packet in audio.encode(frame):
            container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()


def test_a_wide_pixel_video_makes_a_reel_of_the_right_shape(tmp_path):
    """A DVD rip's reel kept its 720x480 but lost the note that its pixels
    are wide, so every player showed it at 3:2 and everyone looked squeezed.
    The reel is now square-pixelled at the size it was meant to be seen."""
    source = tmp_path / "dvd.mkv"
    write_wide_pixel_video(source)
    output = tmp_path / "reel.mp4"

    cut_segments(source, [AppearanceInterval(0.5, 3.0)], output)

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (854, 480)
        assert stream.sample_aspect_ratio in (None, Fraction(1))
        frame = next(container.decode(stream)).to_ndarray(format="gray")
    rows, columns = np.where(frame > 128)
    width, height = columns.max() - columns.min(), rows.max() - rows.min()
    # The circle is round again, to within a couple of pixels.
    assert abs(width - height) <= 4


def test_a_cancelled_export_leaves_nothing_behind(tmp_path):
    """Every cancel in the stress test left its partial reel on disk."""
    from app.video.cutter import partial_path

    source = tmp_path / "clip.mkv"
    write_container(source, "libx264", "aac", even(24), 1024, {"g": "12"})
    output = tmp_path / "reel.mp4"

    class Stop(Exception):
        pass

    def stop_after_the_first_cut(index, total, segment):
        raise Stop()

    with pytest.raises(Stop):
        cut_segments(source, SEGMENTS, output, on_segment=stop_after_the_first_cut)

    assert not output.exists()
    assert not partial_path(output).exists()


def test_a_failed_export_keeps_the_reel_it_would_have_replaced(tmp_path):
    source = tmp_path / "clip.mkv"
    write_container(source, "libx264", "aac", even(24), 1024, {"g": "12"})
    output = tmp_path / "reel.mp4"
    output.write_bytes(b"yesterday's reel")

    def fail(index, total, segment):
        raise RuntimeError("the disk filled up")

    with pytest.raises(RuntimeError):
        cut_segments(source, SEGMENTS, output, on_segment=fail)

    assert output.read_bytes() == b"yesterday's reel"


def test_a_finished_export_leaves_only_the_reel(tmp_path):
    from app.video.cutter import partial_path

    source = tmp_path / "clip.mkv"
    write_container(source, "libx264", "aac", even(24), 1024, {"g": "12"})
    output = tmp_path / "reel.mkv"

    result = cut_segments(source, SEGMENTS[:2], output)

    assert result.output_path == output and output.is_file()
    assert not partial_path(output).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mkv", "reel.mkv"]
