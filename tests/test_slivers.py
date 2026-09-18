"""Leaving slivers of another shot off the edges of each cut.

A segment is an appearance padded at both ends, and the padding often
reaches across a camera cut, so a clip would open or close on a few frames
of somebody else. The footage here is generated with hard cuts at known
frames, and a steady tone, so it needs no sample video.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from app.video.cutter import SLIVER_FRAMES, cut_segments
from app.video.timeline import AppearanceInterval

FPS = 24
RATE = 48000
# A new shot every 48 frames (2s): shots start at 0s, 2s, 4s, ...
SHOT = 48


def shot_image(n: int) -> np.ndarray:
    """Each shot a different flat colour with a little texture, so a cut is
    a big change and frames within a shot are nearly alike."""
    shot = n // SHOT
    generator = np.random.default_rng(shot)
    base = generator.integers(20, 235, size=3)
    texture = generator.integers(-8, 8, size=(90, 160, 1))
    return np.clip(base + texture, 0, 255).astype(np.uint8)


def write_shots(path: Path, seconds: int, flash_at: int | None = None) -> None:
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=FPS)
    video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
    video.options = {"crf": "18", "g": "12"}
    audio = container.add_stream("aac", rate=RATE)
    audio.layout = "mono"
    for n in range(seconds * FPS):
        image = np.full((90, 160, 3), 255, np.uint8) if n == flash_at else shot_image(n)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts, frame.time_base = n, Fraction(1, FPS)
        for packet in video.encode(frame):
            container.mux(packet)
    total = seconds * RATE
    tone = 0.4 * np.sin(2 * np.pi * 440 * np.arange(total) / RATE)
    for offset in range(0, total, 1024):
        block = np.pad(tone[offset : offset + 1024], (0, max(0, 1024 - len(tone[offset:offset + 1024]))))
        frame = av.AudioFrame.from_ndarray(
            (block * 32767).astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.rate, frame.pts, frame.time_base = RATE, offset, Fraction(1, RATE)
        for packet in audio.encode(frame):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()


@pytest.fixture(scope="module")
def shots(tmp_path_factory):
    path = tmp_path_factory.mktemp("shots") / "shots.mp4"
    write_shots(path, seconds=12)
    return path


def frames_of(path: Path) -> list[np.ndarray]:
    container = av.open(str(path))
    try:
        return [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    finally:
        container.close()


def which_shot(image: np.ndarray) -> int:
    """The shot a decoded frame came from, by its colour."""
    colour = image.reshape(-1, 3).mean(axis=0)
    distances = [np.abs(colour - shot_image(k * SHOT)[0, 0].astype(float)).sum() for k in range(8)]
    return int(np.argmin(distances))


def lengths(path: Path) -> tuple[float, float]:
    container = av.open(str(path))
    try:
        video, audio = container.streams.video[0], container.streams.audio[0]
        n = sum(1 for _ in container.decode(video))
        container.seek(0)
        samples = sum(f.samples for f in container.decode(audio))
        return n / FPS, samples / RATE
    finally:
        container.close()


def test_a_cut_opening_on_the_last_frames_of_the_shot_before_starts_on_the_new_one(shots, tmp_path):
    # Shot 1 starts at 2.0s; the segment starts 5 frames earlier.
    output = tmp_path / "cut.mp4"
    cut_segments(shots, [AppearanceInterval(2.0 - 5 / FPS, 3.5)], output)

    reel = frames_of(output)
    assert which_shot(reel[0]) == 1
    assert {which_shot(f) for f in reel} == {1}


def test_a_cut_closing_on_the_first_frames_of_the_shot_after_ends_before_them(shots, tmp_path):
    # Shot 2 starts at 4.0s; the segment runs 6 frames into it.
    output = tmp_path / "cut.mp4"
    cut_segments(shots, [AppearanceInterval(2.5, 4.0 + 6 / FPS)], output)

    reel = frames_of(output)
    assert which_shot(reel[-1]) == 1
    assert {which_shot(f) for f in reel} == {1}


def test_more_than_a_sliver_of_the_other_shot_is_kept(shots, tmp_path):
    """Past half a second it is a real part of the cut, not a flash."""
    output = tmp_path / "cut.mp4"
    cut_segments(shots, [AppearanceInterval(2.5, 4.0 + (SLIVER_FRAMES + 8) / FPS)], output)

    assert {which_shot(f) for f in frames_of(output)} == {1, 2}


def test_trimming_can_be_turned_off(shots, tmp_path):
    output = tmp_path / "cut.mp4"
    cut_segments(shots, [AppearanceInterval(2.0 - 5 / FPS, 3.5)], output, trim_slivers=False)

    assert which_shot(frames_of(output)[0]) == 0


def test_the_sound_is_trimmed_with_the_picture(shots, tmp_path):
    """Both edges trimmed on three segments: sound and picture still end
    together, so nothing after them moves."""
    output = tmp_path / "cut.mp4"
    segments = [
        AppearanceInterval(2.0 - 5 / FPS, 4.0 + 6 / FPS),
        AppearanceInterval(6.0 - 4 / FPS, 8.0 + 3 / FPS),
        AppearanceInterval(10.0 - 7 / FPS, 11.5),
    ]
    cut_segments(shots, segments, output)

    picture, sound = lengths(output)
    # Three whole 2s shots less the half shot at the end: 5.5s.
    assert picture == pytest.approx(5.5, abs=2 / FPS)
    assert abs(picture - sound) < 0.03


def test_a_one_frame_flash_is_not_taken_for_a_cut(tmp_path):
    """A flash changes the picture twice in two frames, and goes back.
    Read as cuts, it would trim the frames before it off the segment."""
    path = tmp_path / "flash.mp4"
    write_shots(path, seconds=4, flash_at=SHOT + 6)
    output = tmp_path / "cut.mp4"

    cut_segments(path, [AppearanceInterval(2.0, 3.5)], output)

    picture, _ = lengths(output)
    assert picture == pytest.approx(1.5, abs=2 / FPS)


def test_a_short_segment_is_never_trimmed_away(shots, tmp_path):
    output = tmp_path / "cut.mp4"
    cut_segments(shots, [AppearanceInterval(2.0 - 8 / FPS, 2.0 + 6 / FPS)], output)

    assert len(frames_of(output)) >= 6
