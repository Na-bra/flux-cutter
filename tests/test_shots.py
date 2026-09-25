"""Shots: where one camera shot ends and the next begins.

Most of these feed the detector small synthetic frames -- a "scene" is a
smooth pattern in two colours of its own, and frames within it drift,
flicker and brighten slightly the way footage does -- so the rules are
tested without a video. Two tests read real video through the frame
pipeline: one made here with known cuts, and the sample footage when it
is present.
"""

from fractions import Fraction

import cv2
import numpy as np
import pytest

from app.video.shots import (
    SHOT_SAMPLE_INTERVAL,
    Shot,
    ShotDetector,
    ShotSettings,
    describe,
    detect_shots,
    shots_of_video,
)
from tests.conftest import TEST_VIDEO

HEIGHT, WIDTH = 90, 160
STEP = 0.5  # seconds between samples, as a scan samples

# Two colours per scene, far apart in hue, so scenes differ as shots do.
PALETTES = [
    ((200, 40, 40), (240, 200, 60)),
    ((30, 90, 200), (20, 160, 90)),
    ((150, 60, 180), (230, 230, 230)),
    ((40, 40, 40), (90, 200, 220)),
    ((220, 120, 20), (60, 30, 120)),
]


def scene(number: int):
    """A function giving frame k of scene `number`: a smooth two-colour
    pattern, drifting two pixels a sample, with a little noise and flicker."""
    rng = np.random.default_rng(number)
    field = cv2.resize(rng.random((9, 16)).astype(np.float32), (WIDTH * 2, HEIGHT), interpolation=cv2.INTER_CUBIC)
    first, second = (np.array(colour, np.float32) for colour in PALETTES[number % len(PALETTES)])

    def frame(k: int, brightness: float = 1.0) -> np.ndarray:
        shifted = np.clip(field[:, 2 * k : 2 * k + WIDTH], 0, 1)[..., None]
        image = first * shifted + second * (1 - shifted)
        noise = np.random.default_rng(1000 * number + k).normal(0, 3, image.shape)
        flicker = 1 + 0.03 * np.sin(k)
        return np.clip(image * brightness * flicker + noise, 0, 255).astype(np.uint8)

    return frame


def samples(*runs, start: float = 0.0, brightness: float = 1.0):
    """(timestamp, frame) for runs of (scene number, how many samples)."""
    out, moment = [], start
    for number, count in runs:
        make = scene(number)
        for k in range(count):
            out.append((round(moment, 3), make(k, brightness)))
            moment += STEP
    return out


def starts(shots):
    return [shot.start_time for shot in shots]


# ------------------------------------------------------------------ cuts


def test_footage_with_no_cut_is_one_shot():
    shots = detect_shots(samples((0, 20)), video_duration=10.0)

    assert len(shots) == 1
    assert (shots[0].start_time, shots[0].end_time) == (0.0, 10.0)


def test_one_cut_makes_two_shots_at_the_cut():
    shots = detect_shots(samples((0, 10), (1, 10)), video_duration=10.0)

    assert starts(shots) == [0.0, 5.0]
    # The cut happened after the last sample of the first shot.
    assert shots[1].earliest_start == 4.5
    assert shots[1].change > ShotSettings().threshold


def test_several_cuts_make_several_shots():
    shots = detect_shots(samples((0, 6), (1, 6), (2, 6), (3, 6)), video_duration=12.0)

    assert starts(shots) == [0.0, 3.0, 6.0, 9.0]


def test_drift_noise_and_flicker_within_a_shot_are_not_cuts():
    """A pan, sensor noise and a light dipping by a tenth: still one shot."""
    make = scene(2)
    frames = [(k * STEP, make(k, brightness=0.9 if k % 5 == 0 else 1.0)) for k in range(30)]

    assert len(detect_shots(frames)) == 1


def test_a_cut_between_two_dark_scenes_is_still_a_cut():
    """At night everything is near black, so a real cut moves the raw pixel
    values very little; the picture is compared with its brightness and
    contrast taken out."""
    shots = detect_shots(samples((0, 10), (1, 10), brightness=0.18), video_duration=10.0)

    assert starts(shots) == [0.0, 5.0]


def test_rapid_cutting_keeps_every_cut():
    """A montage of one-second shots in wholly different colours: every
    cut is big enough to be one whatever surrounds it."""
    runs = [(n % 5, 2) for n in range(8)]
    shots = detect_shots(samples(*runs), video_duration=8.0)

    assert starts(shots) == [float(n) for n in range(8)]


# Angles on one set: the same walls and skin in every shot, and a colour of
# its own covering a quarter of the frame. Cuts between them change the
# colour mix by 0.40-0.46 -- above the threshold, below the unconditional
# 0.5 -- so whether they are found rests on standing out from their
# neighbours.
SET_COLOURS = ((90, 70, 60), (200, 170, 140))
ACCENTS = ((40, 60, 150), (150, 40, 40), (40, 130, 60), (170, 150, 40))


def set_angle(number: int):
    rng = np.random.default_rng(50 + number)
    field = cv2.resize(rng.random((9, 16)).astype(np.float32), (WIDTH * 2, HEIGHT), interpolation=cv2.INTER_CUBIC)
    accent = cv2.resize(rng.random((9, 16)).astype(np.float32), (WIDTH * 2, HEIGHT), interpolation=cv2.INTER_CUBIC)
    wall, skin, own = (np.array(c, np.float32) for c in (*SET_COLOURS, ACCENTS[number % len(ACCENTS)]))

    def frame(k: int) -> np.ndarray:
        blend = np.clip(field[:, 2 * k : 2 * k + WIDTH], 0, 1)[..., None]
        patch = (accent[:, 2 * k : 2 * k + WIDTH] > 0.75)[..., None]
        image = np.where(patch, own, wall * blend + skin * (1 - blend))
        noise = np.random.default_rng(k + 99 * number).normal(0, 3, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    return frame


def test_cuts_among_cuts_are_found_by_the_quieter_neighbours():
    """The fan edit's problem: shots of one and two samples in turn, so two
    of every three gaps are cuts. Judged against the median of its
    neighbours -- a cut -- no cut stands out, and 3 of these 11 shots were
    found; judged against the quieter neighbours, all 11 are."""
    frames, moment, expected = [], 0.0, []
    for number, length in enumerate([2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]):
        expected.append(round(moment, 3))
        make = set_angle(number)
        for k in range(length):
            frames.append((round(moment, 3), make(k)))
            moment += STEP

    shots = detect_shots(frames, video_duration=moment, settings=ShotSettings(min_shot_seconds=0))

    assert starts(shots) == expected


# ------------------------------------------------------ the shortest shots


def test_a_flash_is_not_a_shot_of_its_own():
    """A jump away and straight back: both cuts go, and the footage around
    it stays one shot."""
    make_a, make_b = scene(0), scene(1)
    frames = [(k * STEP, make_a(k)) for k in range(10)]
    frames += [(5.0, make_b(0))]
    frames += [(5.5 + k * STEP, make_a(10 + k)) for k in range(10)]

    assert len(detect_shots(frames, video_duration=10.5)) == 1


def test_a_brief_shot_is_merged_below_the_minimum_and_kept_without_one():
    one_sample_between = samples((0, 10), (1, 1), (2, 10))

    merged = detect_shots(one_sample_between, video_duration=10.5)
    kept = detect_shots(
        one_sample_between, video_duration=10.5, settings=ShotSettings(min_shot_seconds=0)
    )

    assert len(merged) == 2
    assert starts(kept) == [0.0, 5.0, 5.5]


# -------------------------------------------------------------- the timeline


def test_shots_tile_the_video_in_order():
    shots = detect_shots(samples((0, 7), (1, 5), (2, 9)), video_duration=10.9)

    assert shots[0].start_time == 0.0
    assert shots[-1].end_time == 10.9
    assert [s.index for s in shots] == list(range(len(shots)))
    for earlier, later in zip(shots, shots[1:]):
        assert earlier.end_time == later.start_time
        assert earlier.start_time < later.start_time
        assert later.earliest_start < later.start_time


def test_the_first_shot_starts_at_zero_even_when_the_footage_does_not():
    """An MP4 with a start offset: its first sample is not at 0, but the
    video is, and the first shot covers it."""
    shots = detect_shots(samples((0, 6), (1, 6), start=0.042), video_duration=6.1)

    assert shots[0].start_time == 0.0
    assert starts(shots)[1] == pytest.approx(3.042)


def test_without_a_duration_the_last_shot_ends_at_the_last_sample():
    shots = detect_shots(samples((0, 6), (1, 6)))

    assert shots[-1].end_time == 5.5


def test_boundaries_are_the_samples_own_timestamps():
    """Never a sample count times an interval: frames that arrive unevenly
    -- variable frame rate, a skipped frame -- put the boundary where the
    new shot's first sample really was."""
    uneven = [0.0, 0.52, 0.98, 1.61, 2.04, 2.49, 3.13, 3.5, 4.07, 4.55]
    first, second = scene(0), scene(1)
    frames = [(t, first(k)) for k, t in enumerate(uneven[:5])]
    frames += [(t, second(k)) for k, t in enumerate(uneven[5:])]

    shots = detect_shots(frames, video_duration=5.0)

    assert starts(shots) == [0.0, 2.49]
    assert shots[1].earliest_start == 2.04


def test_samples_out_of_order_are_refused():
    detector = ShotDetector()
    make = scene(0)
    detector.add(1.0, make(0))

    with pytest.raises(ValueError, match="time order"):
        detector.add(0.5, make(1))


# ----------------------------------------------------------- edge of nothing


def test_no_samples_and_no_duration_is_no_shots():
    assert detect_shots([]) == []


def test_no_samples_with_a_duration_is_one_shot_of_it():
    assert detect_shots([], video_duration=4.0) == [Shot(index=0, start_time=0.0, end_time=4.0)]


def test_a_single_sample_is_one_shot():
    shots = detect_shots(samples((0, 1)), video_duration=0.4)

    assert len(shots) == 1
    assert (shots[0].start_time, shots[0].end_time) == (0.0, 0.4)


def test_switched_off_a_video_is_one_shot():
    shots = detect_shots(
        samples((0, 6), (1, 6)), video_duration=6.0, settings=ShotSettings(enabled=False)
    )

    assert len(shots) == 1
    assert shots[0].end_time == 6.0


def test_a_higher_threshold_finds_fewer_cuts():
    frames = samples(*[(n, 4) for n in range(5)])
    default = detect_shots(frames, settings=ShotSettings(min_shot_seconds=0))
    strict = detect_shots(frames, settings=ShotSettings(threshold=0.99, min_shot_seconds=0))

    assert len(strict) <= len(default)


def test_shots_can_be_read_as_text():
    shots = detect_shots(samples((0, 10), (1, 10)), video_duration=10.0)

    assert describe(shots) == [
        "Shot 1: 00:00.00 → 00:05.00 (5.00s)",
        "Shot 2: 00:05.00 → 00:10.00 (5.00s)",
    ]


# ------------------------------------------------------------ real video


av = pytest.importorskip("av")


def write_scenes(path, cuts_at, seconds, fps=24):
    """H.264 footage whose scene changes at `cuts_at`, drifting within each."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"crf": "20", "g": "48"}
    bounds = [0.0, *cuts_at, float(seconds)]
    for n in range(seconds * fps):
        moment = n / fps
        which = sum(1 for cut in cuts_at if moment >= cut)
        make = scene(which)
        k = int((moment - bounds[which]) * 4)
        frame = av.VideoFrame.from_ndarray(make(k), format="rgb24")
        frame.pts, frame.time_base = n, Fraction(1, fps)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_a_video_s_cuts_are_found_through_the_frame_pipeline(tmp_path):
    """Read as a scan reads it: extract_frames, real timestamps, the
    video's own duration."""
    path = tmp_path / "three-scenes.mp4"
    write_scenes(path, cuts_at=[4.0, 9.0], seconds=14)

    shots = shots_of_video(path)

    assert len(shots) == 3
    for shot, cut in zip(shots[1:], (4.0, 9.0)):
        # The cut lies within the window sampling can see it in.
        assert shot.earliest_start < cut <= shot.start_time + 1e-6
        assert shot.start_time - shot.earliest_start <= SHOT_SAMPLE_INTERVAL + 0.05
    assert shots[0].start_time == 0.0
    assert shots[-1].end_time == pytest.approx(14.0, abs=0.05)


@pytest.mark.skipif(not TEST_VIDEO.is_file(), reason="no sample footage")
def test_the_sample_footage_divides_into_shots_that_tile_it():
    """test.mp4 is a fast, dark fan edit. Its cut at 17.5s is the one both
    full-rate references agree on."""
    shots = shots_of_video(TEST_VIDEO)

    assert len(shots) > 3
    assert shots[0].start_time == 0.0
    assert shots[-1].end_time == pytest.approx(23.37, abs=0.05)
    assert all(a.end_time == b.start_time for a, b in zip(shots, shots[1:]))
    assert any(shot.earliest_start < 17.5 <= shot.start_time + 1e-6 for shot in shots[1:])


def test_the_command_line_lists_the_shots(tmp_path, monkeypatch, capsys):
    """`python -m app shots VIDEO` -- the diagnostic this layer has for now."""
    import sys

    from app.__main__ import main

    path = tmp_path / "three-scenes.mp4"
    write_scenes(path, cuts_at=[4.0, 9.0], seconds=14)
    monkeypatch.setattr(sys, "argv", ["app", "shots", str(path)])

    main()

    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("Shot 1: 00:00.00 → ")
    assert sum(line.startswith("Shot ") for line in printed) == 3
    assert "3 shots in 14.0s" in printed[-1]
