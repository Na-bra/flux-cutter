"""One reel cut from several videos.

A season's episodes are not guaranteed to be alike: a special in a
different shape, a remaster at another sample rate, an episode with no
sound track. Everything the single-video cutter promises -- sound in step
with picture, no manufactured silence, no clicks at a join -- has to hold
across a join between two files as well, where the two sides share
nothing.

The footage is generated, so like the rest of the sync tests these need no
sample video and run in CI.
"""

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from app.video import cutter
from app.video.cutter import Clip, CutterError, cut_clips
from app.video.timeline import AppearanceInterval
from tests.test_sync import read_marks, write_marked_video, write_tone_video

# Each segment holds exactly one flash and its burst, half a second in.
FOUR = [AppearanceInterval(k * 3 + 0.5, k * 3 + 2.0) for k in range(4)]


@pytest.fixture(scope="module")
def footage(tmp_path_factory):
    folder = tmp_path_factory.mktemp("episodes")
    made = {
        "wide_48k": dict(rate=48000),
        "square_44k": dict(rate=44100, width=120, height=90),
        "silent": dict(rate=None),
        "pal": dict(fps=25),
    }
    paths = {}
    for name, options in made.items():
        paths[name] = folder / f"{name}.mp4"
        write_marked_video(paths[name], seconds=12, **options)
    return paths


def offsets(flashes, bursts):
    return np.array([b - f for b, f in zip(bursts, flashes)])


def test_sound_stays_with_picture_across_videos_that_differ(footage, tmp_path):
    """Different sample rate, different shape, same reel."""
    output = tmp_path / "reel.mp4"
    cut_clips(
        [Clip(footage["wide_48k"], FOUR), Clip(footage["square_44k"], FOUR)],
        output,
    )

    flashes, bursts, fragments = read_marks(output)
    assert len(flashes) == len(bursts) == 8
    assert fragments == 0

    spread = offsets(flashes, bursts)
    # The same bound the single-video sync test holds itself to.
    assert spread.max() - spread.min() < 0.05


def test_the_reel_takes_the_first_video_s_picture_and_sound(footage, tmp_path):
    output = tmp_path / "reel.mp4"
    cut_clips(
        [Clip(footage["wide_48k"], FOUR[:1]), Clip(footage["square_44k"], FOUR[:1])],
        output,
    )

    container = av.open(str(output))
    try:
        assert (container.streams.video[0].width, container.streams.video[0].height) == (160, 90)
        assert container.streams.audio[0].rate == 48000
    finally:
        container.close()


def test_a_different_shape_is_fitted_with_bars_not_stretched(footage, tmp_path):
    """The 4:3 episode's white flash must not reach the edges of a 16:9 reel."""
    output = tmp_path / "reel.mp4"
    cut_clips([Clip(footage["wide_48k"], FOUR[:1]), Clip(footage["square_44k"], FOUR[:1])], output)

    container = av.open(str(output))
    try:
        brightest = None
        for frame in container.decode(container.streams.video[0]):
            # The second clip starts after the first's 1.5 seconds.
            if frame.time is not None and frame.time > 1.6:
                image = frame.to_ndarray(format="gray").astype(float)
                if brightest is None or image.mean() > brightest.mean():
                    brightest = image
    finally:
        container.close()

    assert brightest is not None
    # 120x90 scaled into 160x90 is 120 wide, leaving 20 columns each side.
    assert brightest[:, :16].mean() < 30
    assert brightest[:, -16:].mean() < 30
    assert brightest[:, 30:130].mean() > 200


def test_a_video_with_no_sound_is_silence_of_exactly_its_length(footage, tmp_path):
    """If the silent episode came out a frame short or long, every burst
    after it would move against its flash."""
    output = tmp_path / "reel.mp4"
    cut_clips(
        [
            Clip(footage["wide_48k"], FOUR[:2]),
            Clip(footage["silent"], FOUR[:2]),
            Clip(footage["wide_48k"], FOUR[2:]),
        ],
        output,
    )

    flashes, bursts, _ = read_marks(output)
    assert len(flashes) == 6
    assert len(bursts) == 4

    heard = [flashes[0], flashes[1], flashes[4], flashes[5]]
    spread = offsets(heard, bursts)
    # Tighter than the usual 50ms, on purpose. Sound is counted from the
    # start of the reel, so a silent video one frame too long throws only
    # the next segment out -- by 42.7ms, measured -- and the one after is
    # exact again. At 50ms that slipped through; on this footage the honest
    # spread is 16ms.
    assert spread.max() - spread.min() < 0.03


def test_a_reel_whose_first_video_is_silent_still_has_sound(footage, tmp_path):
    output = tmp_path / "reel.mp4"
    cut_clips([Clip(footage["silent"], FOUR[:1]), Clip(footage["wide_48k"], FOUR[:1])], output)

    container = av.open(str(output))
    try:
        assert container.streams.audio, "sound from the second video was dropped"
    finally:
        container.close()


def test_different_frame_rates_are_refused_before_anything_is_written(
    footage, tmp_path
):
    """25fps footage in a 24fps reel would play 4% slow and take the wrong
    length of sound with it. Refused up front, naming both videos, and
    before a single frame is encoded."""
    output = tmp_path / "reel.mp4"

    with pytest.raises(CutterError) as refused:
        cut_clips([Clip(footage["wide_48k"], FOUR), Clip(footage["pal"], FOUR)], output)

    message = str(refused.value)
    assert "wide_48k.mp4" in message and "pal.mp4" in message
    assert "24fps" in message and "25fps" in message
    assert not output.exists()


def test_no_silence_is_manufactured_at_a_join_between_videos(tmp_path, monkeypatch):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    write_tone_video(first, seconds=8)
    write_tone_video(second, seconds=8, rate=44100)

    made_up = []
    original = cutter._AudioState._silence

    def counting(self, samples):
        made_up.append(samples)
        return original(self, samples)

    monkeypatch.setattr(cutter._AudioState, "_silence", counting)

    spans = [AppearanceInterval(1.013, 3.271), AppearanceInterval(4.502, 6.918)]
    cut_clips([Clip(first, spans), Clip(second, spans)], tmp_path / "reel.mp4")

    assert sum(made_up) == 0


def test_no_click_at_a_join_between_videos(tmp_path):
    """Two tones from two files have no phase in common, so a join between
    them is the worst case for a click."""
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    write_tone_video(first, seconds=8)
    write_tone_video(second, seconds=8, rate=44100)
    output = tmp_path / "reel.mp4"

    spans = [AppearanceInterval(1.013, 3.271), AppearanceInterval(4.502, 6.918)]
    cut_clips([Clip(first, spans), Clip(second, spans)], output)

    def decoded(path):
        container = av.open(str(path))
        try:
            return np.concatenate(
                [
                    f.to_ndarray().astype(np.float64).reshape(-1)
                    for f in container.decode(container.streams.audio[0])
                ]
            )
        finally:
            container.close()

    source_steps = np.abs(np.diff(decoded(first)))
    reel = decoded(output)
    assert np.abs(np.diff(reel[2048:-2048])).max() < 2 * source_steps.max()


def test_progress_is_reported_per_video_and_counted_across_the_reel(footage, tmp_path):
    clips_started, segments_done = [], []
    cut_clips(
        [Clip(footage["wide_48k"], FOUR[:2]), Clip(footage["square_44k"], FOUR[:1])],
        tmp_path / "reel.mp4",
        on_clip=lambda index, total, path: clips_started.append((index, total, path.name)),
        on_segment=lambda index, total, segment: segments_done.append((index, total)),
    )

    assert clips_started == [(0, 2, "wide_48k.mp4"), (1, 2, "square_44k.mp4")]
    assert segments_done == [(0, 3), (1, 3), (2, 3)]


def test_videos_with_nothing_to_take_are_passed_over(footage, tmp_path):
    result = cut_clips(
        [Clip(footage["wide_48k"], []), Clip(footage["square_44k"], FOUR[:1])],
        tmp_path / "reel.mp4",
    )
    assert result.segment_count == 1


def test_nothing_to_take_from_any_video_is_an_error(footage, tmp_path):
    with pytest.raises(CutterError):
        cut_clips([Clip(footage["wide_48k"], [])], tmp_path / "reel.mp4")
