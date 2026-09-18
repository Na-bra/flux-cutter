"""Leaving footage a season reel has already shown out of later episodes.

Most of these build fingerprints by hand, so the rules can be checked
exactly. One runs the whole path -- decode, hash, match -- on generated
footage and a re-encoded excerpt of it, so it needs no sample video.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app.video.repeats import (
    MAX_BITS,
    Fingerprints,
    Repeat,
    find_repeats,
    fingerprint,
    fingerprint_kept,
    without_repeats,
)
from app.video.timeline import AppearanceInterval

STEP = 0.5
rng = np.random.default_rng(7)


def random_hashes(n):
    return rng.integers(0, 2**63, size=n, dtype=np.uint64)


def prints(hashes, start=0.0, usable=None):
    hashes = np.asarray(hashes, dtype=np.uint64)
    times = start + STEP * np.arange(len(hashes))
    return Fingerprints(times, hashes, np.ones(len(hashes), bool) if usable is None else usable)


def blur(hashes, bits=4):
    """The same frames, a few bits off, as a re-encode leaves them."""
    out = hashes.copy()
    for i in range(len(out)):
        for bit in rng.choice(63, size=bits, replace=False):
            out[i] ^= np.uint64(1) << np.uint64(bit)
    return out


def test_a_block_lifted_from_an_earlier_video_is_found_with_its_offset():
    earlier = random_hashes(400)
    # 60s of the earlier video, from 30s in, opens the later one.
    later = np.concatenate([blur(earlier[60:180]), random_hashes(200)])

    found = find_repeats(prints(later), prints(earlier))

    assert len(found) == 1
    assert found[0].offset == pytest.approx(30.0, abs=0.05)
    assert found[0].start == pytest.approx(0.0)
    assert found[0].end == pytest.approx(59.5)


def test_unrelated_videos_repeat_nothing():
    assert find_repeats(prints(random_hashes(400)), prints(random_hashes(400))) == []


def test_frames_with_no_detail_are_not_evidence():
    """Black frames look alike in every video; a run of them is not a recap."""
    black = np.zeros(40, dtype=np.uint64)
    earlier = prints(np.concatenate([random_hashes(100), black]),
                     usable=np.r_[np.ones(100, bool), np.zeros(40, bool)])
    later = prints(np.concatenate([black, random_hashes(100)]),
                   usable=np.r_[np.zeros(40, bool), np.ones(100, bool)])

    assert find_repeats(later, earlier) == []


def test_a_fast_moment_inside_a_recap_does_not_break_it_in_two():
    earlier = random_hashes(400)
    block = blur(earlier[60:180])
    block[40:46] = random_hashes(6)  # 3s where nothing hashes alike
    found = find_repeats(prints(block), prints(earlier))

    assert len(found) == 1
    assert found[0].end - found[0].start == pytest.approx(59.5)


def test_a_moment_too_short_to_be_a_recap_is_not_one():
    earlier = random_hashes(400)
    later = random_hashes(200)
    later[50:54] = earlier[100:104]  # 2s, under the 3s a stretch needs

    assert find_repeats(prints(later), prints(earlier)) == []


def test_frames_just_outside_the_bit_limit_do_not_match():
    earlier = random_hashes(200)
    later = blur(earlier[20:120], bits=MAX_BITS + 6)

    assert find_repeats(prints(later), prints(earlier)) == []


# ---------------------------------------------------------- leaving it out


def span(a, b):
    return AppearanceInterval(a, b)


def test_what_the_reel_already_showed_is_left_out_of_the_later_video():
    """Video 1 opens with 0-60s of video 0's footage from 30s in."""
    repeats = {(1, 0): [Repeat(0.0, 59.5, 30.0, 120)]}
    plans = [(0, [span(40.0, 50.0)]), (1, [span(5.0, 25.0), span(100.0, 110.0)])]

    trimmed, removed = without_repeats(plans, repeats)

    # 10-20s of video 1 is 40-50s of video 0, which the reel already has.
    assert trimmed[1][1] == [span(5.0, 10.0), span(20.0, 25.0), span(100.0, 110.0)]
    assert removed == {1: pytest.approx(10.0)}
    assert trimmed[0] == plans[0]


def test_a_recap_of_something_the_reel_skipped_stays():
    """The only time it appears in the reel, so it is not a repeat of it."""
    repeats = {(1, 0): [Repeat(0.0, 59.5, 30.0, 120)]}
    plans = [(0, [span(200.0, 210.0)]), (1, [span(5.0, 25.0)])]

    trimmed, removed = without_repeats(plans, repeats)

    assert trimmed[1][1] == [span(5.0, 25.0)]
    assert removed == {}


def test_a_repeat_of_a_video_not_in_the_reel_stays():
    repeats = {(1, 0): [Repeat(0.0, 59.5, 30.0, 120)]}
    trimmed, removed = without_repeats([(1, [span(5.0, 25.0)])], repeats)

    assert trimmed == [(1, [span(5.0, 25.0)])]
    assert removed == {}


def test_slivers_left_by_a_repeat_are_not_kept_as_cuts():
    repeats = {(1, 0): [Repeat(0.0, 59.5, 30.0, 120)]}
    plans = [(0, [span(35.0, 54.8)]), (1, [span(5.0, 25.0)])]

    trimmed, _ = without_repeats(plans, repeats)

    # 5-24.8s goes; the 0.2s left at the end is not worth a cut.
    assert trimmed[1][1] == []


def test_an_opening_repeated_in_every_episode_is_kept_once():
    """Episodes 1 and 2 both open with episode 0's first 30s."""
    repeats = {
        (1, 0): [Repeat(0.0, 29.5, 0.0, 60)],
        (2, 0): [Repeat(0.0, 29.5, 0.0, 60)],
        (2, 1): [Repeat(0.0, 29.5, 0.0, 60)],
    }
    plans = [(v, [span(0.0, 30.0)]) for v in range(3)]

    trimmed, removed = without_repeats(plans, repeats)

    assert trimmed[0][1] == [span(0.0, 30.0)]
    assert trimmed[1][1] == [] and trimmed[2][1] == []
    assert set(removed) == {1, 2}


# ------------------------------------------------------ the whole path


av = pytest.importorskip("av")


def write_textured(path: Path, seconds: int, fps: int = 24, seed: int = 1) -> None:
    """Footage with detail that changes over time, as real footage has."""
    generator = np.random.default_rng(seed)
    base = generator.integers(0, 255, size=(90, 160, 3), dtype=np.uint8)
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
    video.options = {"crf": "18", "g": "12"}
    for n in range(seconds * fps):
        # A new scene every 2s, drifting within it.
        scene = np.roll(base, shift=(n // (2 * fps)) * 37, axis=1)
        frame = np.roll(scene, shift=n % (2 * fps), axis=0)
        f = av.VideoFrame.from_ndarray(frame, format="rgb24")
        f.pts, f.time_base = n, Fraction(1, fps)
        for packet in video.encode(f):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def test_a_re_encoded_excerpt_is_found_through_decoding_and_hashing(tmp_path):
    from app.video.cutter import cut_segments

    episode = tmp_path / "episode.mp4"
    write_textured(episode, seconds=40)
    other = tmp_path / "other.mp4"
    write_textured(other, seconds=40, seed=2)
    recap = tmp_path / "recap.mp4"
    # At an offset that does not fall on the sampling grid, at another quality.
    cut_segments(episode, [AppearanceInterval(10.23, 30.23)], recap, quality=30, include_audio=False)

    source = fingerprint(episode)
    found = find_repeats(fingerprint(recap), source)

    assert len(found) == 1
    assert found[0].offset == pytest.approx(10.23, abs=0.15)
    assert found[0].end - found[0].start >= 15.0
    assert find_repeats(fingerprint(other), source) == []


def test_a_fingerprint_is_made_once_and_kept(tmp_path, monkeypatch):
    from app.video import repeats

    episode = tmp_path / "episode.mp4"
    write_textured(episode, seconds=4)
    made = []
    real = repeats.fingerprint
    monkeypatch.setattr(repeats, "fingerprint", lambda path: made.append(path) or real(path))

    first = fingerprint_kept(episode, tmp_path / "prints")
    second = fingerprint_kept(episode, tmp_path / "prints")

    assert made == [episode]
    np.testing.assert_array_equal(first.hashes, second.hashes)
