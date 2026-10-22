"""Choosing a scene pack: the clearest scenes, up to a chosen length.

The scenes are cut lists built by hand and the faces are (time, quality)
pairs, so every rule can be checked exactly without footage or models.
"""

import pytest

from app.video.cuts import Cut, kept, reel_seconds
from app.video.scenepack import (
    MIN_SCENE_SECONDS,
    Seen,
    available_seconds,
    choose,
    clarity,
    clip_names,
    expected_seconds,
    parse_length,
)

INTERVAL = 1.0


def cut(start, end, video=0, dropped=False):
    return Cut(video, start, end, start, end, dropped)


def faces(start, end, quality):
    """One face a second across [start, end), all of one quality."""
    return [Seen(float(t), quality) for t in range(int(start), int(end))]


def pick(cuts, seen, seconds):
    """Choosing with nothing lost at the edges, so the rules show exactly."""
    return choose(cuts, {0: seen}, {0: INTERVAL}, seconds, edge_loss=0.0)


def test_the_clearest_scenes_are_chosen_and_the_rest_dropped():
    cuts = [cut(0, 10), cut(20, 30), cut(40, 50)]
    seen = faces(0, 10, 0.2) + faces(20, 30, 0.9) + faces(40, 50, 0.5)

    chosen = pick(cuts, seen, 20)

    assert [c.dropped for c in chosen] == [True, False, False]
    assert reel_seconds(chosen) == 20


def test_the_pack_keeps_story_order_whatever_the_ranking():
    cuts = [cut(0, 10), cut(20, 30)]
    seen = faces(0, 10, 0.3) + faces(20, 30, 0.9)

    chosen = kept(pick(cuts, seen, 20))

    assert [c.start for c in chosen] == [0, 20]


def test_the_scene_that_does_not_fit_is_trimmed_around_its_clearest_moment():
    cuts = [cut(0, 10), cut(100, 130)]
    seen = faces(0, 10, 0.9) + faces(100, 130, 0.5) + [Seen(120.0, 0.95)]

    chosen = pick(cuts, seen, 16)

    assert reel_seconds(chosen) == pytest.approx(16)
    trimmed = chosen[1]
    assert trimmed.end - trimmed.start == pytest.approx(6)
    assert trimmed.start <= 120 <= trimmed.end
    # It shows as changed in the cut list, and can be put back from there.
    assert trimmed.changed and (trimmed.planned_start, trimmed.planned_end) == (100, 130)


def test_a_trimmed_scene_stays_inside_its_cut():
    cuts = [cut(100, 130)]
    seen = faces(100, 130, 0.5) + [Seen(129.0, 0.99)]

    chosen = pick(cuts, seen, 10)

    assert (chosen[0].start, chosen[0].end) == (120, 130)


def test_too_little_left_to_be_a_scene_is_not_filled():
    cuts = [cut(0, 9), cut(20, 40)]
    seen = faces(0, 9, 0.9) + faces(20, 40, 0.5)

    chosen = pick(cuts, seen, 9 + MIN_SCENE_SECONDS - 0.5)

    assert chosen[1].dropped
    assert reel_seconds(chosen) == 9


def test_a_scene_they_are_never_seen_in_is_never_chosen():
    cuts = [cut(0, 10), cut(20, 30)]
    seen = faces(0, 10, 0.9)

    chosen = pick(cuts, seen, 60)

    assert chosen[1].dropped
    assert available_seconds(cuts, {0: seen}, {0: INTERVAL}) == 10


def test_a_scene_mostly_of_other_people_ranks_below_one_mostly_of_them():
    # Same face quality, but in the second they are seen for 2s of 10.
    full, glimpsed = cut(0, 10), cut(20, 30)
    seen = faces(0, 10, 0.6) + faces(20, 22, 0.6)

    assert clarity(full, seen, INTERVAL).score > clarity(glimpsed, seen, INTERVAL).score


def test_cuts_already_dropped_stay_dropped():
    cuts = [cut(0, 10, dropped=True), cut(20, 30)]
    seen = faces(0, 10, 0.99) + faces(20, 30, 0.1)

    chosen = pick(cuts, seen, 60)

    assert chosen[0].dropped and not chosen[1].dropped


def test_less_footage_than_asked_gives_all_of_it():
    cuts = [cut(0, 10), cut(20, 30)]
    seen = faces(0, 10, 0.5) + faces(20, 30, 0.5)

    assert reel_seconds(pick(cuts, seen, 300)) == 20


def test_a_folder_pack_scores_each_video_by_its_own_faces():
    cuts = [cut(0, 10, video=0), cut(0, 10, video=1)]
    seen = {0: faces(0, 10, 0.2), 1: faces(0, 10, 0.9)}

    chosen = choose(cuts, seen, {0: 1.0, 1: 1.0}, 10, edge_loss=0.0)

    assert [c.dropped for c in chosen] == [True, False]


def test_the_pack_is_planned_long_by_what_cutting_takes_off_each_scene():
    cuts = [cut(0, 10), cut(20, 30), cut(40, 50)]
    seen = faces(0, 10, 0.9) + faces(20, 30, 0.8) + faces(40, 50, 0.7)

    chosen = choose(cuts, {0: seen}, {0: INTERVAL}, 22, edge_loss=0.5)

    # Two whole scenes come out 19s; the third is trimmed to the 3s still
    # wanted plus what its own edges will lose.
    assert reel_seconds(chosen) == pytest.approx(23.5)
    assert expected_seconds(chosen, edge_loss=0.5) == pytest.approx(22.0)


def test_clips_are_numbered_in_story_order_and_say_where_they_start():
    cuts = [cut(14.5, 20), cut(30, 40, dropped=True), cut(247, 260)]

    assert clip_names("Ben", "mp4", cuts) == ["Ben 001 (0m14s).mp4", "Ben 002 (4m07s).mp4"]


def test_a_folder_pack_names_the_video_each_clip_is_from():
    cuts = [cut(5, 9, video=1)]

    names = clip_names("Ben", "mkv", cuts, videos=["e1.mkv", "S01E02.mkv"])

    assert names == ["Ben 001 (S01E02 0m05s).mkv"]


@pytest.mark.parametrize(
    "typed, seconds",
    [("90", 90), ("90s", 90), ("1:30", 90), ("1m30s", 90), ("2m", 120), (" 45 ", 45), ("", None)],
)
def test_a_length_can_be_typed_the_ways_people_type_one(typed, seconds):
    assert parse_length(typed) == seconds


@pytest.mark.parametrize("typed", ["soon", "0", "-5", "1:xx"])
def test_what_is_not_a_length_is_refused(typed):
    with pytest.raises(ValueError):
        parse_length(typed)


# ----------------------------------------------------- writing the pack


def test_a_pack_is_a_reel_and_a_clip_per_scene(tmp_path):
    av = pytest.importorskip("av")
    from tests.test_repeats import write_textured
    from app.ui.worker import export_scene_pack
    from app.video.timeline import AppearanceInterval

    episode = tmp_path / "episode.mp4"
    write_textured(episode, seconds=12)
    scenes = [(episode, AppearanceInterval(1.0, 3.0)), (episode, AppearanceInterval(6.0, 9.0))]
    progress = []

    written = export_scene_pack(
        scenes, ["Ben 001.mp4", "Ben 002.mp4"], tmp_path / "Ben", "Ben.mp4",
        on_progress=lambda fraction, done, total: progress.append((done, total)),
    )

    assert [p.name for p in written] == ["Ben.mp4", "Ben 001.mp4", "Ben 002.mp4"]

    def seconds(path):
        with av.open(str(path)) as container:
            return float(container.duration) / av.time_base

    assert seconds(written[1]) == pytest.approx(2.0, abs=0.15)
    assert seconds(written[2]) == pytest.approx(3.0, abs=0.15)
    assert seconds(written[0]) == pytest.approx(5.0, abs=0.25)
    assert progress[-1] == (4, 4)


def test_a_cancelled_pack_leaves_nothing_behind(tmp_path):
    pytest.importorskip("av")
    import threading

    from tests.test_repeats import write_textured
    from app.ui.worker import Cancelled, export_scene_pack
    from app.video.timeline import AppearanceInterval

    episode = tmp_path / "episode.mp4"
    write_textured(episode, seconds=8)
    cancel = threading.Event()
    scenes = [(episode, AppearanceInterval(1.0, 3.0)), (episode, AppearanceInterval(4.0, 6.0))]

    def stop_after_first(fraction, done, total):
        if done == 1:
            cancel.set()

    with pytest.raises(Cancelled):
        export_scene_pack(
            scenes, ["a.mp4", "b.mp4"], tmp_path / "pack", "reel.mp4",
            on_progress=stop_after_first, cancel=cancel,
        )

    assert not (tmp_path / "pack").exists()


def test_a_second_pack_does_not_land_in_the_first_ones_folder(tmp_path):
    from app.ui.worker import pack_directory

    first = tmp_path / "Ben"
    assert pack_directory(first) == first
    first.mkdir()
    assert pack_directory(first) == first  # empty is fine
    (first / "Ben 001.mp4").write_bytes(b"x")
    assert pack_directory(first) == tmp_path / "Ben (2)"
