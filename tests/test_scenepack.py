"""Choosing a scene pack: one video of someone's clearest whole shots.

Shots and faces are built by hand -- shots as the detector reports them,
faces as (time, quality) samples -- so every rule can be checked exactly
without footage or models.
"""

import pytest

from app.video.cuts import kept, reel_seconds
from app.video.scenepack import (
    MIN_PIECE_SECONDS,
    MIN_PRESENCE,
    Seen,
    available_seconds,
    choose,
    expected_seconds,
    find_pieces,
    parse_length,
    score_shots,
    slack,
)
from app.video.shots import Shot

STEP = 0.5  # the sampling interval


def shots_at(*bounds):
    """Shots tiling 0..bounds[-1], cut at each inner bound, as the detector
    reports them: a cut's earliest_start is the sample before it."""
    edges = [0.0, *bounds]
    return [
        Shot(index=i, start_time=edges[i], end_time=edges[i + 1],
             earliest_start=max(0.0, edges[i] - STEP) if i else 0.0)
        for i in range(len(edges) - 1)
    ]


def seen_in(start, end, quality=0.8):
    """A face at every sample in [start, end)."""
    faces, t = [], start
    while t < end - 1e-9:
        faces.append(Seen(round(t, 3), quality))
        t += STEP
    return faces


def pieces_of(shots, seen, within=None, video=0):
    return find_pieces(score_shots(shots, seen, STEP), video, within)


# ----------------------------------------------------------- the pieces


def test_a_shot_mostly_of_someone_else_is_not_theirs():
    shots = shots_at(10, 20)
    seen = seen_in(0, 10) + seen_in(10, 12)  # 2s of a 10s shot

    pieces = pieces_of(shots, seen)

    assert len(pieces) == 1 and pieces[0].end <= 10


def test_back_to_back_shots_they_hold_play_as_one_piece():
    # A conversation: their close-up, their wide shot, then someone else.
    shots = shots_at(6, 12, 20)
    seen = seen_in(0, 12)

    pieces = pieces_of(shots, seen)

    assert len(pieces) == 1
    assert len(pieces[0].shots) == 2
    assert (pieces[0].start, pieces[0].end) == (0.0, 11.5)


def test_a_piece_starts_and_ends_on_samples_known_to_be_its_shots():
    # The cut into shot 1 lies between 5.5 and 6.0; nothing of shot 0 may
    # open the piece, and nothing of shot 2 may close it.
    shots = shots_at(6, 12, 20)
    seen = seen_in(6, 12)

    piece = pieces_of(shots, seen)[0]

    assert piece.start == 6.0      # the shot's first sample
    assert piece.end == 11.5       # its last sample, before the cut out


def test_a_flash_of_them_is_not_a_piece():
    shots = shots_at(10, 11, 20)
    seen = seen_in(10, 11)

    assert pieces_of(shots, seen) == []
    assert MIN_PIECE_SECONDS > 0.5


def test_two_cards_in_one_frame_are_one_moment_not_two():
    shots = shots_at(10, 20)
    seen = seen_in(0, 2) * 2 + seen_in(10, 20)  # 2s of shot 0, counted twice

    scored = score_shots(shots, seen, STEP)

    assert scored[0].presence < MIN_PRESENCE


def test_a_piece_outside_the_plan_is_left_out():
    # The plan is what the reel keeps: a folder's has already left out what
    # an earlier episode showed, so a recap stays out of the pack too.
    shots = shots_at(10, 20, 30)
    seen = seen_in(0, 10) + seen_in(20, 30)

    pieces = pieces_of(shots, seen, within=[(18.0, 30.0)])

    assert [p.start for p in pieces] == [20.0]


def test_a_quick_cutaway_to_someone_else_ends_the_run():
    """Found in review: runs used to join across any gap under 1.5s, so a
    0.75s reaction shot of someone else was packed inside the piece."""
    shots = [
        Shot(index=0, start_time=0.0, end_time=3.0),
        Shot(index=1, start_time=3.0, end_time=3.75, earliest_start=2.75),
        Shot(index=2, start_time=3.75, end_time=8.0, earliest_start=3.5),
    ]
    seen = [Seen(round(t * 0.25, 2), 0.8) for t in range(0, 12)] + [Seen(round(3.75 + t * 0.25, 2), 0.8) for t in range(17)]

    pieces = find_pieces(score_shots(shots, seen, 0.25))

    assert [(p.start, p.end) for p in pieces] == [(0.0, 2.75), (3.75, 8.0)]


def test_at_a_slow_interval_their_back_to_back_shots_still_join():
    """Found in review: at 2s sampling the gap between two of their shots is
    a whole interval, and a time-based rule never joined them."""
    edges = [0.0, 6.0, 12.0, 20.0]
    shots = [Shot(index=i, start_time=edges[i], end_time=edges[i + 1],
                  earliest_start=max(0.0, edges[i] - 2.0) if i else 0.0) for i in range(3)]
    seen = [Seen(float(t), 0.8) for t in range(0, 12, 2)]

    pieces = find_pieces(score_shots(shots, seen, 2.0))

    assert len(pieces) == 1 and len(pieces[0].shots) == 2


def test_a_one_sample_shot_of_someone_else_still_breaks_the_run():
    shots = shots_at(5, 5.5, 12)  # shot 1 is a single sample
    seen = seen_in(0, 5) + seen_in(5.5, 12)

    pieces = pieces_of(shots, seen)

    assert len(pieces) == 2


# ------------------------------------------------------------ choosing


def three_shots_of_different_clarity():
    shots = shots_at(10, 15, 25, 30, 40, 45)
    seen = seen_in(0, 10, 0.2) + seen_in(15, 25, 0.9) + seen_in(30, 40, 0.5)
    return pieces_of(shots, seen)


def test_the_clearest_pieces_are_taken_and_the_rest_listed_dropped():
    pieces = three_shots_of_different_clarity()

    chosen = choose(pieces, 20, edge_loss=0.0)

    assert [c.dropped for c in chosen] == [True, False, False]
    assert len(chosen) == 3  # every piece is a row, so one can be swapped in


def test_longer_pieces_go_in_before_clearer_quick_ones():
    # A sharp 2s close-up and a less sharp 6s stretch: the stretch first,
    # since a pack of quick close-ups plays as a strobe.
    shots = shots_at(2.5, 10, 16.5, 30)
    seen = seen_in(0, 2.5, 0.99) + seen_in(10, 16.5, 0.6)
    pieces = pieces_of(shots, seen)

    chosen = choose(pieces, 6, edge_loss=0.0)

    assert [(c.start, c.dropped) for c in chosen] == [(0.0, True), (10.0, False)]


def test_quick_pieces_still_fill_what_long_ones_cannot():
    shots = shots_at(2.5, 10, 16.5, 30)
    seen = seen_in(0, 2.5, 0.99) + seen_in(10, 16.5, 0.6)
    pieces = pieces_of(shots, seen)

    # 6s of long pieces is not within 2s of 9; the 2s close-up fills it.
    chosen = choose(pieces, 9, edge_loss=0.0)

    assert all(not c.dropped for c in chosen)


def test_the_pack_plays_in_story_order():
    chosen = kept(choose(three_shots_of_different_clarity(), 20, edge_loss=0.0))

    assert [c.start for c in chosen] == sorted(c.start for c in chosen)


def test_it_lands_within_the_slack_on_whole_shots_rather_than_exactly():
    pieces = three_shots_of_different_clarity()  # pieces of 9.5s each

    chosen = choose(pieces, 20, edge_loss=0.0)

    total = reel_seconds(chosen)
    assert abs(total - 20) <= slack(20)
    assert all(not c.changed for c in kept(chosen))  # nothing cut inside a shot


def test_a_run_too_long_loses_its_weaker_end_shots_not_a_cut_mid_shot():
    # One run of four shots, clearest at the start.
    shots = shots_at(5, 10, 15, 20, 30)
    seen = seen_in(0, 5, 0.9) + seen_in(5, 10, 0.8) + seen_in(10, 15, 0.4) + seen_in(15, 20, 0.3)
    pieces = pieces_of(shots, seen)

    chosen = kept(choose(pieces, 10, edge_loss=0.0))

    assert len(chosen) == 1
    assert chosen[0].start == 0.0 and chosen[0].end == pytest.approx(9.5)  # shots 1-2, whole


def test_only_a_single_shot_longer_than_the_pack_is_cut_inside_around_its_best_face():
    shots = shots_at(60, 70)
    seen = seen_in(0, 60, 0.5) + [Seen(40.0, 0.99)]
    pieces = pieces_of(shots, seen)

    chosen = kept(choose(pieces, 10, edge_loss=0.0))

    assert len(chosen) == 1
    assert chosen[0].start <= 40.0 <= chosen[0].end
    assert abs(chosen[0].seconds - 10) <= slack(10)


def test_a_long_run_keeps_its_long_shot_not_a_sliver_beside_it():
    """Found in review: trimming by score alone dropped the 38s shot, kept a
    1s one, rejected that as too short and left the pack empty."""
    shots = [Shot(index=0, start_time=0.0, end_time=1.5),
             Shot(index=1, start_time=1.5, end_time=40.0, earliest_start=1.0)]
    seen = seen_in(0, 1.5, 0.9) + seen_in(1.5, 40, 0.85)
    pieces = pieces_of(shots, seen)

    chosen = kept(choose(pieces, 20, edge_loss=0.0))

    assert len(chosen) == 1
    assert abs(chosen[0].seconds - 20) <= slack(20)
    assert chosen[0].start >= 1.5  # inside the long shot


def test_a_long_run_keeps_the_longest_stretch_of_whole_shots_that_fits():
    shots = shots_at(2, 10, 13, 25, 40)
    seen = seen_in(0, 25, 0.8)
    pieces = pieces_of(shots, seen)

    chosen = kept(choose(pieces, 12, edge_loss=0.0))

    # Shots 0-2 (0-12.5s) fit in 12s plus 2s of slack; adding shot 3 would
    # not, and no other stretch of whole shots that fits is longer.
    assert [(c.start, c.end) for c in chosen] == [(0.0, 12.5)]


@pytest.mark.parametrize("seconds", [3, 4])
def test_a_very_short_pack_still_takes_something(seconds):
    """Found in review: with 2s of slack, a pack of 2s or less counted as
    done before anything was taken."""
    pieces = three_shots_of_different_clarity()

    assert kept(choose(pieces, seconds, edge_loss=0.0))


def test_less_than_asked_gives_all_of_it_and_says_how_much_there_was():
    pieces = three_shots_of_different_clarity()

    chosen = choose(pieces, 300, edge_loss=0.0)

    assert all(not c.dropped for c in chosen)
    assert available_seconds(pieces, edge_loss=0.0) == pytest.approx(reel_seconds(chosen))


def test_the_expected_length_allows_for_what_cutting_takes_off():
    chosen = choose(three_shots_of_different_clarity(), 20, edge_loss=0.1)

    assert expected_seconds(chosen, edge_loss=0.1) == pytest.approx(reel_seconds(chosen) - 0.2)


def test_a_folder_pack_takes_the_clearest_across_videos_in_folder_order():
    shots = shots_at(10, 20)
    pieces = pieces_of(shots, seen_in(0, 10, 0.3), video=0) + pieces_of(shots, seen_in(0, 10, 0.9), video=1)

    chosen = choose(pieces, 9.5, edge_loss=0.0)

    assert [(c.video, c.dropped) for c in chosen] == [(0, True), (1, False)]


def test_the_slack_grows_with_the_length():
    assert slack(30) == 2.0
    assert slack(120) == pytest.approx(6.0)


# -------------------------------------------------------------- lengths


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


# ------------------------------------------------------------ the shots


def test_a_scanned_videos_shots_are_found_once_and_kept(tmp_path, monkeypatch):
    pytest.importorskip("av")
    from pathlib import Path

    from tests.test_shots import write_scenes
    from app.ui import worker
    from app.ui.worker import ScanResult, find_shots
    from app.video import shots as shots_module

    episode = tmp_path / "episode.mp4"
    write_scenes(episode, cuts_at=[3.0, 6.0, 9.0], seconds=12)
    monkeypatch.setattr(shots_module, "shots_dir", lambda: tmp_path / "shots")
    result = ScanResult(video_path=Path(episode), video_duration=12.0, sample_interval=0.5, people=[])

    first = find_shots(result)
    read = []
    monkeypatch.setattr(worker, "extract_frames", lambda *a, **k: read.append(1) or iter(()))
    again = find_shots(result)

    assert [round(shot.start_time) for shot in first] == [0, 3, 6, 9]
    assert again == first and read == []  # the second time, nothing was decoded


def test_shots_the_scan_carries_are_used_without_reading_or_looking(tmp_path, monkeypatch):
    """Found in review: shots the scan found were only ever kept on disk, so
    a video moved mid-scan cost its first pack a second read."""
    from pathlib import Path

    from app.ui import worker
    from app.ui.worker import ScanResult, find_shots, known_shots

    carried = [Shot(index=0, start_time=0.0, end_time=9.0)]
    result = ScanResult(video_path=Path(tmp_path / "gone.mp4"), video_duration=9.0,
                        sample_interval=0.5, shots=carried)
    monkeypatch.setattr(worker, "kept_shots", lambda *a: pytest.fail("looked on disk"))
    monkeypatch.setattr(worker, "extract_frames", lambda *a, **k: pytest.fail("read the video"))

    assert known_shots(result) is carried and find_shots(result) is carried


def test_another_videos_kept_shots_are_not_lent_to_this_footage(tmp_path, monkeypatch):
    """Found in review: shots were keyed by whatever file is at the path now,
    while the scan reads its footage through a held descriptor."""
    pytest.importorskip("av")
    import os
    from pathlib import Path

    from tests.test_shots import write_scenes
    from app.ui.worker import ScanResult, find_shots
    from app.video import shots as shots_module
    from app.video.source import VideoSource

    monkeypatch.setattr(shots_module, "shots_dir", lambda: tmp_path / "shots")
    scanned, other = tmp_path / "scanned.mp4", tmp_path / "other.mp4"
    write_scenes(scanned, cuts_at=[4.0], seconds=8)
    write_scenes(other, cuts_at=[2.0, 5.0], seconds=11)
    path = tmp_path / "episode.mp4"
    os.replace(scanned, path)
    source = VideoSource(path, keep_open=True)
    # Another video is saved under the same name, with shots of its own.
    os.replace(other, path)
    shots_module.keep_shots(path, 0.5, [Shot(index=0, start_time=0.0, end_time=1.0)])
    try:
        result = ScanResult(video_path=Path(path), video_duration=8.0, sample_interval=0.5, source=source)
        found = find_shots(result)
    finally:
        source.close()

    assert [round(shot.start_time) for shot in found] == [0, 4]   # the footage read
    assert shots_module.kept_shots(path, 0.5)[0].end_time == 1.0  # the other's left alone
