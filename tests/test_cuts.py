"""The cut list: dropping cuts, moving their ends, and what stops them.

The promise the cutter relies on -- non-overlapping segments, in order,
per video -- is the thing most of these check, because every edit here is
a chance to break it.
"""

import pytest

from app.video.cuts import (
    END,
    MIN_CUT_SECONDS,
    START,
    Cut,
    cuts_from_plans,
    drop,
    kept,
    move,
    plans_from_cuts,
    reel_seconds,
    restore,
)
from app.video.timeline import AppearanceInterval


def plan(*per_video):
    """(video, (start, end), (start, end)...) for each video."""
    return [
        (video, [AppearanceInterval(start_time=s, end_time=e) for s, e in spans])
        for video, *spans in per_video
    ]


def spans(cuts):
    return [(round(cut.start, 3), round(cut.end, 3)) for cut in cuts]


@pytest.fixture
def cuts():
    # One video, three cuts with room between them.
    return cuts_from_plans(plan((0, (10.0, 14.0), (20.0, 24.0), (30.0, 34.0))))


def test_a_plan_becomes_one_numbered_list_in_playing_order():
    made = cuts_from_plans(plan((0, (10.0, 14.0)), (1, (2.0, 5.0), (8.0, 9.0))))

    assert [cut.video for cut in made] == [0, 1, 1]
    assert spans(made) == [(10.0, 14.0), (2.0, 5.0), (8.0, 9.0)]
    assert not any(cut.changed for cut in made)


def test_dropping_a_cut_leaves_it_in_the_list_but_out_of_the_reel(cuts):
    edited = drop(cuts, 1)

    assert edited[1].dropped is True
    assert len(edited) == 3 and len(kept(edited)) == 2
    assert reel_seconds(edited) == pytest.approx(8.0)
    assert spans(kept(edited)) == [(10.0, 14.0), (30.0, 34.0)]


def test_dropping_twice_puts_it_back(cuts):
    assert drop(drop(cuts, 1), 1) == cuts


def test_the_export_gets_what_is_left(cuts):
    plans = plans_from_cuts(drop(cuts, 0))

    assert plans == [(0, [AppearanceInterval(20.0, 24.0), AppearanceInterval(30.0, 34.0)])]


def test_a_video_whose_cuts_are_all_dropped_is_not_exported():
    made = cuts_from_plans(plan((0, (1.0, 5.0)), (1, (2.0, 6.0))))

    assert [video for video, _ in plans_from_cuts(drop(made, 1))] == [0]


def test_moving_an_end_changes_only_that_end(cuts):
    edited = move(cuts, 0, END, 1.5, video_duration=100.0)

    assert spans(edited)[0] == (10.0, 15.5)
    assert edited[0].changed is True
    assert spans(edited)[1:] == [(20.0, 24.0), (30.0, 34.0)]


def test_a_cut_stops_at_its_neighbour_rather_than_overlapping_it(cuts):
    """The cutter is promised non-overlapping segments; pressing repeatedly
    must not be a way to break that."""
    edited = cuts
    for _ in range(20):
        edited = move(edited, 0, END, 1.0, video_duration=100.0)

    assert spans(edited)[0] == (10.0, 20.0)
    assert spans(edited)[1] == (20.0, 24.0)


def test_a_dropped_neighbour_gives_its_footage_back(cuts):
    """Dropping one half of a cut that was split in two and growing the
    other over it is the way that is fixed."""
    edited = drop(cuts, 1)
    for _ in range(20):
        edited = move(edited, 0, END, 1.0, video_duration=100.0)

    assert spans(edited)[0] == (10.0, 30.0)


def test_a_cut_stops_at_the_start_and_end_of_its_video(cuts):
    early = move(cuts, 0, START, -30.0, video_duration=100.0)
    late = move(cuts, 2, END, 500.0, video_duration=34.5)

    assert early[0].start == 0.0
    assert late[2].end == 34.5


def test_a_cut_cannot_be_shortened_into_nothing(cuts):
    squeezed = move(cuts, 0, START, 100.0, video_duration=100.0)

    assert squeezed[0].seconds == pytest.approx(MIN_CUT_SECONDS)
    assert squeezed[0].start < squeezed[0].end


def test_neighbours_in_another_video_do_not_get_in_the_way():
    """Two videos share a timeline of seconds but not a reel of footage."""
    made = cuts_from_plans(plan((0, (10.0, 14.0)), (1, (12.0, 16.0))))

    edited = move(made, 0, END, 20.0, video_duration=100.0)

    assert spans(edited)[0] == (10.0, 34.0)


def test_putting_one_cut_back(cuts):
    edited = drop(move(cuts, 1, START, -3.0, video_duration=100.0), 1)

    assert restore(edited, 1)[1] == cuts[1]
    assert restore(edited, 1)[1].changed is False


def test_an_index_that_is_not_there_changes_nothing(cuts):
    assert drop(cuts, 9) == cuts
    assert move(cuts, -1, END, 1.0, video_duration=100.0) == cuts
    assert restore(cuts, 9) == cuts


def test_an_unknown_edge_is_a_mistake_worth_hearing_about(cuts):
    with pytest.raises(ValueError):
        move(cuts, 0, "middle", 1.0, video_duration=100.0)


def test_editing_returns_a_new_list_and_leaves_the_old_one_alone(cuts):
    before = list(cuts)

    move(drop(cuts, 0), 1, END, 2.0, video_duration=100.0)

    assert cuts == before


def test_a_cut_knows_its_own_length():
    cut = Cut(video=0, start=4.0, end=9.5, planned_start=4.0, planned_end=9.5)

    assert cut.seconds == pytest.approx(5.5)
    assert cut.changed is False
