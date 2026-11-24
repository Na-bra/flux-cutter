"""A folder in the window: scanning it, its cast, and one reel across it.

The scans, edits and cuts here are stand-ins -- each is tested where it
lives. What is tested is the folder's own job: carrying on past a bad file,
turning cards into a cast, and handing the right segments from the right
videos to the cut.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.faces.cast import Answers, CardRef
from app.faces.edits import EditError
from app.faces.grouper import FaceIdentityGroup
from app.ui import folder as folder_module
from app.ui.folder import (
    FolderScan,
    add_videos,
    cast_of,
    cast_preview_frames,
    export_cast,
    name_cast_person,
    plan_cast_export,
    reordered,
    scan_folder,
    without_video,
)
from app.ui.worker import Cancelled, Person, ScanResult, ScanSettings
from app.video.cutter import CutResult, CutterError
from app.video.loader import VideoLoadError
from app.video.timeline import AppearanceInterval

SETTINGS = ScanSettings.for_mode("live")


def face(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def person(index, vector, name=None, detections=10):
    return Person(
        index=index,
        thumbnail=Image.new("RGB", (8, 8)),
        detection_count=detections,
        first_seen=0.0,
        last_seen=1.0,
        group=FaceIdentityGroup(group_id=index, representative_embedding=vector),
        name=name,
    )


def result(name, *people, duration=60.0):
    return ScanResult(
        video_path=Path(f"/season/{name}"),
        video_duration=duration,
        sample_interval=0.5,
        people=list(people),
    )


LEAD, FRIEND, STRANGER = face(1, 0, 0), face(0, 1, 0), face(0, 0, 1)


@pytest.fixture
def season():
    return FolderScan(
        videos=[
            result("e1.mp4", person(0, LEAD, detections=40), person(1, FRIEND)),
            result("e2.mp4", person(0, FRIEND), person(1, LEAD, detections=30)),
            result("e3.mp4", person(0, STRANGER)),
        ],
        settings=SETTINGS,
    )


# ----------------------------------------------------------------- scanning


def test_every_video_is_scanned_and_a_bad_one_is_passed_over(tmp_path, monkeypatch):
    for name in ("e1.mp4", "e2.mp4", "e3.mp4"):
        (tmp_path / name).write_bytes(b"x")
    started, progress, fingerprinted = [], [], []

    def fake_scan(path, settings, on_progress, cancel, on_download, fingerprint):
        fingerprinted.append(fingerprint)
        if path.name == "e2.mp4":
            raise VideoLoadError("cannot read e2")
        on_progress(0.5, 1.0)
        return result(path.name)

    monkeypatch.setattr(folder_module, "scan", fake_scan)

    scanned = scan_folder(
        [tmp_path],
        SETTINGS,
        on_video=lambda i, total, path: started.append((i, total, path.name)),
        on_progress=lambda fraction, at: progress.append(round(fraction, 3)),
    )

    assert [r.video_path.name for r in scanned.videos] == ["e1.mp4", "e3.mp4"]
    assert scanned.skipped == [(tmp_path / "e2.mp4", "cannot read e2")]
    assert started == [(0, 3, "e1.mp4"), (1, 3, "e2.mp4"), (2, 3, "e3.mp4")]
    # Progress is across the folder, not restarting per video.
    assert progress == [0.167, 0.833]
    # Repeats are looked for next, so each is fingerprinted as it is scanned.
    assert fingerprinted == [True, True, True]


def test_a_folder_of_one_video_is_not_fingerprinted(tmp_path, monkeypatch):
    (tmp_path / "e1.mp4").write_bytes(b"x")
    asked = []

    def fake_scan(path, fingerprint, **kwargs):
        asked.append(fingerprint)
        return result(path.name)

    monkeypatch.setattr(folder_module, "scan", fake_scan)
    scan_folder([tmp_path], SETTINGS)
    assert asked == [False]


def test_cancelling_closes_what_was_already_scanned(tmp_path, monkeypatch):
    for name in ("e1.mp4", "e2.mp4"):
        (tmp_path / name).write_bytes(b"x")
    closed = []

    class Held(SimpleNamespace):
        def close(self):
            closed.append(self.name)

    def fake_scan(path, **kwargs):
        if path.name == "e2.mp4":
            raise Cancelled()
        return Held(name=path.name)

    monkeypatch.setattr(folder_module, "scan", fake_scan)

    with pytest.raises(Cancelled):
        scan_folder([tmp_path], SETTINGS)
    assert closed == ["e1.mp4"]


# ------------------------------------------------- adding and taking away


def scanned_at(path, *people):
    """A scan of a real path, so adding and removing can match the file."""
    return ScanResult(
        video_path=path,
        video_duration=60.0,
        sample_interval=0.5,
        people=list(people) or [person(0, LEAD)],
    )


@pytest.fixture
def two_folders(tmp_path, monkeypatch):
    """Two videos in two folders, and the names of what gets scanned."""
    scanned = []

    def fake_scan(path, **kwargs):
        scanned.append(path.name)
        return scanned_at(path)

    monkeypatch.setattr(folder_module, "scan", fake_scan)
    monkeypatch.setattr(folder_module, "find_folder_repeats", lambda *a, **k: {})
    paths = []
    for folder, name in (("a", "e3.mp4"), ("b", "e7.mp4")):
        (tmp_path / folder).mkdir()
        paths.append(tmp_path / folder / name)
        paths[-1].write_bytes(b"x")
    return paths, scanned


def test_a_set_is_the_files_it_was_given_whatever_folders_they_are_in(two_folders):
    (first, second), scanned = two_folders

    picked = scan_folder([first, second], SETTINGS)

    assert [r.video_path for r in picked.videos] == [first, second]
    assert scanned == ["e3.mp4", "e7.mp4"]


def test_adding_a_video_scans_only_the_one_added(two_folders):
    (first, second), scanned = two_folders
    picked = scan_folder([first], SETTINGS)
    assert scanned == ["e3.mp4"]

    grown = add_videos(picked, [second])

    assert scanned == ["e3.mp4", "e7.mp4"]
    assert [r.video_path for r in grown.videos] == [first, second]
    # The set it was added to is untouched, as any FolderScan is.
    assert [r.video_path for r in picked.videos] == [first]


def test_a_video_already_in_the_set_is_not_scanned_again(two_folders):
    (first, second), scanned = two_folders
    picked = scan_folder([first, second], SETTINGS)

    assert add_videos(picked, [second, first]) is picked
    assert scanned == ["e3.mp4", "e7.mp4"]


def test_the_video_added_is_fingerprinted_for_the_repeats(tmp_path, monkeypatch):
    """One video is never a repeat of anything, so a set of one does not
    fingerprint; the second makes both worth looking at."""
    asked = []

    def fake_scan(path, fingerprint, **kwargs):
        asked.append(fingerprint)
        return scanned_at(path)

    monkeypatch.setattr(folder_module, "scan", fake_scan)
    monkeypatch.setattr(folder_module, "find_folder_repeats", lambda *a, **k: {})
    for name in ("e1.mp4", "e2.mp4"):
        (tmp_path / name).write_bytes(b"x")

    add_videos(scan_folder([tmp_path / "e1.mp4"], SETTINGS), [tmp_path / "e2.mp4"])

    assert asked == [False, True]


def test_the_settings_are_the_set_s_own_not_whatever_is_current(tmp_path, monkeypatch):
    """A card scanned under other settings cannot be compared with the
    rest, so an addition scans under the settings the set was scanned on."""
    asked = []

    def remember(path, settings, **kwargs):
        asked.append(settings)
        return scanned_at(path)

    monkeypatch.setattr(folder_module, "scan", remember)
    monkeypatch.setattr(folder_module, "find_folder_repeats", lambda *a, **k: {})
    added = tmp_path / "e2.mp4"
    added.write_bytes(b"x")
    picked = FolderScan(videos=[scanned_at(tmp_path / "e1.mp4")], settings=SETTINGS)

    add_videos(picked, [added])

    assert asked == [SETTINGS]


def test_cancelling_an_addition_leaves_the_set_as_it_was(two_folders, monkeypatch):
    (first, second), _ = two_folders
    picked = scan_folder([first], SETTINGS)

    def cancelled(path, **kwargs):
        raise Cancelled()

    monkeypatch.setattr(folder_module, "scan", cancelled)

    with pytest.raises(Cancelled):
        add_videos(picked, [second])
    assert [r.video_path for r in picked.videos] == [first]


def test_cancelling_the_search_for_repeats_releases_what_was_scanned(tmp_path, monkeypatch):
    """The scans are only handed to a FolderScan -- whose `close` releases
    them -- once the repeats are known, so a cancel in between has to."""
    closed = []
    for name in ("e1.mp4", "e2.mp4"):
        (tmp_path / name).write_bytes(b"x")

    def fake_scan(path, **kwargs):
        return replace(
            scanned_at(path),
            source=SimpleNamespace(close=lambda name=path.name: closed.append(name)),
        )

    monkeypatch.setattr(folder_module, "scan", fake_scan)
    monkeypatch.setattr(folder_module, "find_folder_repeats", _cancelled)

    with pytest.raises(Cancelled):
        scan_folder([tmp_path], SETTINGS)
    assert closed == ["e1.mp4", "e2.mp4"]


def test_cancelling_an_addition_keeps_the_set_s_own_footage_open(tmp_path, monkeypatch):
    """The videos already in the set are still on screen: releasing their
    footage under them would take the descriptor out of an export."""
    closed = []
    (tmp_path / "e2.mp4").write_bytes(b"x")
    held = replace(
        scanned_at(tmp_path / "e1.mp4"),
        source=SimpleNamespace(close=lambda: closed.append("e1.mp4")),
    )

    def fake_scan(path, **kwargs):
        return replace(
            scanned_at(path), source=SimpleNamespace(close=lambda: closed.append("e2.mp4"))
        )

    monkeypatch.setattr(folder_module, "scan", fake_scan)
    monkeypatch.setattr(folder_module, "find_folder_repeats", _cancelled)

    with pytest.raises(Cancelled):
        add_videos(FolderScan(videos=[held], settings=SETTINGS), [tmp_path / "e2.mp4"])
    assert closed == ["e2.mp4"]


def _cancelled(*args, **kwargs):
    raise Cancelled()


def test_removing_a_video_takes_its_cards_out_of_the_cast(season):
    before = {p.label: p.detection_count for p in cast_of(season)[0]}
    assert before["Person #1"] == 70  # the lead, in e1 and e2

    shrunk = without_video(season, Path("/season/e2.mp4"))

    assert [r.video_path.name for r in shrunk.videos] == ["e1.mp4", "e3.mp4"]
    cast, _ = cast_of(shrunk)
    assert [p.detection_count for p in cast] == [40, 10, 10]
    assert all(len(p.videos) == 1 for p in cast)


def test_removing_a_video_renumbers_the_repeats(season):
    from app.video.repeats import Repeat

    # e3 repeats e1's opening, and e2 repeats e1's.
    season.repeats = {(2, 0): [Repeat(0.0, 30.0, 0.0, 60)], (1, 0): [Repeat(0.0, 5.0, 0.0, 10)]}

    shrunk = without_video(season, Path("/season/e2.mp4"))

    # What was video 2 is video 1 now, and e2's own repeat is gone with it.
    assert list(shrunk.repeats) == [(1, 0)]
    assert shrunk.repeats[(1, 0)][0].end == 30.0


def test_removing_a_video_releases_its_footage(season):
    closed = []
    season.videos[1] = replace(
        season.videos[1], source=SimpleNamespace(close=lambda: closed.append("e2"))
    )

    without_video(season, Path("/season/e2.mp4"))

    assert closed == ["e2"]


def test_the_only_video_left_cannot_be_removed(season):
    one = FolderScan(videos=[season.videos[0]], settings=SETTINGS)

    with pytest.raises(EditError):
        without_video(one, Path("/season/e1.mp4"))


def test_a_video_that_is_not_in_the_set_cannot_be_removed(season):
    with pytest.raises(EditError):
        without_video(season, Path("/season/e9.mp4"))


def test_removing_a_video_that_could_not_be_read_takes_it_off_the_list(season):
    season.skipped = [(Path("/season/broken.mp4"), "cannot read it")]

    shrunk = without_video(season, Path("/season/broken.mp4"))

    assert shrunk.skipped == []
    assert len(shrunk.videos) == 3


def test_an_answer_given_before_a_video_was_added_still_holds(on_disk, tmp_path):
    """Answers are kept beside the scans, keyed by card; the ones given in
    this window must survive a video joining the set, whether or not they
    reached the cache."""
    from app.ui.folder import merged_answers

    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 1), same=False)
    added = tmp_path / "season" / "e3.mp4"
    added.write_bytes(b"e3")
    grown = replace(on_disk, videos=on_disk.videos + [scanned_at(added)])

    merged = merged_answers(grown, answers)

    assert frozenset({CardRef(0, 0), CardRef(1, 1)}) in merged.different


# ------------------------------------------------------------- their order


def test_the_set_is_put_in_the_order_asked_for(season):
    moved, places = reordered(
        season,
        [Path("/season/e3.mp4"), Path("/season/e1.mp4"), Path("/season/e2.mp4")],
    )

    assert [r.video_path.name for r in moved.videos] == ["e3.mp4", "e1.mp4", "e2.mp4"]
    assert places == {0: 1, 1: 2, 2: 0}
    # The set it was asked of is untouched, as every FolderScan is.
    assert [r.video_path.name for r in season.videos] == ["e1.mp4", "e2.mp4", "e3.mp4"]


def test_a_video_the_order_says_nothing_about_keeps_its_place_behind(season):
    """The videos view's list can hold a video that has not been scanned
    into the set yet, so an order can name fewer videos than the set holds."""
    moved, places = reordered(season, [Path("/season/e2.mp4")])

    assert [r.video_path.name for r in moved.videos] == ["e2.mp4", "e1.mp4", "e3.mp4"]
    assert places == {0: 1, 1: 0, 2: 2}


def test_an_order_that_changes_nothing_changes_nothing(season):
    moved, places = reordered(season, [r.video_path for r in season.videos])

    assert [r.video_path.name for r in moved.videos] == ["e1.mp4", "e2.mp4", "e3.mp4"]
    assert places == {0: 0, 1: 1, 2: 2}


def test_the_reel_is_joined_in_the_set_s_new_order(season, monkeypatch):
    monkeypatch.setattr(
        folder_module, "plan_export", lambda cards, **kwargs: (None, [AppearanceInterval(1.0, 2.0)])
    )
    person = cast_of(season)[0][0]  # the lead, in e1 and e2

    moved, places = reordered(season, [Path("/season/e2.mp4"), Path("/season/e1.mp4")])
    renumbered = folder_module.renumbered_people([person], places)[0]

    assert [r.video_path.name for r, _ in plan_cast_export(season, person)] == ["e1.mp4", "e2.mp4"]
    assert [r.video_path.name for r, _ in plan_cast_export(moved, renumbered)] == [
        "e2.mp4",
        "e1.mp4",
    ]


def test_a_repeat_whose_videos_swap_is_told_the_other_way_round(season):
    """A repeat is footage two videos share, and which of them is the later
    one is exactly what reordering changes. It is turned round rather than
    looked for again, which would mean reading both videos."""
    from app.video.repeats import Repeat

    # e2's 0-30s is e1's 10-40s: t in e2 is t + 10 in e1.
    season.repeats = {(1, 0): [Repeat(0.0, 30.0, 10.0, 60)]}

    moved, _ = reordered(season, [Path("/season/e2.mp4"), Path("/season/e1.mp4")])

    # e1 is the later one now, and the same footage is its 10-40s.
    assert list(moved.repeats) == [(1, 0)]
    repeat = moved.repeats[(1, 0)][0]
    assert (repeat.start, repeat.end, repeat.offset) == (10.0, 40.0, -10.0)
    assert repeat.frames == 60


def test_a_repeat_whose_videos_keep_their_order_is_untouched(season):
    from app.video.repeats import Repeat

    season.repeats = {(2, 0): [Repeat(0.0, 30.0, 0.0, 60)]}

    moved, _ = reordered(season, [Path("/season/e1.mp4"), Path("/season/e3.mp4")])

    # e3 is video 1 now, and still the later of the two.
    assert list(moved.repeats) == [(1, 0)]
    assert moved.repeats[(1, 0)][0].offset == 0.0


def test_the_reel_still_leaves_out_what_is_already_in_it_after_a_reorder(season, monkeypatch):
    """The order decides which of two videos loses the footage they share:
    whichever plays second."""
    from app.video.repeats import Repeat

    monkeypatch.setattr(
        folder_module,
        "plan_export",
        lambda cards, **kwargs: (None, [AppearanceInterval(10.0, 12.0)]),
    )
    # e1's and e2's 0-60s are the same footage at no offset, so the lead at
    # 10-12s is the same two seconds in both.
    season.repeats = {(1, 0): [Repeat(0.0, 60.0, 0.0, 120)]}
    person = cast_of(season)[0][0]

    before = plan_cast_export(season, person)
    moved, places = reordered(season, [Path("/season/e2.mp4"), Path("/season/e1.mp4")])
    after = plan_cast_export(moved, folder_module.renumbered_people([person], places)[0])

    # Either way one video carries the footage and the other does not: the
    # one that plays first keeps it.
    assert [r.video_path.name for r, _ in before] == ["e1.mp4"]
    assert [r.video_path.name for r, _ in after] == ["e2.mp4"]


def test_the_answers_move_with_the_videos_they_are_about(season):
    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 1), same=True)
    answers.record(CardRef(2, 0), CardRef(0, 1), same=False)

    _, places = reordered(season, [Path("/season/e3.mp4")])
    moved = folder_module.renumbered_answers(answers, places)

    # e3 is video 0 now; e1 is video 1 and e2 video 2.
    assert moved.same == {frozenset({CardRef(1, 0), CardRef(2, 1)})}
    assert moved.different == {frozenset({CardRef(0, 0), CardRef(1, 1)})}


# --------------------------------------------------------------------- cast


def test_the_cast_is_one_card_per_person_across_videos(season):
    cast, questions = cast_of(season)

    lead = cast[0]
    assert lead.detection_count == 70
    assert lead.videos == [0, 1]
    assert [(v, p.index) for v, p in lead.appearances] == [(0, 0), (1, 1)]
    assert {tuple(p.videos) for p in cast} == {(0, 1), (2,)}
    assert questions == []
    assert [p.index for p in cast] == list(range(len(cast)))


def test_a_name_on_any_card_names_the_person(season):
    season.videos[1] = result("e2.mp4", person(0, FRIEND), person(1, LEAD, name="Lead"))
    cast, _ = cast_of(season)

    assert cast[0].label == "Lead"
    assert cast[1].label == "Person #2"


def test_answers_change_the_cast(season):
    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 1), same=False)

    cast, _ = cast_of(season, answers)

    # Two people with the lead's face, one per video, rather than one of 70.
    assert sorted(p.detection_count for p in cast if 70 >= p.detection_count >= 30) == [30, 40]


# ------------------------------------------------------------------- export


def test_the_plan_takes_each_video_s_own_segments(season, monkeypatch):
    calls = []

    def fake_plan(cards, video_duration, sample_interval, settings):
        calls.append(([c.index for c in cards], video_duration))
        return [], [AppearanceInterval(1.0, 2.0)]

    monkeypatch.setattr(folder_module, "plan_export", fake_plan)
    cast, _ = cast_of(season)

    plans = plan_cast_export(season, cast[0])

    assert [r.video_path.name for r, _ in plans] == ["e1.mp4", "e2.mp4"]
    assert calls == [([0], 60.0), ([1], 60.0)]


@pytest.fixture
def cutting(monkeypatch):
    state = SimpleNamespace(clips=None, rates={})

    monkeypatch.setattr(
        folder_module,
        "plan_export",
        lambda cards, **kwargs: ([], [AppearanceInterval(1.0, 3.0)]),
    )
    monkeypatch.setattr(
        folder_module,
        "probe_clip",
        lambda source, include_audio=True: SimpleNamespace(
            frame_rate=state.rates.get(Path(source).name, 24)
        ),
    )

    def fake_cut(clips, output_path, on_segment=None, **kwargs):
        state.clips = clips
        for i in range(len(clips)):
            on_segment(i, len(clips), None)
        return CutResult(output_path, len(clips), 2.0 * len(clips), 1.0)

    monkeypatch.setattr(folder_module, "cut_clips", fake_cut)
    return state


def test_one_reel_is_cut_from_every_video_the_person_is_in(season, cutting, tmp_path):
    cast, _ = cast_of(season)
    progress = []

    cut, left_out = export_cast(
        season, cast[0], tmp_path / "lead.mp4",
        on_progress=lambda fraction, done, total: progress.append((done, total)),
    )

    assert [Path(c.video).name for c in cutting.clips] == ["e1.mp4", "e2.mp4"]
    assert left_out == []
    assert progress == [(1, 2), (2, 2)]
    assert cut.output_path == tmp_path / "lead.mp4"


def test_a_video_at_another_frame_rate_is_in_the_reel(season, cutting, tmp_path):
    """It used to be left out; the cut converts it now."""
    cutting.rates = {"e2.mp4": 25}
    cast, _ = cast_of(season)

    _, left_out = export_cast(season, cast[0], tmp_path / "lead.mp4")

    assert [Path(c.video).name for c in cutting.clips] == ["e1.mp4", "e2.mp4"]
    assert left_out == []


def test_a_video_that_cannot_be_read_is_left_out_by_name(season, cutting, monkeypatch, tmp_path):
    def probe(source, include_audio=True):
        if Path(source).name == "e2.mp4":
            raise CutterError("e2.mp4 has no video stream.")
        return SimpleNamespace(frame_rate=24)

    monkeypatch.setattr(folder_module, "probe_clip", probe)
    cast, _ = cast_of(season)

    _, left_out = export_cast(season, cast[0], tmp_path / "lead.mp4")

    assert [Path(c.video).name for c in cutting.clips] == ["e1.mp4"]
    assert left_out == [(Path("/season/e2.mp4"), "e2.mp4 has no video stream.")]


def test_cancelling_an_export_stops_it(season, cutting, tmp_path):
    cancel = __import__("threading").Event()
    cancel.set()
    cast, _ = cast_of(season)

    with pytest.raises(Cancelled):
        export_cast(season, cast[0], tmp_path / "lead.mp4", cancel=cancel)


def test_nothing_to_cut_is_an_error_not_an_empty_file(season, cutting, monkeypatch, tmp_path):
    monkeypatch.setattr(folder_module, "plan_export", lambda cards, **kwargs: ([], []))
    cast, _ = cast_of(season)

    with pytest.raises(CutterError, match="nothing to cut"):
        export_cast(season, cast[0], tmp_path / "lead.mp4")


# ------------------------------------------------------------------- naming


def test_naming_a_person_names_their_card_in_every_video(season, monkeypatch):
    edits = []

    def fake_edit(scan_result, settings, operation, indexes, name=None):
        edits.append((scan_result.video_path.name, operation, indexes, name))
        people = [
            Person(**{**p.__dict__, "name": name}) if p.index in indexes else p
            for p in scan_result.people
        ]
        return ScanResult(**{**scan_result.__dict__, "people": people})

    monkeypatch.setattr(folder_module, "apply_edit", fake_edit)
    cast, _ = cast_of(season)

    named = name_cast_person(season, cast[0], "Lead")

    assert edits == [
        ("e1.mp4", "rename", [0], "Lead"),
        ("e2.mp4", "rename", [1], "Lead"),
    ]
    renamed, _ = cast_of(named)
    assert renamed[0].label == "Lead"
    # The folder it came from is not changed underneath the window.
    assert cast_of(season)[0][0].name is None


# ------------------------------------------------------------------ preview


def test_the_preview_draws_on_every_video(season, monkeypatch):
    monkeypatch.setattr(
        folder_module,
        "preview_frames",
        lambda result, cards, limit: [
            (float(i), Image.new("RGB", (4, 4))) for i in range(limit)
        ],
    )
    cast, _ = cast_of(season)

    frames = cast_preview_frames(season, cast[0], limit=6)

    assert [name for name, _, _ in frames] == ["e1.mp4"] * 3 + ["e2.mp4"] * 3


# ------------------------------------------------------------------ repeats


def test_a_person_s_reel_leaves_out_what_an_earlier_video_already_showed(season, monkeypatch):
    from app.ui.folder import repeated_seconds
    from app.video.repeats import Repeat

    monkeypatch.setattr(
        folder_module, "plan_export",
        lambda cards, video_duration, **kwargs: ([], [AppearanceInterval(10.0, 20.0)]),
    )
    # e2 opens with e1's footage from 0s on, so e2's 10-20s is e1's 10-20s.
    season.repeats = {(1, 0): [Repeat(0.0, 30.0, 0.0, 60)]}
    lead = cast_of(season)[0][0]

    plans = plan_cast_export(season, lead)

    assert [r.video_path.name for r, _ in plans] == ["e1.mp4"]
    assert repeated_seconds(season, lead) == pytest.approx(10.0)


def test_repeats_are_looked_for_once_the_videos_are_scanned(tmp_path, monkeypatch):
    from app.video.repeats import Repeat

    for name in ("e1.mp4", "e2.mp4"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(folder_module, "scan", lambda path, **kwargs: result(path.name))
    monkeypatch.setattr(folder_module, "fingerprint_kept", lambda path, directory: path.name)
    compared = []

    def fake_find(later, earlier):
        compared.append((later, earlier))
        return [Repeat(0.0, 10.0, 0.0, 20)]

    monkeypatch.setattr(folder_module, "find_repeats", fake_find)
    statuses = []

    scanned = scan_folder([tmp_path], SETTINGS, on_status=statuses.append)

    assert compared == [("e2.mp4", "e1.mp4")]
    assert list(scanned.repeats) == [(1, 0)]
    assert len(statuses) == 2


def test_a_video_that_cannot_be_fingerprinted_is_never_taken_for_a_repeat(tmp_path, monkeypatch):
    for name in ("e1.mp4", "e2.mp4"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(folder_module, "scan", lambda path, **kwargs: result(path.name))

    def broken(path, directory):
        raise RuntimeError("cannot decode")

    monkeypatch.setattr(folder_module, "fingerprint_kept", broken)

    assert scan_folder([tmp_path], SETTINGS).repeats == {}


# ------------------------------------------------------- remembered answers


@pytest.fixture
def on_disk(tmp_path):
    """The season fixture, with videos that exist so their scans have keys."""
    folder = tmp_path / "season"
    folder.mkdir()
    videos = []
    for name, people in (
        ("e1.mp4", (person(0, LEAD, detections=40), person(1, FRIEND))),
        ("e2.mp4", (person(0, FRIEND), person(1, LEAD, detections=30))),
    ):
        (folder / name).write_bytes(name.encode())
        videos.append(ScanResult(video_path=folder / name, video_duration=60.0,
                                 sample_interval=0.5, people=list(people)))
    return FolderScan(videos=videos, settings=SETTINGS)


def test_an_answer_holds_the_next_time_the_folder_is_opened(on_disk):
    from app.ui.folder import load_answers, remember_answer

    assert remember_answer(on_disk, CardRef(0, 0), CardRef(1, 1), same=False)

    reopened = load_answers(on_disk)

    assert frozenset({CardRef(0, 0), CardRef(1, 1)}) in reopened.different
    cast, _ = cast_of(on_disk, reopened)
    # The lead's two cards stay apart; the friend is still linked.
    assert sorted(p.detection_count for p in cast) == [20, 30, 40]


def test_changing_an_answer_replaces_the_one_kept(on_disk):
    from app.ui.folder import load_answers, remember_answer

    remember_answer(on_disk, CardRef(0, 0), CardRef(1, 1), same=False)
    remember_answer(on_disk, CardRef(1, 1), CardRef(0, 0), same=True)

    reopened = load_answers(on_disk)
    assert frozenset({CardRef(0, 0), CardRef(1, 1)}) in reopened.same
    assert not reopened.different


def test_an_answer_follows_the_card_not_its_place_in_the_gallery(on_disk):
    """A correction renumbers the gallery; the answer must stay with the
    card it was about, not land on whichever card took its number."""
    from app.ui.folder import load_answers, remember_answer

    remember_answer(on_disk, CardRef(0, 0), CardRef(1, 1), same=False)
    e2 = on_disk.videos[1]
    renumbered = ScanResult(
        **{**e2.__dict__, "people": [
            Person(**{**e2.people[1].__dict__, "index": 0}),
            Person(**{**e2.people[0].__dict__, "index": 1}),
        ]}
    )
    moved = FolderScan(videos=[on_disk.videos[0], renumbered], settings=SETTINGS)

    reopened = load_answers(moved)

    assert frozenset({CardRef(0, 0), CardRef(1, 0)}) in reopened.different


def test_an_answer_about_a_card_since_changed_no_longer_applies(on_disk):
    from app.ui.folder import load_answers, remember_answer

    remember_answer(on_disk, CardRef(0, 0), CardRef(1, 1), same=False)
    e1 = on_disk.videos[0]
    merged_since = ScanResult(**{**e1.__dict__, "people": [
        Person(**{**e1.people[0].__dict__, "detection_count": 55}), e1.people[1],
    ]})

    reopened = load_answers(FolderScan(videos=[merged_since, on_disk.videos[1]], settings=SETTINGS))

    assert not reopened.different


def test_a_damaged_answers_file_is_ignored_not_fatal(on_disk):
    from app.scans import scan_cache_dir
    from app.ui.folder import ANSWERS_FILE, load_answers

    scan_cache_dir().mkdir(parents=True, exist_ok=True)
    (scan_cache_dir() / ANSWERS_FILE).write_text("{not json")

    assert not load_answers(on_disk).same


# -------------------------------------------------------------- corrections


def test_detaching_a_named_card_clears_its_copy_of_the_name(season, monkeypatch):
    """A shared name would put the card straight back: cards with one name
    are one person whatever their faces score."""
    from app.ui.folder import detach_cards

    season.videos[0] = result("e1.mp4", person(0, LEAD, name="Lead", detections=40), person(1, FRIEND))
    season.videos[1] = result("e2.mp4", person(0, FRIEND), person(1, LEAD, name="Lead", detections=30))
    renamed = []

    def fake_edit(scan_result, settings, operation, indexes, name=None):
        renamed.append((scan_result.video_path.name, operation, indexes, name))
        people = [Person(**{**p.__dict__, "name": name}) if p.index in indexes else p
                  for p in scan_result.people]
        return ScanResult(**{**scan_result.__dict__, "people": people})

    monkeypatch.setattr(folder_module, "apply_edit", fake_edit)
    answers = Answers()
    lead = cast_of(season)[0][0]

    split = detach_cards(season, answers, lead, [CardRef(1, 1)])

    assert renamed == [("e2.mp4", "rename", [1], "")]
    cast, _ = cast_of(split, answers)
    assert sorted(p.detection_count for p in cast if p.detection_count >= 30) == [30, 40]


def test_discarding_never_empties_a_video(season, monkeypatch):
    from app.faces.edits import EditError
    from app.ui.folder import discard_people

    def fake_edit(scan_result, settings, operation, indexes, name=None):
        if scan_result.video_path.name == "e3.mp4":
            raise EditError("That would discard everyone this scan found.")
        return scan_result

    monkeypatch.setattr(folder_module, "apply_edit", fake_edit)
    cast, _ = cast_of(season)
    stranger = next(p for p in cast if p.videos == [2])

    _, refused = discard_people(season, [stranger])

    assert refused == ["e3.mp4"]


def test_several_people_together_are_one_reel_of_either():
    from app.ui.folder import CastPerson, together

    a = CastPerson(0, "Lead", [(0, person(0, LEAD))], 40)
    b = CastPerson(1, None, [(1, person(0, FRIEND))], 10)

    both = together([a, b])

    assert both.name == "Lead and Person #2"
    assert both.detection_count == 50
    assert both.videos == [0, 1]


def test_merging_overrides_an_earlier_split(season):
    """An old 'different' between their cards vetoed the join, and the
    merge did nothing without saying so."""
    from app.ui.folder import join_people

    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 1), same=False)
    apart, _ = cast_of(season, answers)
    leads = [p for p in apart if p.detection_count in (40, 30)]
    assert len(leads) == 2

    join_people(season, answers, leads)

    cast, _ = cast_of(season, answers)
    assert any(p.detection_count == 70 for p in cast)
