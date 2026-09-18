"""A folder in the window: scanning it, its cast, and one reel across it.

The scans, edits and cuts here are stand-ins -- each is tested where it
lives. What is tested is the folder's own job: carrying on past a bad file,
turning cards into a cast, and handing the right segments from the right
videos to the cut.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.faces.cast import Answers, CardRef
from app.faces.grouper import FaceIdentityGroup
from app.ui import folder as folder_module
from app.ui.folder import (
    FolderScan,
    cast_of,
    cast_preview_frames,
    export_cast,
    name_cast_person,
    plan_cast_export,
    scan_folder,
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
    started, progress = [], []

    def fake_scan(path, settings, on_progress, cancel, on_download):
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
