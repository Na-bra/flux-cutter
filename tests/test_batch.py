"""Cutting one person out of a folder of videos.

None of this decodes anything: the batch's own job is choosing what to run
on, carrying on past what fails, and reporting what it did. The scan and
the cut it delegates to are tested where they live.
"""

from contextlib import contextmanager
from pathlib import Path

import pytest

from app import main as app_main
from app.faces.reference import ReferenceFace
from app.main import BatchOutcome, batch_output_path, collect_videos, run_batch
from app.video.loader import VideoLoadError

import numpy as np


def video(folder: Path, name: str) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pretend footage")
    return path


@pytest.fixture
def reference(tmp_path):
    return ReferenceFace(
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        embedding_space="arcface-w600k-r50",
        detection=None,
        source=tmp_path / "them.jpg",
        face_count=1,
    )


# ------------------------------------------------------- choosing the videos


def test_a_folder_becomes_the_videos_inside_it(tmp_path):
    video(tmp_path, "b.mp4")
    video(tmp_path, "a.MOV")
    video(tmp_path, "notes.txt")
    video(tmp_path, "poster.jpg")

    found = collect_videos([tmp_path])

    assert [path.name for path in found] == ["a.MOV", "b.mp4"]


def test_folders_are_searched_one_level_deep_unless_asked_otherwise(tmp_path):
    video(tmp_path, "top.mp4")
    video(tmp_path / "season-2", "buried.mp4")

    assert [p.name for p in collect_videos([tmp_path])] == ["top.mp4"]
    assert [p.name for p in collect_videos([tmp_path], recursive=True)] == [
        "buried.mp4",
        "top.mp4",
    ]


def test_naming_a_folder_and_a_file_in_it_does_not_cut_the_reel_twice(tmp_path):
    inside = video(tmp_path, "episode.mp4")

    assert collect_videos([tmp_path, inside]) == [inside]


def test_a_named_file_is_taken_as_given(tmp_path):
    """Even one this app would not open: refusing it belongs to the loader,
    which says why, rather than to a filter that drops it silently."""
    odd = video(tmp_path, "episode.mkv")

    assert collect_videos([odd]) == [odd]


def test_a_reel_is_named_after_the_video_it_came_from(tmp_path):
    assert (
        batch_output_path(Path("/footage/S01E03.mp4"), tmp_path).name
        == "S01E03-reel.mp4"
    )


# ------------------------------------------------------------- running them


class FakeExport:
    def __init__(self, path, seconds=12.0):
        self.output_path = path
        self.exported_seconds = seconds
        self.segment_count = 3
        self.encode_seconds = 1.0


@contextmanager
def fake_container(path):
    yield object()


def test_every_video_is_processed_and_reported(tmp_path, reference, monkeypatch):
    folder = tmp_path / "season"
    video(folder, "e1.mp4")
    video(folder, "e2.mp4")
    out = tmp_path / "reels"

    monkeypatch.setattr(app_main, "load_video", fake_container)
    monkeypatch.setattr(
        app_main,
        "run_export",
        lambda container, **kwargs: FakeExport(kwargs["output_path"]),
    )

    outcomes = run_batch([folder], reference, out, export_settings={})

    assert [o.video_path.name for o in outcomes] == ["e1.mp4", "e2.mp4"]
    assert all(o.succeeded for o in outcomes)
    assert [Path(o.output_path).name for o in outcomes] == ["e1-reel.mp4", "e2-reel.mp4"]
    assert out.is_dir()


def test_one_bad_episode_does_not_end_the_season(tmp_path, reference, monkeypatch):
    """The failure mode that would make this unusable: nineteen good scans
    thrown away because episode three is unreadable or holds nobody."""
    folder = tmp_path / "season"
    video(folder, "e1.mp4")
    video(folder, "e2.mp4")
    video(folder, "e3.mp4")

    def flaky(container, **kwargs):
        name = kwargs["video_path"].name
        if name == "e1.mp4":
            raise app_main.SelectionError("nobody here matches")
        if name == "e2.mp4":
            raise VideoLoadError("could not open")
        return FakeExport(kwargs["output_path"], seconds=20.0)

    monkeypatch.setattr(app_main, "load_video", fake_container)
    monkeypatch.setattr(app_main, "run_export", flaky)

    outcomes = run_batch([folder], reference, tmp_path / "reels", export_settings={})

    assert [o.succeeded for o in outcomes] == [False, False, True]
    assert "nobody here matches" in outcomes[0].skipped_because
    assert "could not open" in outcomes[1].skipped_because
    assert outcomes[2].reel_seconds == 20.0


def test_a_video_with_nothing_worth_cutting_is_a_skip_not_a_crash(
    tmp_path, reference, monkeypatch
):
    """run_export returns nothing when there were no frames, no identities
    or no segments -- all normal, none of them an error."""
    folder = tmp_path / "season"
    video(folder, "quiet.mp4")

    monkeypatch.setattr(app_main, "load_video", fake_container)
    monkeypatch.setattr(app_main, "run_export", lambda container, **kwargs: None)

    outcomes = run_batch([folder], reference, tmp_path / "reels", export_settings={})

    assert outcomes == [
        BatchOutcome(
            video_path=folder / "quiet.mp4",
            skipped_because="nothing to cut for this person",
        )
    ]


def test_an_empty_folder_produces_nothing_and_says_so(tmp_path, reference, capsys):
    assert run_batch([tmp_path], reference, tmp_path / "out", export_settings={}) == []
    assert "No videos found" in capsys.readouterr().err


def test_the_same_reference_is_used_for_every_video(tmp_path, reference, monkeypatch):
    """Cross-video identity is the photo, not a centroid carried forward:
    matching each episode against the same vector is what makes it work."""
    folder = tmp_path / "season"
    video(folder, "e1.mp4")
    video(folder, "e2.mp4")
    seen = []

    monkeypatch.setattr(app_main, "load_video", fake_container)

    def record(container, **kwargs):
        seen.append(kwargs["reference"])
        return FakeExport(kwargs["output_path"])

    monkeypatch.setattr(app_main, "run_export", record)
    run_batch([folder], reference, tmp_path / "reels", export_settings={})

    assert len(seen) == 2
    assert all(item is reference for item in seen)
