"""Tests for the window itself.

Unlike tests/test_ui_worker.py these need Tk, so they skip rather than fail
where there is no display. They exist because three real bugs in this file
were only reachable by driving widgets: a filename that named the wrong
video, a gallery that accepted clicks during an export, and settings that
stayed live while the job using them ran.

They build a synthetic ScanResult instead of scanning a video, so they cost
milliseconds rather than the ten seconds a real scan takes.
"""

from pathlib import Path

import numpy as np
import pytest

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.ui.worker import Person, ScanResult

pytest.importorskip("customtkinter")

import tkinter  # noqa: E402

from PIL import Image  # noqa: E402

from app.ui.app import FluxCutterApp  # noqa: E402


def make_person(index: int) -> Person:
    observations = [
        FaceObservation(
            embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
            detection=FaceDetection(
                box=BoundingBox(x_min=0, y_min=0, x_max=80, y_max=80),
                confidence=0.9,
            ),
            face_crop=np.zeros((80, 80, 3), dtype=np.uint8),
            source_timestamp=timestamp,
        )
        for timestamp in (10.0, 10.5, 11.0)
    ]
    return Person(
        index=index,
        thumbnail=Image.new("RGB", (32, 32)),
        detection_count=len(observations),
        first_seen=10.0,
        last_seen=11.0,
        group=FaceIdentityGroup(group_id=index, observations=observations),
    )


def make_scan_result(video_path: str = "/videos/documentary.mp4") -> ScanResult:
    return ScanResult(
        video_path=Path(video_path),
        video_duration=120.0,
        sample_interval=0.5,
        people=[make_person(0), make_person(1), make_person(2)],
        frame_count=240,
        detection_count=9,
        min_detections=6,
    )


class _AliveWorker:
    """Stands in for a running job without starting a thread."""

    @staticmethod
    def is_alive() -> bool:
        return True


@pytest.fixture
def app():
    try:
        window = FluxCutterApp()
    except tkinter.TclError as error:  # pragma: no cover - depends on the host
        pytest.skip(f"no display available: {error}")
    window.update_idletasks()
    yield window
    window._on_close()


def test_scan_results_become_one_card_per_person(app):
    app._show_people(make_scan_result())

    assert len(app._cards) == 3
    assert [card.person.index for card in app._cards] == [0, 1, 2]


def test_selecting_a_face_enables_export_and_previews_the_cut(app):
    app._show_people(make_scan_result())
    app._cards[1]._clicked()

    assert app._selected.index == 1
    assert app._export_button.cget("state") == "normal"
    assert "Person #2 selected" in app._selection_label.cget("text")
    assert "cuts" in app._selection_label.cget("text")


def test_filename_is_named_after_the_scanned_video_not_the_path_box(app):
    """The box can be edited after a scan; the export still cuts the old video.

    Reading the box here produced `no_faces-person-2.mp4` for a reel cut
    entirely from `test.mp4`.
    """
    app._show_people(make_scan_result("/videos/documentary.mp4"))
    app._video_var.set("/somewhere/else/holiday.mp4")

    app._cards[1]._clicked()

    assert app._filename_var.get() == "documentary-person-2.mp4"


def test_a_hand_typed_filename_survives_changing_the_selection(app):
    app._show_people(make_scan_result())
    app._cards[0]._clicked()
    app._filename_var.set("my-own-name.mp4")

    app._cards[2]._clicked()

    assert app._filename_var.get() == "my-own-name.mp4"


def test_an_untouched_filename_follows_the_selection(app):
    app._show_people(make_scan_result())
    app._cards[0]._clicked()
    assert app._filename_var.get() == "documentary-person-1.mp4"

    app._cards[2]._clicked()

    assert app._filename_var.get() == "documentary-person-3.mp4"


def test_output_path_joins_the_folder_and_the_name(app):
    app._output_dir_var.set("/tmp/reels")
    app._filename_var.set("out.mp4")

    assert app._output_path() == Path("/tmp/reels/out.mp4")


def test_output_path_supplies_a_missing_extension(app):
    app._output_dir_var.set("/tmp/reels")
    app._filename_var.set("no-extension")

    assert app._output_path() == Path("/tmp/reels/no-extension.mp4")


def test_the_gallery_ignores_clicks_while_a_job_runs(app):
    """A running export holds its own person; the display must not disagree.

    Clicking mid-export used to relabel the window and rename the output
    field to describe someone the encode was not cutting.
    """
    app._show_people(make_scan_result())
    app._cards[0]._clicked()
    app._worker = _AliveWorker()

    app._cards[2]._clicked()

    assert app._selected.index == 0
    assert app._filename_var.get() == "documentary-person-1.mp4"
    assert "Person #1 selected" in app._selection_label.cget("text")


def test_every_setting_freezes_while_a_job_runs(app):
    """A control that still moves claims an influence it does not have."""
    app._set_busy(True, scanning=False)

    for widget in (app._interval_menu, app._encoder_menu, app._quality_menu):
        assert widget.cget("state") == "disabled"

    app._set_busy(False)

    for widget in (app._interval_menu, app._encoder_menu, app._quality_menu):
        assert widget.cget("state") == "normal"


def test_a_scan_finding_nobody_explains_the_cutoff(app):
    app._show_people(
        ScanResult(
            video_path=Path("/videos/empty.mp4"),
            video_duration=60.0,
            sample_interval=0.5,
            people=[],
            min_detections=6,
        )
    )

    assert app._cards == []
    assert "at least 6 detections" in app._empty_label.cget("text")
