"""Tests for the window's Python side.

These replace tests/test_ui_app.py, which drove Tk widgets. They cost
nothing to run and need no display, because the bridge holds every decision
and the page only renders what it is told -- which is the point of keeping
the bridge thin.

They exist because three real bugs lived in the old window and were only
reachable by driving it: a filename that named the wrong video, a gallery
that accepted clicks during an export, and an export that started against
footage that had moved. Each of those has a test here.

A synthetic ScanResult stands in for a real scan, so these are
milliseconds rather than the ten seconds a scan of the sample clip takes.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.ui.worker import Person, ScanResult

pytest.importorskip("webview")

from app.ui import web  # noqa: E402


# ------------------------------------------------------------------ doubles


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


def make_scan_result(video_path: str = "/videos/documentary.mp4", source=None) -> ScanResult:
    return ScanResult(
        source=source,
        video_path=Path(video_path),
        video_duration=120.0,
        sample_interval=0.5,
        people=[make_person(0), make_person(1), make_person(2)],
        frame_count=240,
        detection_count=9,
        min_detections=6,
    )


class FakeWindow:
    """Records what the bridge would have told the page to do."""

    def __init__(self, dialog=None, confirm=True):
        self.calls = []
        self._dialog = dialog
        self._confirm = confirm

    def evaluate_js(self, script):
        self.calls.append(script)

    def create_file_dialog(self, *args, **kwargs):
        return self._dialog

    def create_confirmation_dialog(self, title, message):
        return self._confirm

    def emitted(self, function):
        """The payload passed to `function`, or None if it was never called."""
        for call in self.calls:
            if call.startswith(function + "("):
                body = call[len(function) + 1 : -1]
                return json.loads(body) if body else {}
        return None


class AliveWorker:
    """Stands in for a running job without starting a thread."""

    @staticmethod
    def is_alive() -> bool:
        return True


@pytest.fixture
def bridge():
    made = web.Bridge()
    made.window = FakeWindow()
    return made


# -------------------------------------------------------------- the gallery


def test_a_scan_becomes_one_card_per_person(bridge):
    payload = bridge._scan_payload(make_scan_result())

    assert len(payload["people"]) == 3
    assert [p["index"] for p in payload["people"]] == [0, 1, 2]


def test_thumbnails_travel_as_data_uris(bridge):
    """The page cannot read a PIL object, and there is no file to serve."""
    payload = bridge._scan_payload(make_scan_result())

    for person in payload["people"]:
        assert person["thumbnail"].startswith("data:image/jpeg;base64,")
        assert len(person["thumbnail"]) > 100


def test_selecting_a_face_previews_the_cut(bridge):
    bridge._scan_result = make_scan_result()

    answer = bridge.select_person(1, "reel.mp4")

    assert answer["accepted"] is True
    assert answer["index"] == 1
    assert answer["cuts"] >= 1
    assert ":" in answer["reel"]
    assert "cuts" in answer["summary"]


def test_the_gallery_ignores_clicks_while_a_job_runs(bridge):
    """The running job already holds its own person.

    Letting the click through changed the label and the filename to
    describe someone the encode was not cutting.
    """
    bridge._scan_result = make_scan_result()
    bridge._worker = AliveWorker()

    assert bridge.select_person(1, "reel.mp4") == {"accepted": False}


def test_selecting_someone_who_is_not_there_is_refused(bridge):
    bridge._scan_result = make_scan_result()

    assert bridge.select_person(99, "reel.mp4") == {"accepted": False}


# ------------------------------------------------------------- the filename


def test_the_filename_names_the_scanned_video_not_the_path_box(bridge):
    """The box can have been edited since the scan.

    Export always cuts the scanned footage, so reading the box named the
    file after footage it does not contain.
    """
    bridge._scan_result = make_scan_result("/videos/documentary.mp4")

    answer = bridge.select_person(0, "reel.mp4")

    assert answer["filename"] == "documentary-person-1.mp4"


def test_an_untouched_filename_follows_the_selection(bridge):
    bridge._scan_result = make_scan_result()

    first = bridge.select_person(0, "reel.mp4")["filename"]
    second = bridge.select_person(2, first)["filename"]

    assert first == "documentary-person-1.mp4"
    assert second == "documentary-person-3.mp4"


def test_a_hand_typed_filename_survives_changing_the_selection(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "reel.mp4")

    answer = bridge.select_person(1, "my own name.mp4")

    assert answer["filename"] is None


# ------------------------------------------------------------ where it goes


def test_output_path_joins_the_folder_and_the_name():
    assert web.output_path("/tmp/reels", "one.mp4") == Path("/tmp/reels/one.mp4")


def test_output_path_supplies_a_missing_extension():
    assert web.output_path("/tmp/reels", "one") == Path("/tmp/reels/one.mp4")


def test_output_path_falls_back_when_both_are_blank():
    assert web.output_path("  ", "  ") == web.DEFAULT_OUTPUT_DIR / web.DEFAULT_FILENAME


# ------------------------------------------------------- footage that moved


class MovedSource:
    """A scan's footage handle whose file is no longer reachable."""

    def __init__(self, path="/videos/documentary.mp4", relocate_error=None):
        self.path = Path(path)
        self.relocated_to = None
        self._relocate_error = relocate_error

    def is_available(self):
        return False

    def relocate(self, path):
        if self._relocate_error is not None:
            raise self._relocate_error
        self.relocated_to = path
        self.path = Path(path)

    def close(self):
        pass


def test_a_scan_with_no_source_is_left_alone(bridge):
    bridge._scan_result = make_scan_result(source=None)

    assert bridge._ensure_source_available() is True


def test_export_proceeds_when_the_footage_is_still_there(bridge, tmp_path):
    video = tmp_path / "documentary.mp4"
    video.write_bytes(b"not really a video")

    class Present(MovedSource):
        def is_available(self):
            return True

    bridge._scan_result = make_scan_result(source=Present(str(video)))

    assert bridge._ensure_source_available() is True


def test_declining_to_locate_a_moved_video_stops_the_export(bridge):
    bridge.window = FakeWindow(confirm=False)
    bridge._scan_result = make_scan_result(source=MovedSource())

    assert bridge._ensure_source_available() is False


def test_cancelling_the_file_dialog_stops_the_export(bridge):
    bridge.window = FakeWindow(dialog=None, confirm=True)
    bridge._scan_result = make_scan_result(source=MovedSource())

    assert bridge._ensure_source_available() is False


def test_locating_a_moved_video_lets_the_export_run(bridge, tmp_path):
    found = tmp_path / "documentary.mp4"
    found.write_bytes(b"not really a video")
    source = MovedSource()
    bridge.window = FakeWindow(dialog=(str(found),), confirm=True)
    bridge._scan_result = make_scan_result(source=source)

    assert bridge._ensure_source_available() is True
    assert source.relocated_to == str(found)
    assert bridge.window.emitted("onRelocated")["path"] == str(found)


def test_pointing_at_the_wrong_video_is_refused(bridge, tmp_path):
    from app.video.source import SourceMismatch

    other = tmp_path / "something-else.mp4"
    other.write_bytes(b"different")
    source = MovedSource(relocate_error=SourceMismatch("that is a different file"))
    bridge.window = FakeWindow(dialog=(str(other),), confirm=True)
    bridge._scan_result = make_scan_result(source=source)

    assert bridge._ensure_source_available() is False
    assert bridge.window.emitted("onFailed")["title"] == "Not the same video"


# ------------------------------------------------------------------ scanning


def test_a_scan_needs_a_video_that_exists(bridge):
    answer = bridge.start_scan("/videos/not-here.mp4", "live", 0.5)

    assert answer["started"] is False
    assert "Choose a video" in answer["reason"]


def test_a_second_scan_is_refused_while_one_runs(bridge, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    bridge._worker = AliveWorker()

    assert bridge.start_scan(str(video), "live", 0.5)["started"] is False


def test_export_is_refused_before_anyone_is_chosen(bridge):
    answer = bridge.start_export("/tmp", "reel.mp4", "libx264", "High")

    assert answer["started"] is False
    assert "Choose a person" in answer["reason"]


# --------------------------------------------------------------- the modes


def test_choosing_a_mode_reports_what_it_needs(bridge):
    answer = bridge.set_mode("animation")

    assert bridge._mode == "animation"
    assert answer["status"].startswith("Animation mode.")


def test_an_unknown_mode_is_ignored(bridge):
    """The page is treated as untrusted; a bad value must not change state."""
    before = bridge._mode

    answer = bridge.set_mode("claymation")

    assert bridge._mode == before
    assert answer["status"].startswith("Live Action mode.")


def test_the_window_opens_in_live_action(bridge):
    assert bridge.initial_state()["mode"] == "live"


# ------------------------------------------------------------- the page itself


def test_the_page_ships_with_the_checkout():
    assert (Path(web.__file__).with_name("window.html")).is_file()


def test_the_page_is_readable_and_wired_to_the_bridge():
    """Every function the bridge calls has to exist on the page.

    A renamed callback would otherwise fail silently at runtime -- the
    bridge swallows evaluate_js errors on purpose, because a window that
    closed mid-scan should not raise.
    """
    page = web._page()

    for callback in (
        "onScanProgress",
        "onScanned",
        "onScanCancelled",
        "onExportProgress",
        "onExported",
        "onExportCancelled",
        "onDownload",
        "onRelocated",
        "onFailed",
        "onStatus",
    ):
        assert f"window.{callback} =" in page, f"the page never defines {callback}"


def test_the_frozen_build_reads_the_page_from_the_bundle(tmp_path, monkeypatch):
    """PyInstaller unpacks data files somewhere else entirely."""
    bundled = tmp_path / "bundle"
    bundled.mkdir()
    (bundled / "window.html").write_text("<title>from the bundle</title>")
    monkeypatch.setattr(web.sys, "_MEIPASS", str(bundled), raising=False)

    assert web._page() == "<title>from the bundle</title>"
