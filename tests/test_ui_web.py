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
import time

import pytest
from PIL import Image

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.ui.worker import (
    ExportSettings,
    Person,
    ScanResult,
    ScanSettings,
    quality_for,
)

pytest.importorskip("webview")

import webview  # noqa: E402

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
        self.dialogs = []
        self._dialog = dialog
        self._confirm = confirm

    def evaluate_js(self, script):
        self.calls.append(script)

    def create_file_dialog(self, *args, **kwargs):
        self.dialogs.append(args[0] if args else None)
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
    assert answer["indexes"] == [1]
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
    both = bridge.select_person(2, first)["filename"]

    assert first == "documentary-person-1.mp4"
    assert both == "documentary-person-1+3.mp4"


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


# ------------------------------------------------------- several at once


def test_a_second_card_joins_the_first_rather_than_replacing_it(bridge):
    """"Every scene either lead is in" is one reel, and a thing people ask
    for. Clicking used to displace the previous choice."""
    bridge._scan_result = make_scan_result()

    bridge.select_person(0, "reel.mp4")
    answer = bridge.select_person(2, "reel.mp4")

    assert answer["indexes"] == [0, 2]
    assert answer["name"] == "People #1 and #3"
    assert "People #1 and #3 selected" in answer["summary"]


def test_clicking_a_chosen_card_again_removes_it(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "reel.mp4")
    bridge.select_person(2, "reel.mp4")

    answer = bridge.select_person(0, "reel.mp4")

    assert answer["indexes"] == [2]


def test_clearing_the_last_card_leaves_nothing_to_export(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(1, "reel.mp4")

    answer = bridge.select_person(1, "reel.mp4")

    assert answer["accepted"] is True
    assert answer["indexes"] == []
    assert bridge.start_export("/tmp", "reel.mp4", "libx264", "Standard") == {
        "started": False,
        "reason": "Choose a person first.",
    }


def test_the_selection_is_named_however_many_there_are(bridge):
    bridge._scan_result = make_scan_result()

    assert bridge.select_person(0, "")["name"] == "Person #1"
    assert bridge.select_person(1, "")["name"] == "People #1 and #2"
    assert bridge.select_person(2, "")["name"] == "People #1, #2 and #3"


def test_a_reel_of_two_people_is_at_least_as_long_as_either_alone(bridge):
    """The union of two appearance timelines cannot be shorter than one."""
    bridge._scan_result = make_scan_result()

    alone = bridge.select_person(0, "")
    bridge.select_person(0, "")
    other = bridge.select_person(2, "")
    together = bridge.select_person(0, "")

    assert together["indexes"] == [0, 2]
    assert together["detections"] == alone["detections"] + other["detections"]
    assert together["reel"] >= max(alone["reel"], other["reel"])


# --------------------------------------------------------- the filmstrip


def test_a_selection_carries_a_token_for_its_preview(bridge):
    bridge._scan_result = make_scan_result()

    first = bridge.select_person(0, "")
    second = bridge.select_person(1, "")

    assert second["token"] > first["token"]


def test_a_late_filmstrip_for_an_old_selection_is_dropped(bridge, monkeypatch):
    """Six seeks can land after the user has clicked again, and drawing
    them would show frames from a reel that is no longer selected."""
    emitted = []
    monkeypatch.setattr(bridge, "_emit", lambda name, payload=None: emitted.append(name))
    bridge._scan_result = make_scan_result()
    bridge._selected = [bridge._scan_result.people[0]]

    def clicked_again_while_seeking(result, chosen):
        bridge._preview_token += 1
        return [(1.0, Image.new("RGB", (4, 3)))]

    monkeypatch.setattr("app.ui.web.preview_frames", clicked_again_while_seeking)
    bridge._start_preview()

    time.sleep(0.2)
    assert "onPreview" not in emitted


def test_a_current_filmstrip_is_drawn(bridge, monkeypatch):
    sent = []
    monkeypatch.setattr(
        bridge, "_emit", lambda name, payload=None: sent.append((name, payload))
    )
    monkeypatch.setattr(
        "app.ui.web.preview_frames",
        lambda result, chosen: [(65.0, Image.new("RGB", (4, 3)))],
    )
    bridge._scan_result = make_scan_result()
    bridge._selected = [bridge._scan_result.people[0]]

    bridge._start_preview()

    for _ in range(50):
        if sent:
            break
        time.sleep(0.01)
    name, payload = sent[0]
    assert name == "onPreview"
    assert payload["token"] == bridge._preview_token
    assert payload["frames"][0]["at"] == "1:05"
    assert payload["frames"][0]["image"].startswith("data:image/jpeg;base64,")


def test_clearing_the_selection_asks_for_no_preview(bridge, monkeypatch):
    asked = []
    monkeypatch.setattr(
        "app.ui.web.preview_frames", lambda result, chosen: asked.append(chosen) or []
    )
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    answer = bridge.select_person(0, "")

    assert answer["indexes"] == []
    assert "token" not in answer


# ------------------------------------------------------------- corrections


def test_merging_two_cards_redraws_the_gallery(bridge, monkeypatch):
    bridge._scan_result = make_scan_result()
    before = len(bridge._scan_result.people)
    bridge.select_person(0, "")
    bridge.select_person(1, "")

    answer = bridge.edit_people("merge")

    assert answer["applied"] is True
    assert len(answer["people"]) == before - 1
    assert "Merged #1, #2" in answer["note"]
    assert len(bridge._scan_result.people) == before - 1


def test_an_edit_clears_the_selection(bridge):
    """The cards are renumbered, so keeping the old indexes selected would
    leave the rail describing whoever now happens to sit at that number."""
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")
    bridge.select_person(1, "")

    bridge.edit_people("merge")

    assert bridge._selected == []


def test_discarding_a_card_removes_it(bridge):
    bridge._scan_result = make_scan_result()
    before = len(bridge._scan_result.people)
    bridge.select_person(1, "")

    answer = bridge.edit_people("discard")

    assert answer["applied"] is True
    assert len(answer["people"]) == before - 1
    assert "Discarded #2" in answer["note"]


def test_an_edit_needs_a_selection(bridge):
    bridge._scan_result = make_scan_result()

    assert bridge.edit_people("merge") == {
        "applied": False,
        "reason": "Choose a person first.",
    }


def test_an_impossible_edit_says_why_and_changes_nothing(bridge):
    bridge._scan_result = make_scan_result()
    before = len(bridge._scan_result.people)
    bridge.select_person(0, "")

    answer = bridge.edit_people("merge")

    assert answer["applied"] is False
    assert "two people" in answer["reason"]
    assert len(bridge._scan_result.people) == before


def test_the_gallery_refuses_edits_while_a_job_runs(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")
    bridge.select_person(1, "")
    bridge._worker = AliveWorker()

    answer = bridge.edit_people("merge")

    assert answer["applied"] is False


# ------------------------------------------------------- starting an export


def test_pressing_export_builds_settings_the_worker_accepts(bridge, monkeypatch):
    """The window's headline action, and it was broken from the day the web
    view replaced Tk: the bridge passed `encoder=` to a dataclass whose
    field is `video_encoder`, so every export raised TypeError before it
    started. Every test that touched start_export stopped at an earlier
    guard -- no video, no person, no footage -- so nothing ever reached the
    line that mattered.
    """
    started = []
    monkeypatch.setattr(bridge, "_start", lambda target, *args: started.append(args))
    monkeypatch.setattr(bridge, "_ensure_source_available", lambda: True)
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    answer = bridge.start_export("/tmp/reels", "out.mp4", "libx264", "Standard")

    assert answer == {"started": True}
    path, settings = started[0]
    assert path == Path("/tmp/reels/out.mp4")
    assert settings.video_encoder == "libx264"
    assert isinstance(settings.quality, int)


def test_the_chosen_quality_level_reaches_the_encoder(bridge, monkeypatch):
    """The two encoders' scales run in opposite directions, so a level that
    did not translate would silently encode at the wrong quality."""
    started = []
    monkeypatch.setattr(bridge, "_start", lambda target, *args: started.append(args))
    monkeypatch.setattr(bridge, "_ensure_source_available", lambda: True)
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    bridge.start_export("/tmp", "a.mp4", "libx264", "Maximum")
    bridge.start_export("/tmp", "b.mp4", "h264_videotoolbox", "Maximum")

    libx264_quality = started[0][1].quality
    videotoolbox_quality = started[1][1].quality
    assert libx264_quality == quality_for("libx264", "Maximum")
    assert videotoolbox_quality == quality_for("h264_videotoolbox", "Maximum")
    assert libx264_quality != videotoolbox_quality


# ------------------------------------------------------------------ naming


def test_naming_a_card_keeps_it_selected(bridge):
    """Renaming changes no membership, so being deselected by naming
    somebody would mean re-picking them to export."""
    bridge._scan_result = make_scan_result()
    bridge.select_person(1, "")

    answer = bridge.edit_people("rename", [], "Jamie")

    assert answer["applied"] is True
    assert [p.index for p in bridge._selected] == [1]
    assert answer["note"] == "#2 is now Jamie."


def test_a_name_reaches_the_page(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    answer = bridge.edit_people("rename", [], "Jamie Lee")

    assert answer["people"][0]["name"] == "Jamie Lee"
    assert answer["people"][0]["label"] == "Jamie Lee"
    assert answer["people"][1]["label"] == "Person #2"


def test_a_named_person_names_the_file(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")
    bridge.edit_people("rename", [], "Jamie Lee")

    answer = bridge.select_person(0, "")
    answer = bridge.select_person(0, "reel.mp4")

    assert answer["filename"] == "documentary-jamie-lee.mp4"


def test_an_unnamed_selection_keeps_the_compact_filename(bridge):
    bridge._scan_result = make_scan_result()

    bridge.select_person(0, "reel.mp4")
    answer = bridge.select_person(2, "reel.mp4")

    assert answer["filename"] == "documentary-person-1+3.mp4"


def test_a_mixed_selection_names_who_it_can(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")
    bridge.edit_people("rename", [], "Jamie")
    bridge.select_person(0, "")

    bridge.select_person(0, "reel.mp4")
    answer = bridge.select_person(2, "reel.mp4")

    assert answer["filename"] == "documentary-jamie+person-3.mp4"
    assert answer["name"] == "Jamie and Person #3"


def test_clearing_a_name_puts_the_number_back(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")
    bridge.edit_people("rename", [], "Jamie")

    answer = bridge.edit_people("rename", [], "  ")

    assert answer["people"][0]["name"] is None
    assert answer["people"][0]["label"] == "Person #1"
    assert answer["note"] == "Cleared the name on #1."


def test_a_name_that_could_not_be_a_filename_is_refused(bridge):
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    answer = bridge.edit_people("rename", [], "Jamie/Lee")

    assert answer["applied"] is False
    assert "cannot contain" in answer["reason"]


# ------------------------------------------------------------- the dialogs


def test_the_file_dialogs_use_the_current_pywebview_api(bridge):
    """pywebview 6 deprecated the OPEN_DIALOG and FOLDER_DIALOG integers in
    favour of the FileDialog enum, warning on every use and promising to
    remove them. The enum's members carry the same values, so this pins the
    call rather than the number it happens to equal."""
    window = FakeWindow(dialog=["/videos/episode.mp4"])
    bridge.window = window

    bridge.choose_video()
    bridge.choose_folder()

    assert window.dialogs == [webview.FileDialog.OPEN, webview.FileDialog.FOLDER]
    for asked in window.dialogs:
        assert isinstance(asked, webview.FileDialog)


def test_choosing_a_video_reports_the_path(bridge):
    bridge.window = FakeWindow(dialog=["/videos/episode.mp4"])

    assert bridge.choose_video() == {"path": "/videos/episode.mp4"}


def test_cancelling_a_dialog_chooses_nothing(bridge):
    bridge.window = FakeWindow(dialog=None)

    assert bridge.choose_video() == {"path": None}
    assert bridge.choose_folder() == {"path": None}


# ------------------------------------------- the paths only the window walks
#
# Every method below was reachable from the page and executed by no test.
# That is exactly how `start_export` came to pass `encoder=` to a dataclass
# whose field is `video_encoder` and raise TypeError on every export for
# three releases: the tests that named it all stopped at a guard above the
# line that mattered. These drive the bodies.


def test_pressing_scan_builds_settings_the_worker_accepts(bridge, tmp_path, monkeypatch):
    """The window's other headline button, and its body had never run."""
    started = []
    monkeypatch.setattr(bridge, "_start", lambda target, *args: started.append((target, args)))
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"pretend footage")

    answer = bridge.start_scan(str(video), "live", 0.25)

    assert answer == {"started": True}
    target, (path, settings) = started[0]
    assert target == bridge._scan_worker
    assert path == video
    assert settings.sample_interval == 0.25
    assert settings.mode == "live"
    # Kept, because an edit is written back to the cache entry these key.
    assert bridge._scan_settings is settings


def test_a_scan_remembers_the_mode_it_was_given(bridge, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "_start", lambda *args: None)
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")

    bridge.start_scan(str(video), "animation", 1.0)

    assert bridge._mode == "animation"


def test_a_mode_the_app_does_not_have_is_ignored(bridge, tmp_path, monkeypatch):
    """The page sends it, so it is not trusted."""
    monkeypatch.setattr(bridge, "_start", lambda *args: None)
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")

    bridge.start_scan(str(video), "interpretive-dance", 1.0)

    assert bridge._mode == web.DEFAULT_MODE


def test_the_scan_worker_reports_the_gallery(bridge, monkeypatch):
    window = FakeWindow()
    bridge.window = window
    result = make_scan_result()
    monkeypatch.setattr("app.ui.web.scan", lambda *args, **kwargs: result)

    bridge._scan_worker(Path("/videos/episode.mp4"), ScanSettings())

    assert window.emitted("onScanned") is not None
    assert bridge._scan_result is result
    assert bridge._selected == []


def test_a_cancelled_scan_says_so_rather_than_failing(bridge, monkeypatch):
    window = FakeWindow()
    bridge.window = window

    def cancelled(*args, **kwargs):
        raise web.Cancelled()

    monkeypatch.setattr("app.ui.web.scan", cancelled)

    bridge._scan_worker(Path("/videos/episode.mp4"), ScanSettings())

    assert window.emitted("onScanCancelled") is not None
    assert window.emitted("onFailed") is None


def test_a_broken_scan_reports_why(bridge, monkeypatch):
    window = FakeWindow()
    bridge.window = window

    def broken(*args, **kwargs):
        raise RuntimeError("the model would not load")

    monkeypatch.setattr("app.ui.web.scan", broken)

    bridge._scan_worker(Path("/videos/episode.mp4"), ScanSettings())

    assert window.emitted("onFailed") is not None
    assert "the model would not load" in window.calls[-1]


def test_the_export_worker_makes_the_folder_and_reports_the_file(
    bridge, tmp_path, monkeypatch
):
    window = FakeWindow()
    bridge.window = window
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")
    monkeypatch.setattr("app.ui.web.export", lambda *args, **kwargs: None)
    destination = tmp_path / "reels" / "out.mp4"

    bridge._export_worker(destination, ExportSettings())

    assert destination.parent.is_dir()
    assert window.emitted("onExported") is not None


def test_a_cancelled_export_says_so(bridge, tmp_path, monkeypatch):
    window = FakeWindow()
    bridge.window = window
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    def cancelled(*args, **kwargs):
        raise web.Cancelled()

    monkeypatch.setattr("app.ui.web.export", cancelled)

    bridge._export_worker(tmp_path / "out.mp4", ExportSettings())

    assert window.emitted("onExportCancelled") is not None
    assert window.emitted("onExported") is None


def test_a_broken_export_reports_why(bridge, tmp_path, monkeypatch):
    window = FakeWindow()
    bridge.window = window
    bridge._scan_result = make_scan_result()
    bridge.select_person(0, "")

    def broken(*args, **kwargs):
        raise RuntimeError("the encoder is not available")

    monkeypatch.setattr("app.ui.web.export", broken)

    bridge._export_worker(tmp_path / "out.mp4", ExportSettings())

    assert window.emitted("onFailed") is not None
    assert "the encoder is not available" in window.calls[-1]


def test_the_split_picker_describes_the_tracks_it_offers(bridge, monkeypatch):
    from PIL import Image as PILImage

    bridge._scan_result = make_scan_result()
    monkeypatch.setattr(
        "app.ui.web.track_previews",
        lambda result, person: [
            (0, 12.0, PILImage.new("RGB", (4, 3))),
            (3, 65.0, PILImage.new("RGB", (4, 3))),
        ],
    )

    answer = bridge.tracks_of(1)

    assert answer["index"] == 1
    assert [t["track"] for t in answer["tracks"]] == [0, 3]
    assert [t["at"] for t in answer["tracks"]] == ["0:12", "1:05"]
    assert answer["tracks"][0]["image"].startswith("data:image/jpeg;base64,")


def test_the_split_picker_offers_nothing_for_a_card_that_is_not_there(bridge):
    bridge._scan_result = make_scan_result()

    assert bridge.tracks_of(99) == {"tracks": []}


def test_the_split_picker_is_quiet_while_a_job_runs(bridge):
    bridge._scan_result = make_scan_result()
    bridge._worker = AliveWorker()

    assert bridge.tracks_of(0) == {"tracks": []}


class ClosingSource:
    """Stands in for the held footage descriptor, and says when it is let go."""

    def __init__(self):
        self.closed = 0
        self.path = Path("/videos/documentary.mp4")
        self.size = 1

    def close(self):
        self.closed += 1

    def is_available(self):
        return True


def test_closing_the_window_stops_work_and_releases_the_footage(bridge):
    """Bound to the window's closed event, so it runs on the way out and
    has to let go of the descriptor the scan is holding."""
    source = ClosingSource()
    bridge._scan_result = make_scan_result(source=source)

    assert bridge.shutdown() == {"ok": True}
    assert bridge._cancel.is_set()
    assert source.closed == 1


def test_closing_a_window_that_never_scanned_is_fine(bridge):
    assert bridge.shutdown() == {"ok": True}


def test_cancelling_asks_the_running_job_to_stop(bridge):
    """The action button becomes Cancel during a scan or an export, and
    this is all it does -- the worker notices at its next checkpoint."""
    assert bridge.cancel() == {"cancelling": True}
    assert bridge._cancel.is_set()


def test_starting_a_job_clears_a_cancellation_from_the_last_one(bridge, tmp_path):
    """Otherwise a scan cancelled at noon would stop the next one dead."""
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")
    bridge.cancel()

    bridge.start_scan(str(video), "live", 1.0)
    bridge._worker.join(timeout=5)

    assert not bridge._cancel.is_set()
