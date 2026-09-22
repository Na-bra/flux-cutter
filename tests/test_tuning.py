"""The Advanced settings in the window: what is kept, and what a scan gets."""

import pytest

from app.ui import tuning
from app.ui.worker import ScanSettings

LIVE = ScanSettings.for_mode("live", sample_interval=0.5)


def test_nothing_changed_is_exactly_the_command_line_s_settings():
    """So the window and the command line still share every kept scan."""
    assert tuning.changed("live") == {}
    assert tuning.apply(LIVE, tuning.changed("live")) == LIVE


def test_a_change_is_kept_for_its_mode_only():
    tuning.set_value("live", "face_size", 60)

    assert tuning.changed("live") == {"face_size": 60.0}
    assert tuning.changed("animation") == {}


def test_likeness_moves_the_merge_threshold_with_it():
    """Moved alone, the looser merge would undo a stricter floor."""
    tuning.set_value("live", "likeness", 0.5)

    tuned = tuning.apply(LIVE, tuning.changed("live"))

    assert tuned.similarity_threshold == pytest.approx(0.5)
    assert tuned.consolidation_threshold == pytest.approx(
        0.5 + LIVE.consolidation_threshold - LIVE.similarity_threshold
    )


def test_screen_time_becomes_detections_at_the_sampling_interval():
    tuning.set_value("live", "screen_time", 10)

    assert tuning.apply(LIVE, tuning.changed("live")).min_detections == 20
    at_one_second = ScanSettings.for_mode("live", sample_interval=1.0)
    assert tuning.apply(at_one_second, tuning.changed("live")).min_detections == 10


def test_values_are_kept_inside_their_range_and_on_their_step():
    tuning.set_value("live", "likeness", 5.0)
    tuning.set_value("live", "face_size", 43)

    assert tuning.changed("live")["likeness"] == pytest.approx(0.55)
    assert tuning.changed("live")["face_size"] == 44


def test_setting_the_mode_s_own_value_counts_as_unchanged():
    tuning.set_value("live", "face_size", 60)
    tuning.set_value("live", "face_size", 40)

    assert tuning.changed("live") == {}


def test_none_puts_one_setting_back_and_reset_puts_them_all_back():
    tuning.set_value("live", "face_size", 60)
    tuning.set_value("live", "screen_time", 5)

    tuning.set_value("live", "face_size", None)
    assert tuning.changed("live") == {"screen_time": 5.0}

    tuning.reset("live")
    assert tuning.changed("live") == {}


def test_something_that_is_not_a_setting_is_refused():
    with pytest.raises(KeyError):
        tuning.set_value("live", "volume", 11)


def test_a_damaged_settings_file_is_ignored():
    tuning.settings_dir().mkdir(parents=True, exist_ok=True)
    (tuning.settings_dir() / "tuning.json").write_text("{not json")

    assert tuning.changed("live") == {}
    tuning.set_value("live", "face_size", 60)
    assert tuning.changed("live") == {"face_size": 60.0}


def test_the_panel_shows_the_mode_s_own_values_and_what_changed():
    tuning.set_value("animation", "likeness", 0.8)

    rows = {row["key"]: row for row in tuning.describe("animation")}

    assert rows["likeness"]["own"] == pytest.approx(0.75)
    assert rows["likeness"]["value"] == pytest.approx(0.8)
    assert rows["likeness"]["changed"] is True
    assert rows["face_size"]["value"] == 24 and rows["face_size"]["changed"] is False
    # Screen time left alone is worked out per video, so it has no one value.
    assert rows["screen_time"]["value"] is None
    assert rows["likeness"]["min"] < 0.75 < rows["likeness"]["max"]


# ------------------------------------------------------------ in the window


web = pytest.importorskip("app.ui.web")


def test_a_scan_from_the_window_uses_what_was_changed(tmp_path, monkeypatch):
    bridge = web.Bridge()
    started = []
    monkeypatch.setattr(bridge, "_start", lambda target, *args: started.append(args))
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")

    assert bridge.set_tuning("face_size", 70)["applied"] is True
    bridge.start_scan(str(video), "live", 0.5)

    (_, settings), = started
    assert settings.min_face_size == 70
    assert settings.similarity_threshold == LIVE.similarity_threshold


def test_a_folder_scan_uses_it_too(tmp_path, monkeypatch):
    bridge = web.Bridge()
    started = []
    monkeypatch.setattr(bridge, "_start", lambda target, *args: started.append(args))

    bridge.set_tuning("screen_time", 4)
    bridge.start_folder_scan(str(tmp_path), "live", 0.5)

    (_, settings), = started
    assert settings.min_detections == 8


def test_changing_mode_shows_that_mode_s_settings():
    bridge = web.Bridge()
    bridge.set_tuning("face_size", 70)

    answer = bridge.set_mode("animation")

    face = next(r for r in answer["tuning"] if r["key"] == "face_size")
    assert face["value"] == 24 and face["changed"] is False


def test_settings_cannot_change_while_a_job_runs():
    class Alive:
        @staticmethod
        def is_alive():
            return True

    bridge = web.Bridge()
    bridge._worker = Alive()

    assert bridge.set_tuning("face_size", 70)["applied"] is False
    assert bridge.reset_tuning()["applied"] is False
    assert tuning.changed("live") == {}


def test_the_window_opens_with_the_panel_s_settings():
    rows = web.Bridge().initial_state()["tuning"]
    assert [r["key"] for r in rows] == ["likeness", "screen_time", "face_size"]
