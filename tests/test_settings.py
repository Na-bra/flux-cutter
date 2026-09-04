"""Tests for remembering the user's choices.

The bar is low and specific: per-mode thresholds survive a restart, a
damaged file never stops the app starting, and the selected mode is *not*
remembered -- live action is the default every time.
"""

import json

import pytest

from app import settings as user_settings
from app.modes import ANIMATION, DEFAULT_MODE, LIVE


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXCUTTER_SETTINGS", str(tmp_path / "settings.json"))


def test_live_action_is_the_default():
    assert DEFAULT_MODE == LIVE


def test_the_mode_is_not_remembered():
    """Animation is opted into per session, never inherited from the last one.

    It used to be restored here, which meant a mode picked once in the
    window silently became the default for every later command-line run,
    with no flag in sight to say so.
    """
    assert not hasattr(user_settings.load(), "mode")


def test_a_file_that_remembers_a_mode_no_longer_strands_the_user():
    """Written by a version that persisted it. Read, discarded, thresholds kept."""
    user_settings.settings_path().parent.mkdir(parents=True, exist_ok=True)
    user_settings.settings_path().write_text(
        json.dumps({"version": 1, "mode": ANIMATION, ANIMATION: {"similarity_threshold": 0.8}})
    )

    reloaded = user_settings.load()

    assert not hasattr(reloaded, "mode")
    assert reloaded.for_mode(ANIMATION)["similarity_threshold"] == 0.8
    assert "mode" not in json.loads(user_settings.save(reloaded).read_text())


def test_per_mode_values_are_kept_apart():
    """An animation threshold must never be read as a live-action one."""
    remembered = user_settings.load()
    remembered.set_for_mode(LIVE, {"detector_confidence": 0.6})
    remembered.set_for_mode(ANIMATION, {"detector_confidence": 0.3})
    user_settings.save(remembered)

    reloaded = user_settings.load()
    assert reloaded.for_mode(LIVE)["detector_confidence"] == 0.6
    assert reloaded.for_mode(ANIMATION)["detector_confidence"] == 0.3


def test_a_corrupt_file_falls_back_to_defaults():
    """Losing a preference is a nuisance; failing to launch is a bug."""
    user_settings.settings_path().parent.mkdir(parents=True, exist_ok=True)
    user_settings.settings_path().write_text("{not json at all")

    assert user_settings.load().per_mode == {}


def test_a_missing_file_falls_back_to_defaults():
    assert not user_settings.settings_path().exists()
    assert user_settings.load().per_mode == {}


def test_the_file_is_readable_by_a_person():
    remembered = user_settings.load()
    remembered.set_for_mode(ANIMATION, {"similarity_threshold": 0.75})
    path = user_settings.save(remembered)

    written = json.loads(path.read_text())
    assert LIVE in written and ANIMATION in written
    assert written[ANIMATION]["similarity_threshold"] == 0.75


def test_saving_leaves_no_partial_file_behind():
    user_settings.save(user_settings.load())
    leftovers = list(user_settings.settings_path().parent.glob("*.part"))
    assert leftovers == []
