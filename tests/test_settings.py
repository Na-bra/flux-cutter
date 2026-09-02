"""Tests for remembering the user's choices.

The bar is low and specific: the choice survives a restart, and a damaged
file never stops the app starting.
"""

import json

import pytest

from app import settings as user_settings
from app.modes import ANIMATION, DEFAULT_MODE, LIVE


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXCUTTER_SETTINGS", str(tmp_path / "settings.json"))


def test_a_fresh_install_is_live_action():
    assert user_settings.load().mode == DEFAULT_MODE == LIVE


def test_the_chosen_mode_survives_a_restart():
    remembered = user_settings.load()
    remembered.mode = ANIMATION
    user_settings.save(remembered)

    assert user_settings.load().mode == ANIMATION


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

    assert user_settings.load().mode == DEFAULT_MODE


def test_a_missing_file_falls_back_to_defaults():
    assert not user_settings.settings_path().exists()
    assert user_settings.load().mode == DEFAULT_MODE


def test_an_unknown_mode_name_falls_back():
    """A hand-edited file, or one written by a later version."""
    user_settings.settings_path().parent.mkdir(parents=True, exist_ok=True)
    user_settings.settings_path().write_text(json.dumps({"mode": "claymation"}))

    assert user_settings.load().mode == DEFAULT_MODE


def test_the_file_is_readable_by_a_person():
    remembered = user_settings.load()
    remembered.mode = ANIMATION
    path = user_settings.save(remembered)

    written = json.loads(path.read_text())
    assert written["mode"] == ANIMATION
    assert LIVE in written and ANIMATION in written


def test_saving_leaves_no_partial_file_behind():
    user_settings.save(user_settings.load())
    leftovers = list(user_settings.settings_path().parent.glob("*.part"))
    assert leftovers == []
