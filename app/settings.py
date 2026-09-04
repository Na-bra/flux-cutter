"""Remembering what the user chose, between runs.

One small JSON file next to the model cache. Not a database, not a config
framework: the whole contents are a mode name and a couple of per-mode
overrides, and anything heavier would be more machinery than the data.

The shape keeps the two modes' settings apart, which is the point rather
than tidiness -- an animation threshold and a live-action threshold with the
same name mean different things, and a flat file would eventually let one be
written where the other was meant:

    {
      "mode": "live",
      "live":      {"detector_confidence": 0.6},
      "animation": {"detector_confidence": 0.3}
    }

Reads never raise. A settings file that is missing, empty, unreadable or
corrupt yields the defaults, because failing to start over a preferences
file would be a worse bug than forgetting a preference.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.models import cache_dir
from app.modes import mode_ids

SETTINGS_VERSION = 1


def settings_path() -> Path:
    """Where the settings file lives.

    Beside the downloaded models rather than inside the app: a frozen .app
    may sit on a read-only volume, and this is user data either way.
    FLUXCUTTER_SETTINGS overrides it, which is what tests use.
    """
    override = os.environ.get("FLUXCUTTER_SETTINGS")
    if override:
        return Path(override).expanduser()
    return cache_dir().parent / "settings.json"


@dataclass
class Settings:
    """The user's remembered choices.

    The selected mode is deliberately not one of them. Live action is the
    default every time FluxCutter starts, and animation is chosen per run.
    Remembering it meant a mode picked once in the window silently became
    the default for every later command-line run too, with no flag in sight
    to say so -- a live-action job quietly embedded through the anime model.
    Per-mode thresholds are still remembered, because those belong to the
    mode rather than deciding which mode you are in.
    """

    per_mode: dict[str, dict] = None

    def __post_init__(self):
        if self.per_mode is None:
            self.per_mode = {}

    def for_mode(self, mode_id: str) -> dict:
        """Overrides recorded for one mode, empty when none were."""
        return dict(self.per_mode.get(mode_id, {}))

    def set_for_mode(self, mode_id: str, values: dict) -> None:
        self.per_mode[mode_id] = dict(values)

    def to_dict(self) -> dict:
        return {
            "version": SETTINGS_VERSION,
            **{mode: self.per_mode.get(mode, {}) for mode in mode_ids()},
        }


def load() -> Settings:
    """Reads the settings file, falling back to defaults on any problem."""
    path = settings_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return Settings()

    if not isinstance(raw, dict):
        return Settings()

    # A "mode" key written by an older version is read and discarded: the
    # mode is no longer restored, so an animation session cannot outlive
    # itself.
    per_mode = {}
    for mode_id in mode_ids():
        values = raw.get(mode_id)
        if isinstance(values, dict):
            per_mode[mode_id] = values

    return Settings(per_mode=per_mode)


def save(settings: Settings) -> Path:
    """Writes the settings file, atomically.

    Atomic because the alternative is a truncated JSON file if the app is
    quit mid-write, which would silently reset every preference on the next
    launch -- the exact failure this module exists to avoid.
    """
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(settings.to_dict(), indent=2) + "\n")
    temporary.replace(path)
    return path
