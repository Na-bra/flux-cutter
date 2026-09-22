"""The three detection settings the window lets people change, and remembers.

The window offered a mode and a sampling interval; everything else was a
command-line flag. These are the three that change most what a scan finds,
each described by what it does rather than what it is called in the code.

Only what someone has changed is stored, per mode, beside the other per-user
data -- a setting left alone follows the mode's own value, including if that
value is revised in a later release. Nothing changed means exactly the
settings the command line uses, so the two still share every kept scan.

A module like this was removed in 1.9.0 (`app/settings.py`) because nothing
used it. This one exists because something does.

What each does, measured on the 22-minute test episode at a 1s interval
(38 people with every setting at live action's own):

    how alike       0.25 -> 38 people, lead's card 545 faces; 0.55 -> 40, 491
    screen time     1s -> 118 people; 3s -> 48; 10s -> 35; 30s -> 21
    smallest face   24px -> 41 people; 80px -> 18, lead's card 538 -> 442

Screen time is the strong one, and it only ever removes minor people; the
main cards are identical at every value. "How alike" moves faces between
cards more than it changes how many there are. The descriptions below say
those things rather than what the settings are called in the code.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

from app.modes import get_mode

TUNING_FORMAT = 1


@dataclass(frozen=True)
class Setting:
    """One control: how it is shown, and the range it may take."""

    key: str
    label: str
    explains: str
    lower: str
    higher: str
    unit: str
    step: float


SETTINGS = (
    Setting(
        key="likeness",
        label="How alike two faces must be",
        explains="to be counted as the same person.",
        lower="Lower puts more of each person's faces on their card, and risks two people sharing one.",
        higher="Higher keeps faces apart, so a person's card can come apart into several.",
        unit="",
        step=0.01,
    ),
    Setting(
        key="screen_time",
        label="Least screen time worth a card",
        explains="People seen for less are left out of the gallery.",
        lower="Lower shows people who are barely in it, and many more stray faces.",
        higher="Higher shows only the people who are in it most. The main cards do not change.",
        unit="s",
        step=0.5,
    ),
    Setting(
        key="face_size",
        label="Smallest face to look at",
        explains="in pixels across.",
        lower="Lower counts faces further from the camera, so more people and more of each.",
        higher="Higher counts only close faces: fewer people, and each loses their distant shots.",
        unit="px",
        step=2,
    ),
)
BY_KEY = {setting.key: setting for setting in SETTINGS}

# How far from the mode's own value each may go. Wide enough to matter,
# narrow enough that nothing reachable here is nonsense.
LIKENESS_REACH = 0.2
SCREEN_TIME_RANGE = (0.5, 60.0)
FACE_SIZE_RANGE = (12, 160)


def settings_dir() -> Path:
    """FLUXCUTTER_SETTINGS_DIR overrides it, which is what the tests use."""
    override = os.environ.get("FLUXCUTTER_SETTINGS_DIR")
    if override:
        return Path(override).expanduser()
    from app.models import cache_dir

    return cache_dir().parent


def _path() -> Path:
    return settings_dir() / "tuning.json"


def _read() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != TUNING_FORMAT:
        return {}
    modes = data.get("modes")
    return modes if isinstance(modes, dict) else {}


def _write(modes: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": TUNING_FORMAT, "modes": modes}), encoding="utf-8")
    os.replace(temporary, path)


def _bounds(mode: str, key: str) -> tuple[float, float]:
    if key == "likeness":
        own = get_mode(mode).grouping.similarity_threshold
        return max(0.05, own - LIKENESS_REACH), min(0.98, own + LIKENESS_REACH)
    if key == "screen_time":
        return SCREEN_TIME_RANGE
    return FACE_SIZE_RANGE


def _own(mode: str, key: str) -> float | None:
    """The mode's own value. None for screen time, which it works out per video."""
    spec = get_mode(mode)
    if key == "likeness":
        return spec.grouping.similarity_threshold
    if key == "face_size":
        return spec.detection.min_face_size
    return None


def changed(mode: str) -> dict[str, float]:
    """What someone has changed for this mode, and nothing else."""
    stored = _read().get(mode, {})
    return {key: float(value) for key, value in stored.items() if key in BY_KEY}


def set_value(mode: str, key: str, value) -> dict[str, float]:
    """Changes one setting, or puts it back to the mode's own with None.

    Values are kept inside the setting's range and on its step, so nothing
    the page sends can store something the scan cannot use.
    """
    if key not in BY_KEY:
        raise KeyError(key)
    get_mode(mode)  # raises on a mode that does not exist
    modes = _read()
    current = dict(modes.get(mode, {}))
    if value is None:
        current.pop(key, None)
    else:
        low, high = _bounds(mode, key)
        step = BY_KEY[key].step
        value = min(high, max(low, float(value)))
        value = round(round(value / step) * step, 4)
        if _own(mode, key) is not None and abs(value - _own(mode, key)) < step / 2:
            current.pop(key, None)
        else:
            current[key] = value
    if current:
        modes[mode] = current
    else:
        modes.pop(mode, None)
    _write(modes)
    return changed(mode)


def reset(mode: str) -> None:
    modes = _read()
    if modes.pop(mode, None) is not None:
        _write(modes)


def apply(settings, overrides: dict[str, float]):
    """A ScanSettings with someone's changes applied.

    "How alike" is two numbers in the pipeline: the floor that joins faces
    into cards, and a looser one that later merges cards whose faces are
    close. Moving the first alone would let the second undo it, so both
    move together and keep the gap the mode was tuned with.
    """
    if not overrides:
        return settings
    changes = {}
    if "likeness" in overrides:
        gap = settings.consolidation_threshold - settings.similarity_threshold
        changes["similarity_threshold"] = overrides["likeness"]
        changes["consolidation_threshold"] = round(overrides["likeness"] + gap, 4)
    if "face_size" in overrides:
        changes["min_face_size"] = int(overrides["face_size"])
    if "screen_time" in overrides:
        changes["min_detections"] = max(
            1, math.ceil(overrides["screen_time"] / settings.sample_interval - 1e-9)
        )
    return replace(settings, **changes)


def describe(mode: str) -> list[dict]:
    """Everything the page needs to draw the panel for one mode."""
    overrides = changed(mode)
    rows = []
    for setting in SETTINGS:
        low, high = _bounds(mode, setting.key)
        own = _own(mode, setting.key)
        rows.append(
            {
                "key": setting.key,
                "label": setting.label,
                "explains": setting.explains,
                "lower": setting.lower,
                "higher": setting.higher,
                "unit": setting.unit,
                "step": setting.step,
                "min": low,
                "max": high,
                # None for screen time left to the mode: it is worked out per
                # video, from its length, so there is no one number to show.
                "own": own,
                "value": overrides.get(setting.key, own),
                "changed": setting.key in overrides,
            }
        )
    return rows
