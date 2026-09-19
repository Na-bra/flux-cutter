"""The people you have named, remembered across every video you open.

A name used to belong to one video's kept scan. Naming Jamie Lee in season
1 did nothing for season 2, and `batch --person` could only find a name in
the folder it was given. This keeps every named card's face, by name, so
any later scan can say "this looks like Jamie Lee" and a batch run can find
them by name anywhere.

It is an index, not a second copy of the naming. Whenever a kept scan is
saved (app/scans.py), its named cards replace whatever that scan said
before -- so naming, renaming and clearing a name all reach the library
the moment they reach the scan, and nothing has to be kept in step by hand.
Faces outlive the scan they came from: a scan pruned to save space still
taught the library what the person looks like.

One face per person per video, not one average for everyone: a character
across a season changes lighting, costume and age, and matching against
each face they have been seen with keeps the variety an average would blur.

Matching is the rule used everywhere else -- the mode's similarity floor,
and a margin over the runner-up -- in both directions: the person's best
card must be clear of the scan's other cards, and that card's best person
clear of the library's other people. A suggestion is only ever that; it
names nothing until someone says yes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.faces.reference import DEFAULT_MATCH_MARGIN

LIBRARY_FORMAT = 1


def library_dir() -> Path:
    """Beside the model and scan caches, but not inside the scans, so
    clearing kept scans does not forget who anybody is.

    FLUXCUTTER_PEOPLE_DIR overrides it, which is what the tests use.
    """
    override = os.environ.get("FLUXCUTTER_PEOPLE_DIR")
    if override:
        return Path(override).expanduser()
    from app.models import cache_dir

    return cache_dir().parent / "people"


def _path() -> Path:
    return library_dir() / "people.json"


@dataclass(frozen=True)
class KnownPerson:
    """Someone named in at least one video, with each face they were named on."""

    name: str
    space: str
    faces: np.ndarray  # (videos, dimensions), unit vectors
    videos: int


@dataclass(frozen=True)
class Suggestion:
    """A card that looks like somebody already named elsewhere."""

    card: int
    name: str
    similarity: float


def _unit(vector) -> np.ndarray | None:
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def _space_of(group) -> str | None:
    return next(
        (o.embedding_space for o in group.observations if o.embedding_space),
        None,
    )


def _ensure() -> None:
    """Fills the library from the kept scans the first time it is used.

    People named before the library existed are in those scans already;
    without this the first save would create an empty library and they
    would never be found. Once only -- the file existing is the record.
    """
    if _path().exists():
        return
    from app import scans

    people: dict = {}
    _write(people)
    for entry in scans.entries():
        kept = scans.load(entry.path.stem)
        if kept is not None and any(group.name for group in kept.groups):
            remember(entry.path.stem, kept.groups)


def _read() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != LIBRARY_FORMAT:
        return {}
    people = data.get("people")
    return people if isinstance(people, dict) else {}


def _write(people: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": LIBRARY_FORMAT, "people": people}), encoding="utf-8"
    )
    os.replace(temporary, path)


def remember(scan_key: str, groups) -> None:
    """Records a kept scan's named cards, replacing what it said before.

    Called whenever a scan is saved. Best effort: a library that cannot be
    written only means a suggestion missed, never a scan lost.
    """
    try:
        _ensure()
        people = _read()
        for entry in people.values():
            entry.get("faces", {}).pop(scan_key, None)

        named: dict[str, list] = {}
        for group in groups:
            if not group.name:
                continue
            face = _unit(group.representative_embedding)
            space = _space_of(group)
            if face is None or space is None:
                continue
            named.setdefault(f"{space}\n{group.name.casefold()}", []).append(
                (group.name, space, face)
            )
        for slot, faces in named.items():
            name, space, _ = faces[-1]
            average = _unit(np.mean([f for _, _, f in faces], axis=0))
            entry = people.setdefault(slot, {"name": name, "space": space, "faces": {}})
            entry["name"] = name
            entry["faces"][scan_key] = [round(float(x), 6) for x in average]

        people = {slot: entry for slot, entry in people.items() if entry.get("faces")}
        _write(people)
    except OSError:
        pass


def known(space: str | None = None) -> list[KnownPerson]:
    """Everyone named so far, in one embedding space or all of them."""
    try:
        _ensure()
    except OSError:
        return []
    result = []
    for entry in _read().values():
        if space is not None and entry.get("space") != space:
            continue
        faces = [np.asarray(f, dtype=np.float32) for f in entry.get("faces", {}).values()]
        if not faces:
            continue
        result.append(
            KnownPerson(
                name=entry["name"],
                space=entry["space"],
                faces=np.stack(faces),
                videos=len(faces),
            )
        )
    return sorted(result, key=lambda p: p.name.casefold())


def forget(name: str) -> int:
    """Removes a name and every face saved under it. Returns how many went.

    Only from the library: a card still carrying the name in a kept scan
    keeps it, and naming someone again teaches the library afresh.
    """
    try:
        _ensure()
        people = _read()
        wanted = name.casefold()
        remaining = {
            slot: entry
            for slot, entry in people.items()
            if entry.get("name", "").casefold() != wanted
        }
        if len(remaining) != len(people):
            _write(remaining)
        return len(people) - len(remaining)
    except OSError:
        return 0


def find(name: str, space: str | None = None) -> KnownPerson | None:
    """Someone by name, as they were saved."""
    wanted = name.casefold()
    return next((p for p in known(space) if p.name.casefold() == wanted), None)


def suggest(
    groups,
    minimum_similarity: float,
    margin: float = DEFAULT_MATCH_MARGIN,
    people: list[KnownPerson] | None = None,
) -> list[Suggestion]:
    """Unnamed cards in one scan that look like somebody named before.

    Args:
        groups: The scan's cards, in gallery order.
        minimum_similarity: The mode's floor.
        people: The library to match against; read from disk when omitted.
    """
    cards = []
    for position, group in enumerate(groups):
        face = _unit(group.representative_embedding)
        if face is not None:
            cards.append((position, face, _space_of(group), group.name))
    if not cards:
        return []
    spaces = {space for _, _, space, _ in cards if space}
    if people is None:
        people = [p for p in known() if p.space in spaces]
    # A name already on a card in this scan is not suggested again.
    taken = {name.casefold() for _, _, _, name in cards if name}
    people = [p for p in people if p.name.casefold() not in taken]
    candidates = [(position, face) for position, face, _, name in cards if not name]
    if not people or not candidates:
        return []

    # scores[card, person]: the best of that person's faces against the card.
    faces = np.stack([face for _, face in candidates])
    scores = np.stack([(faces @ p.faces.T).max(axis=1) for p in people], axis=1)

    def clear(values: np.ndarray, best: int) -> bool:
        if values[best] < minimum_similarity:
            return False
        others = np.delete(values, best)
        return not len(others) or values[best] - others.max() >= margin

    found = []
    for person_index, person in enumerate(people):
        card_row = int(np.argmax(scores[:, person_index]))
        if not clear(scores[:, person_index], card_row):
            continue
        if int(np.argmax(scores[card_row])) != person_index or not clear(scores[card_row], person_index):
            continue
        found.append(
            Suggestion(
                card=candidates[card_row][0],
                name=person.name,
                similarity=float(scores[card_row, person_index]),
            )
        )
    return sorted(found, key=lambda s: s.card)

