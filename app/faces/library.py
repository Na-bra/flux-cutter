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

import base64
import io
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


# A picture per person per video, for the window's People panel: small,
# because it sits in a JSON file beside everyone else's and is drawn at
# the size of a fingernail, and JPEG because a face crop is a photograph.
PICTURE_SIZE = 72
PICTURE_QUALITY = 82


@dataclass(frozen=True)
class KnownPerson:
    """Someone named in at least one video, with each face they were named on."""

    name: str
    space: str
    faces: np.ndarray  # (videos, dimensions), unit vectors
    videos: int
    # (video file name, JPEG as base64) for each video that has a picture,
    # in the order they were named. Empty for faces saved before pictures
    # were kept, until that video is next saved.
    pictures: tuple[tuple[str, str], ...] = ()
    # Every video named in, by file name where it is known.
    video_names: tuple[str, ...] = ()


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


def _picture(group) -> str | None:
    """The card's own face, small, as base64 JPEG. None when there is none."""
    representative = getattr(group, "representative_observation", None)
    crop = getattr(representative, "face_crop", None)
    if crop is None:
        return None
    try:
        from PIL import Image

        image = Image.fromarray(np.asarray(crop, dtype=np.uint8)).convert("RGB")
        image.thumbnail((PICTURE_SIZE, PICTURE_SIZE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=PICTURE_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except (ValueError, OSError, TypeError):
        return None


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


def remember(scan_key: str, groups, video: str = "") -> None:
    """Records a kept scan's named cards, replacing what it said before.

    Called whenever a scan is saved. Best effort: a library that cannot be
    written only means a suggestion missed, never a scan lost.

    Args:
        video: The video's file name, for the People panel to show beside
            the face. A save that does not know it keeps the one an
            earlier save of the same scan recorded.
    """
    try:
        _ensure()
        people = _read()
        known_as = video
        for entry in people.values():
            entry.get("faces", {}).pop(scan_key, None)
            entry.get("pictures", {}).pop(scan_key, None)
            known_as = known_as or entry.get("videos", {}).get(scan_key, "")
            entry.get("videos", {}).pop(scan_key, None)

        named: dict[str, list] = {}
        for group in groups:
            if not group.name:
                continue
            face = _unit(group.representative_embedding)
            space = _space_of(group)
            if face is None or space is None:
                continue
            named.setdefault(f"{space}\n{group.name.casefold()}", []).append(
                (group.name, space, face, group)
            )
        for slot, faces in named.items():
            name, space, _, _ = faces[-1]
            average = _unit(np.mean([f for _, _, f, _ in faces], axis=0))
            entry = people.setdefault(slot, {"name": name, "space": space, "faces": {}})
            entry["name"] = name
            entry["faces"][scan_key] = [round(float(x), 6) for x in average]
            # The biggest card of that name in this video is the face most
            # people would recognise them by.
            largest = max((g for _, _, _, g in faces), key=lambda g: len(g.observations))
            picture = _picture(largest)
            if picture is not None:
                entry.setdefault("pictures", {})[scan_key] = picture
            if known_as:
                entry.setdefault("videos", {})[scan_key] = known_as

        people = _without_empty(people)
        _write(people)
    except OSError:
        pass


def _without_empty(people: dict) -> dict:
    """Drops people with no face left. Their "not them" answers go too,
    because a name that is nobody's cannot be suggested to anybody."""
    return {slot: entry for slot, entry in people.items() if entry.get("faces")}


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
        videos = entry.get("videos", {})
        pictures = entry.get("pictures", {})
        result.append(
            KnownPerson(
                name=entry["name"],
                space=entry["space"],
                faces=np.stack(faces),
                videos=len(faces),
                pictures=tuple(
                    (videos.get(key, ""), picture)
                    for key, picture in pictures.items()
                    if key in entry.get("faces", {})
                ),
                video_names=tuple(videos[key] for key in entry["faces"] if videos.get(key)),
            )
        )
    return sorted(result, key=lambda p: p.name.casefold())


def forget(name: str, from_scans: bool = False) -> int:
    """Removes a name and every face saved under it. Returns how many went.

    By default only from the library: a card still carrying the name in a
    kept scan keeps it, and the next time that scan is saved the library
    learns the name again. That is what `people forget` on the command
    line has always done.

    `from_scans` makes it stick, which is what the window wants: the name
    comes off every card in every kept scan it was saved from, so nothing
    teaches it back. The cards stay, unnamed.
    """
    try:
        _ensure()
        # Counted first: unnaming the cards re-indexes their scans, which
        # already takes the person out of the library.
        found = sum(1 for entry in _read().values() if _is(entry, name))
        if from_scans:
            _rename_in_scans(name, None)
        people = _read()
        remaining = {slot: entry for slot, entry in people.items() if not _is(entry, name)}
        if len(remaining) != len(people):
            _write(remaining)
        return found
    except OSError:
        return 0


def rename(old: str, new: str) -> int:
    """Calls someone by another name, everywhere they were named.

    If the new name is already somebody's, the two become one: their faces
    are pooled, and every later scan offers the one name. That is the fix
    for one face saved under two names -- "Coach" and "Bald Man" -- which
    until now could only be undone by forgetting one of them.

    The names on the cards are rewritten too, in every kept scan the person
    was named in. The library is an index of those scans (see `remember`),
    so renaming only here would last until one of them was next saved and
    taught it the old name back. Faces whose scan has since been pruned are
    moved by hand, since there is no card left to rename.

    Returns:
        How many videos the person was named in.

    Raises:
        app.faces.edits.EditError: If the new name cannot be used.
    """
    from app.faces.edits import EditError, clean_name

    cleaned = clean_name(new)
    if cleaned is None:
        raise EditError("Give them a name, or use Forget to remove it.")
    _ensure()
    before = [entry for entry in _read().values() if _is(entry, old)]
    if not before:
        return 0
    answered = {key for entry in before for key in entry.get("declined", [])}
    videos = {key for entry in before for key in entry.get("faces", {})}

    _rename_in_scans(old, cleaned)

    # What is left under the old name came from scans that are no longer
    # kept; the saves above re-indexed everything else.
    people = _read()
    for slot in [slot for slot, entry in people.items() if _is(entry, old)]:
        entry = people.pop(slot)
        target = people.setdefault(
            f"{entry['space']}\n{cleaned.casefold()}",
            {"name": cleaned, "space": entry["space"], "faces": {}},
        )
        for field in ("faces", "pictures", "videos"):
            for key, value in entry.get(field, {}).items():
                target.setdefault(field, {}).setdefault(key, value)
    # "Not them" was said about the person, whatever they are called now.
    for entry in people.values():
        if _is(entry, cleaned):
            entry["name"] = cleaned
            entry["declined"] = sorted(set(entry.get("declined", [])) | answered)
    _write(_without_empty(people))
    return len(videos)


def _is(entry: dict, name: str) -> bool:
    return entry.get("name", "").casefold() == name.casefold()


def _rename_in_scans(old: str, new: str | None) -> None:
    """Renames (or, with None, unnames) every card called `old` in the
    kept scans the library says it was named in. Saving each one
    re-indexes it here through `remember`."""
    from dataclasses import replace

    from app import scans

    keys = {
        key
        for entry in _read().values()
        if _is(entry, old)
        for key in entry.get("faces", {})
    }
    for key in sorted(keys):
        kept = scans.load(key)
        if kept is None:
            continue
        changed = False
        for group in kept.groups:
            if group.name and group.name.casefold() == old.casefold():
                group.name = new
                changed = True
        if changed:
            scans.save(key, replace(kept, groups=kept.groups))


def decline(name: str, card: str) -> None:
    """Remembers that one card is not this person, across sessions.

    `card` is whatever the caller uses to tell a card apart from one run
    of the app to the next. Kept with the person, so it follows them
    through a rename or a merge and goes when they are forgotten.
    """
    try:
        _ensure()
        people = _read()
        changed = False
        for entry in people.values():
            if _is(entry, name) and card not in entry.setdefault("declined", []):
                entry["declined"].append(card)
                changed = True
        if changed:
            _write(people)
    except OSError:
        pass


def declined() -> dict[str, set[str]]:
    """Every "not them", as lowercased name -> the cards it was said of."""
    try:
        _ensure()
    except OSError:
        return {}
    answers: dict[str, set[str]] = {}
    for entry in _read().values():
        if entry.get("declined"):
            answers.setdefault(entry["name"].casefold(), set()).update(entry["declined"])
    return answers


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

