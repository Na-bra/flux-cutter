"""A folder of videos in the window: scan them all, show one cast, cut one reel.

The single-video window asks "who is in this video?". This asks the same of
a season, and answers with one card per person across every episode rather
than a gallery per file. Everything below reuses what one video already
does -- each episode is an ordinary scan, kept and reused like any other,
and a person's reel is each episode's plan joined by `cut_clips` -- so the
only new idea is the cast itself (app/faces/cast.py).

Like app/ui/worker.py this imports no toolkit, so all of it is tested
without a window.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image

from app.faces.cast import Answers, CardRef, CastCard, SamePersonQuestion, build_cast
from app.main import collect_videos
from app.modes import get_mode
from app.scans import cache_key, scan_cache_dir
from app.faces.edits import EditError
from app.ui.worker import (
    Cancelled,
    ExportSettings,
    Person,
    ScanResult,
    ScanSettings,
    apply_edit,
    plan_export,
    preview_frames,
    scan,
)
from app.video.cutter import (
    Clip,
    CutResult,
    CutterError,
    cut_clips,
    probe_clip,
)
from app.video.loader import VideoLoadError
from app.video.repeats import Repeat, find_repeats, fingerprint_kept, without_repeats


@dataclass
class FolderScan:
    """Every video in a folder that could be scanned, and why the rest could not."""

    videos: list[ScanResult]
    settings: ScanSettings
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    # (later video, earlier video) -> what the later one repeats of the
    # earlier, found once when the folder is scanned (app/video/repeats.py).
    repeats: dict[tuple[int, int], list[Repeat]] = field(default_factory=dict)

    def close(self) -> None:
        for result in self.videos:
            result.close()


@dataclass(frozen=True)
class CastPerson:
    """One person across the folder, in the form the window draws."""

    index: int
    name: str | None
    # (video, card) for every card that is this person, in video order.
    appearances: list[tuple[int, Person]]
    detection_count: int

    @property
    def label(self) -> str:
        return self.name or f"Person #{self.index + 1}"

    @property
    def videos(self) -> list[int]:
        return sorted({video for video, _ in self.appearances})


def scan_folder(
    paths: list[Path],
    settings: ScanSettings,
    on_video=None,
    on_progress=None,
    cancel: threading.Event | None = None,
    on_download=None,
    recursive: bool = False,
    on_status=None,
) -> FolderScan:
    """Scans every video in a folder, reusing any scan already kept.

    A video that cannot be read is recorded and passed over; one unreadable
    file does not cost the rest of the season.

    Args:
        paths: Folders, files, or a mix.
        on_video: Called as (index, total, path) as each video starts.
        on_progress: Called as (fraction of the whole folder, timestamp).
        on_status: Called with a line of text while repeats are looked for.

    Raises:
        Cancelled: If `cancel` was set. Nothing scanned so far is kept open.
    """
    videos = collect_videos([Path(p) for p in paths], recursive=recursive)
    results: list[ScanResult] = []
    skipped: list[tuple[Path, str]] = []

    for position, path in enumerate(videos):
        if on_video is not None:
            on_video(position, len(videos), path)

        def progress(fraction: float, timestamp: float, _position=position) -> None:
            if on_progress is not None:
                on_progress((_position + fraction) / len(videos), timestamp)

        try:
            results.append(
                scan(
                    path,
                    settings=settings,
                    on_progress=progress,
                    cancel=cancel,
                    on_download=on_download,
                )
            )
        except Cancelled:
            for result in results:
                result.close()
            raise
        except (VideoLoadError, CutterError, OSError) as error:
            skipped.append((path, str(error)))

    return FolderScan(
        videos=results,
        settings=settings,
        skipped=skipped,
        repeats=find_folder_repeats(results, on_status=on_status, cancel=cancel),
    )


def fingerprint_dir() -> Path:
    return scan_cache_dir() / "fingerprints"


def find_folder_repeats(
    results: list[ScanResult], on_status=None, cancel: threading.Event | None = None
) -> dict[tuple[int, int], list[Repeat]]:
    """What each video repeats of every earlier one: recaps, openings.

    Fingerprinting reads each video once more, about 13s for 11 minutes of
    720p, and keeps the result, so a folder opened again pays nothing. A
    video that cannot be fingerprinted is simply never taken for a repeat --
    at worst its footage appears twice, as it did before this existed.
    """
    if len(results) < 2:
        return {}
    prints = {}
    for position, result in enumerate(results):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if on_status is not None:
            on_status(
                f"Checking {result.video_path.name} for footage other videos repeat "
                f"({position + 1} of {len(results)})…"
            )
        try:
            prints[position] = fingerprint_kept(result.video_path, fingerprint_dir())
        except Exception:
            continue
    found = {}
    for later in prints:
        for earlier in prints:
            if earlier < later:
                repeats = find_repeats(prints[later], prints[earlier])
                if repeats:
                    found[(later, earlier)] = repeats
    return found


def cast_of(
    folder: FolderScan, answers: Answers | None = None
) -> tuple[list[CastPerson], list[SamePersonQuestion]]:
    """The folder's people, and what could not be decided about them."""
    by_ref: dict[CardRef, Person] = {}
    cards = []
    for video, result in enumerate(folder.videos):
        for person in result.people:
            ref = CardRef(video, person.index)
            by_ref[ref] = person
            cards.append(
                CastCard(
                    ref,
                    person.group.representative_embedding,
                    person.name,
                    person.detection_count,
                )
            )

    members, questions = build_cast(
        cards,
        get_mode(folder.settings.mode).grouping.similarity_threshold,
        answers=answers,
    )
    cast = [
        CastPerson(
            index=position,
            name=member.name,
            appearances=[(ref.video, by_ref[ref]) for ref in member.cards],
            detection_count=member.weight,
        )
        for position, member in enumerate(members)
    ]
    return cast, questions


# -------------------------------------------------- remembered answers

# Beside the kept scans, because an answer is about cards in them.
ANSWERS_FILE = "cast-answers.json"


def card_key(folder: FolderScan, video: int, person_index: int) -> str | None:
    """A name for one card that is the same next time the folder is opened.

    Its position in the gallery is not: that moves with every correction.
    The kept scan's key says which video under which settings, and the
    card's first sighting and number of detections say which card in it.
    A card that is merged or split afterwards gets a different key, and
    answers about the card it used to be no longer apply to it -- which is
    right, because it is no longer that card.
    """
    result = folder.videos[video]
    person = next((p for p in result.people if p.index == person_index), None)
    if person is None:
        return None
    try:
        scan = cache_key(result.video_path, **dataclasses.asdict(folder.settings))
    except OSError:
        return None
    timestamps = [o.source_timestamp for o in person.group.observations]
    first = min(timestamps) if timestamps else person.first_seen
    return f"{scan}:{first:.3f}:{person.detection_count}"


def _answers_path() -> Path:
    return scan_cache_dir() / ANSWERS_FILE


def _stored() -> dict[str, bool]:
    try:
        data = json.loads(_answers_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    pairs = data.get("pairs") if isinstance(data, dict) else None
    return {k: bool(v) for k, v in pairs.items()} if isinstance(pairs, dict) else {}


def _pair(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def load_answers(folder: FolderScan) -> Answers:
    """Every answer given before about cards in this folder."""
    answers = Answers()
    stored = _stored()
    if not stored:
        return answers
    keys = {}
    for video, result in enumerate(folder.videos):
        for person in result.people:
            key = card_key(folder, video, person.index)
            if key is not None:
                keys[key] = CardRef(video, person.index)
    for pair, same in stored.items():
        first, _, second = pair.partition("|")
        if first in keys and second in keys:
            answers.record(keys[first], keys[second], same=same)
    return answers


def remember_answer(folder: FolderScan, first: CardRef, second: CardRef, same: bool) -> bool:
    """Keeps one answer for next time. Best effort, like a kept edit.

    Returns whether it was written: an answer the user can see must not fail
    because the cache could not be, it would only have to be given again.
    """
    a = card_key(folder, first.video, first.person)
    b = card_key(folder, second.video, second.person)
    if a is None or b is None:
        return False
    stored = _stored()
    stored[_pair(a, b)] = bool(same)
    try:
        path = _answers_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "pairs": stored}), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        return False
    return True


# ------------------------------------------------------------ corrections


def cards_of(person: CastPerson) -> list[CardRef]:
    return [CardRef(video, card.index) for video, card in person.appearances]


def _main_card(person: CastPerson) -> CardRef:
    video, card = max(person.appearances, key=lambda a: a[1].detection_count)
    return CardRef(video, card.index)


def join_people(folder: FolderScan, answers: Answers, people: list[CastPerson]) -> None:
    """Says the chosen people are one person: the cast kept them apart wrongly.

    Recorded as answers, one per person joined, and kept, so the join
    survives reopening the folder the way a single answer does.
    """
    if len(people) < 2:
        raise EditError("Choose at least two people to merge.")
    names = {p.name.casefold() for p in people if p.name}
    if len(names) > 1:
        raise EditError(
            "They have different names, and cards with different names are "
            "never one person. Clear or change a name first."
        )
    # A merge is the newest thing said about these people, so any earlier
    # "different" between their cards gives way to it. Left in place, one
    # old answer vetoed the join and the merge did nothing, silently --
    # found by merging two people split apart in an earlier session.
    groups = [cards_of(person) for person in people]
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            for a in first:
                for b in second:
                    if frozenset((a, b)) in answers.different:
                        answers.record(a, b, same=True)
                        remember_answer(folder, a, b, same=True)
    anchor = _main_card(people[0])
    for other in people[1:]:
        ref = _main_card(other)
        answers.record(anchor, ref, same=True)
        remember_answer(folder, anchor, ref, same=True)


def detach_cards(
    folder: FolderScan, answers: Answers, person: CastPerson, detached: list[CardRef]
) -> FolderScan:
    """Says some of a person's cards are somebody else.

    Every detached card is recorded as a different person from every card
    kept, so neither the scores nor an old answer can put them back. A
    shared name would -- cards with one name are one person -- so a
    detached card's copy of the person's name is cleared in its video's
    kept scan.
    """
    everyone = cards_of(person)
    detached = [ref for ref in detached if ref in everyone]
    kept = [ref for ref in everyone if ref not in detached]
    if not detached or not kept:
        raise EditError("Pick some of their faces, but not all of them.")

    for gone in detached:
        for stays in kept:
            answers.record(gone, stays, same=False)
            remember_answer(folder, gone, stays, same=False)

    videos = list(folder.videos)
    for ref in detached:
        result = videos[ref.video]
        card = next(p for p in result.people if p.index == ref.person)
        if card.name:
            videos[ref.video] = apply_edit(result, folder.settings, "rename", [card.index], name="")
    return replace(folder, videos=videos)


def discard_people(folder: FolderScan, people: list[CastPerson]) -> tuple[FolderScan, list[str]]:
    """Removes cards that are not people, from every video's kept scan.

    Returns the folder and the videos where nothing could be removed -- a
    scan is never emptied of everyone, the same rule as in one video.

    Every video discarded from renumbers its gallery, so answers held by
    gallery position must be reloaded afterwards (`load_answers`).
    """
    grouped: dict[int, list[int]] = {}
    for person in people:
        for video, card in person.appearances:
            grouped.setdefault(video, []).append(card.index)

    videos = list(folder.videos)
    refused = []
    for video, indexes in sorted(grouped.items()):
        try:
            videos[video] = apply_edit(
                videos[video], folder.settings, "discard", sorted(set(indexes))
            )
        except EditError:
            refused.append(videos[video].video_path.name)
    return replace(folder, videos=videos), refused


def _by_video(person: CastPerson) -> dict[int, list[Person]]:
    grouped: dict[int, list[Person]] = {}
    for video, card in person.appearances:
        grouped.setdefault(video, []).append(card)
    return dict(sorted(grouped.items()))


def together(people: list[CastPerson]) -> CastPerson:
    """Several people as one, for a reel of every scene any of them is in."""
    if len(people) == 1:
        return people[0]
    return CastPerson(
        index=people[0].index,
        name=" and ".join(p.label for p in people),
        appearances=[a for p in people for a in p.appearances],
        detection_count=sum(p.detection_count for p in people),
    )


def plan_cast_export(
    folder: FolderScan,
    person: CastPerson,
    settings: ExportSettings | None = None,
) -> list[tuple[ScanResult, list]]:
    """Each video's segments for this person, in folder order.

    Leaving out what an earlier video in the reel has already shown -- see
    `repeated_seconds` for how much that is.
    """
    plans, _ = _planned(folder, person, settings)
    return [(folder.videos[video], segments) for video, segments in plans]


def plan_cast_by_video(
    folder: FolderScan,
    person: CastPerson,
    settings: ExportSettings | None = None,
) -> list[tuple[int, list]]:
    """The same plan, by folder position rather than by scan.

    A cut list holds a number, not a ScanResult: the window sends one back
    with every edit, and a ScanResult is neither hashable nor cheap to
    compare (its people carry embeddings). See app/video/cuts.py.
    """
    return _planned(folder, person, settings)[0]


def repeated_seconds(
    folder: FolderScan, person: CastPerson, settings: ExportSettings | None = None
) -> float:
    """How much of this person's reel was left out as already shown."""
    return sum(_planned(folder, person, settings)[1].values())


def _planned(folder, person, settings):
    settings = settings or ExportSettings()
    per_video = []
    for video, cards in _by_video(person).items():
        result = folder.videos[video]
        _, segments = plan_export(
            cards,
            video_duration=result.video_duration,
            sample_interval=result.sample_interval,
            settings=settings,
        )
        per_video.append((video, segments))
    trimmed, removed = without_repeats(per_video, folder.repeats)
    return [(video, segments) for video, segments in trimmed if segments], removed


def export_cast(
    folder: FolderScan,
    person: CastPerson,
    output_path: Path,
    settings: ExportSettings | None = None,
    on_progress=None,
    cancel: threading.Event | None = None,
    plans: list[tuple[int, list]] | None = None,
) -> tuple[CutResult, list[tuple[Path, str]]]:
    """Cuts one person out of every video they are in, into one reel.

    A video that cannot be read is left out and said so, rather than taking
    the whole reel down. One at another frame rate is converted by the cut.

    Args:
        plans: Cut these (video position, segments) instead of planning
            for `person` again -- how an edited cut list reaches the
            encoder. See app/video/cuts.py.

    Returns:
        The cut, and the videos left out with the reason for each.

    Raises:
        Cancelled: If `cancel` was set during the encode.
        CutterError: If nothing is left to cut, or the cut fails.
    """
    settings = settings or ExportSettings()
    plans = (
        [(folder.videos[video], segments) for video, segments in plans]
        if plans is not None
        else plan_cast_export(folder, person, settings)
    )

    usable, left_out = [], []
    for result, segments in plans:
        source = result.source or result.video_path
        try:
            probe_clip(source, include_audio=settings.include_audio)
        except CutterError as error:
            left_out.append((result.video_path, str(error)))
            continue
        usable.append(Clip(source, segments))

    if not usable:
        raise CutterError(f"{person.label} has nothing to cut in these videos.")

    def report(index: int, total: int, _segment) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if on_progress is not None:
            on_progress((index + 1) / total, index + 1, total)

    cut = cut_clips(
        usable,
        output_path,
        video_encoder=settings.video_encoder,
        audio_encoder=settings.audio_encoder,
        quality=settings.quality,
        include_audio=settings.include_audio,
        on_segment=report,
    )
    return cut, left_out


def name_cast_person(folder: FolderScan, person: CastPerson, name: str) -> FolderScan:
    """Names every card that is this person, in every video's kept scan.

    Written to each scan the same way naming one card in one video is, so
    the name is there next time any of these videos is opened, and
    `batch --person` finds it. It also holds the cast together: cards with
    one name are one person whatever their faces score.

    Raises:
        EditError: If the name cannot be used.
    """
    videos = list(folder.videos)
    for video, cards in _by_video(person).items():
        result = videos[video]
        for card in cards:
            result = apply_edit(result, folder.settings, "rename", [card.index], name=name)
        videos[video] = result
    return replace(folder, videos=videos)


def cast_preview_frames(
    folder: FolderScan, person: CastPerson, limit: int = 6
) -> list[tuple[str, float, Image.Image]]:
    """Frames from the reel, spread across the videos it draws on.

    Each video gets a share, so a reel that is mostly one episode still
    shows that the others are in it.
    """
    grouped = _by_video(person)
    if not grouped:
        return []
    share = max(1, limit // len(grouped))
    frames = []
    for video, cards in grouped.items():
        result = folder.videos[video]
        for timestamp, image in preview_frames(result, cards, limit=share):
            frames.append((result.video_path.name, timestamp, image))
    return frames[:limit]
