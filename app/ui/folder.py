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

import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image

from app.faces.cast import Answers, CardRef, CastCard, SamePersonQuestion, build_cast
from app.main import collect_videos
from app.modes import get_mode
from app.scans import scan_cache_dir
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
    same_frame_rate,
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


def _by_video(person: CastPerson) -> dict[int, list[Person]]:
    grouped: dict[int, list[Person]] = {}
    for video, card in person.appearances:
        grouped.setdefault(video, []).append(card)
    return dict(sorted(grouped.items()))


def plan_cast_export(
    folder: FolderScan,
    person: CastPerson,
    settings: ExportSettings | None = None,
) -> list[tuple[ScanResult, list]]:
    """Each video's segments for this person, in folder order.

    Leaving out what an earlier video in the reel has already shown -- see
    `repeated_seconds` for how much that is.
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
    plans = [(folder.videos[video], segments) for video, segments in trimmed if segments]
    return plans, removed


def export_cast(
    folder: FolderScan,
    person: CastPerson,
    output_path: Path,
    settings: ExportSettings | None = None,
    on_progress=None,
    cancel: threading.Event | None = None,
) -> tuple[CutResult, list[tuple[Path, str]]]:
    """Cuts one person out of every video they are in, into one reel.

    A video at a different frame rate from the first one contributing is
    left out and said so, rather than taking the whole reel down.

    Returns:
        The cut, and the videos left out with the reason for each.

    Raises:
        Cancelled: If `cancel` was set during the encode.
        CutterError: If nothing is left to cut, or the cut fails.
    """
    settings = settings or ExportSettings()
    plans = plan_cast_export(folder, person, settings)

    usable, left_out = [], []
    rate = None
    for result, segments in plans:
        source = result.source or result.video_path
        try:
            profile = probe_clip(source, include_audio=settings.include_audio)
        except CutterError as error:
            left_out.append((result.video_path, str(error)))
            continue
        if rate is None:
            rate = profile.frame_rate
        elif not same_frame_rate(rate, profile.frame_rate):
            left_out.append(
                (
                    result.video_path,
                    f"it is {float(profile.frame_rate):.3f}fps and the reel is "
                    f"{float(rate):.3f}fps",
                )
            )
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
