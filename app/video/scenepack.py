"""A scene pack: someone's clearest scenes, up to a length the user chooses.

A reel is everything a person is in. A scene pack is the opposite question
-- "give me a minute of Ben" -- answered with the scenes that show them
best, delivered as one file per scene (what an editor works from) and one
reel of the same scenes.

It is a cut list, not a second planner. The scenes are the cuts the reel
would have had (app/video/cuts.py); a pack keeps the clearest of them and
drops the rest, so the window can show the choice in the same list, and
whatever is dropped or nudged there afterwards is what gets exported.

Like the cut list, this holds no footage and opens nothing: a scene's
clarity comes from the faces the scan already measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app.faces.grouper import _observation_quality
from app.video.cuts import Cut, kept

# Below this, what is left of the length asked for is not filled. A scene
# trimmed to a second is a flash, not a scene; the pack comes out up to
# this much short instead.
MIN_SCENE_SECONDS = 2.0

# What the encode takes off each scene, on average. The cutter leaves out
# up to 12 frames of the neighbouring shot at either edge (cutter.py,
# SLIVER_FRAMES) and ends on a whole frame, and on the test episode's
# 60-second pack that was 0.17-0.87s a scene, 0.55s on average: 13 scenes
# came out 52.9s. So each scene is counted as this much shorter than
# planned, and the pack is planned long by that much to come out the
# length asked for.
EDGE_LOSS_SECONDS = 0.55


@dataclass(frozen=True)
class Seen:
    """One sampled face of the person: when, and how clearly."""

    time: float
    quality: float


@dataclass(frozen=True)
class Clarity:
    """How well a scene shows the person, and its clearest moment."""

    score: float
    best: float | None


def faces_seen(people) -> list[Seen]:
    """Every face of these people (worker.Person cards), scored.

    Scored the way a card's picture is chosen (grouper._observation_quality):
    confidence, size, sharpness and likeness to the card's own face, so a
    face that would make a good thumbnail makes a good moment in a pack.
    """
    seen = []
    for person in people:
        group = person.group
        centroid = group.representative_embedding
        seen.extend(
            Seen(observation.source_timestamp, _observation_quality(observation, centroid))
            for observation in group.observations
        )
    return sorted(seen, key=lambda face: face.time)


def clarity(cut: Cut, seen: list[Seen], interval: float) -> Clarity:
    """How clearly `cut` shows the person.

    The mean quality of their faces in it, times how much of it they are
    seen in. Mean rather than best, because one sharp frame in a scene
    spent looking at someone else's shoulder is not a clear scene; the
    share seen, because padding and bridged gaps put stretches of other
    people into every reel (Instructions.md 31), and a scene that is mostly
    those is a poor scene of this person however good its one face is.
    """
    faces = [face for face in seen if cut.start <= face.time <= cut.end]
    if not faces or cut.seconds <= 0:
        return Clarity(0.0, None)
    presence = min(1.0, len(faces) * interval / cut.seconds)
    mean = sum(face.quality for face in faces) / len(faces)
    best = max(faces, key=lambda face: face.quality).time
    return Clarity(mean * presence, best)


def choose(
    cuts: list[Cut],
    seen: dict[int, list[Seen]],
    interval: dict[int, float],
    seconds: float,
    edge_loss: float = EDGE_LOSS_SECONDS,
) -> list[Cut]:
    """The clearest scenes, adding up to `seconds` once cut, as a cut list.

    Args:
        cuts: The reel's cut list. Cuts already dropped stay dropped.
        seen: Each video's faces of the person, by the cut's `video`.
        interval: Each video's sampling interval, by the same key.
        seconds: How long the pack should be.
        edge_loss: What cutting takes off each scene (EDGE_LOSS_SECONDS);
            each one is counted as that much shorter than it is planned.

    Scenes are taken clearest first, whole, while they fit. The first one
    that does not is trimmed to what is left, around its clearest moment,
    so the pack comes out the length asked for rather than a scene short
    of it -- unless what is left is under MIN_SCENE_SECONDS. A scene in
    which the person is never seen is never chosen. The list keeps its
    order, which is the order the footage plays in, so the reel of a pack
    is still in story order however the scenes were ranked.
    """
    scores = {
        position: clarity(cut, seen.get(cut.video, []), interval.get(cut.video, 1.0))
        for position, cut in enumerate(cuts)
        if not cut.dropped
    }
    ranked = sorted(
        (position for position, score in scores.items() if score.score > 0),
        key=lambda position: (-scores[position].score, position),
    )

    spans: dict[int, tuple[float, float]] = {}
    remaining = seconds
    for position in ranked:
        if remaining < MIN_SCENE_SECONDS:
            break
        cut = cuts[position]
        if cut.seconds - edge_loss <= remaining:
            spans[position] = (cut.start, cut.end)
            remaining -= max(0.0, cut.seconds - edge_loss)
        else:
            spans[position] = _around(cut, scores[position].best, remaining + edge_loss)
            remaining = 0.0

    return [
        replace(cut, start=spans[position][0], end=spans[position][1])
        if position in spans
        else replace(cut, dropped=True)
        for position, cut in enumerate(cuts)
    ]


def _around(cut: Cut, moment: float | None, length: float) -> tuple[float, float]:
    """`length` seconds of `cut`, centred on `moment` where the cut allows."""
    centre = moment if moment is not None else (cut.start + cut.end) / 2
    start = min(max(centre - length / 2, cut.start), cut.end - length)
    return start, start + length


def expected_seconds(cuts: list[Cut], edge_loss: float = EDGE_LOSS_SECONDS) -> float:
    """How long the kept scenes should come out, once cut."""
    return sum(max(0.0, cut.seconds - edge_loss) for cut in kept(cuts))


def available_seconds(
    cuts: list[Cut],
    seen: dict[int, list[Seen]],
    interval: dict[int, float],
    edge_loss: float = EDGE_LOSS_SECONDS,
) -> float:
    """How long a pack could come out: every kept scene they are seen in,
    once cut -- measured the way `expected_seconds` measures a pack, so the
    two can be compared and said in one breath."""
    return sum(
        max(0.0, cut.seconds - edge_loss)
        for cut in kept(cuts)
        if clarity(cut, seen.get(cut.video, []), interval.get(cut.video, 1.0)).score > 0
    )


def clip_names(
    stem: str, export_format: str, cuts: list[Cut], videos: list[str] | None = None
) -> list[str]:
    """A file name for each kept scene, in the order they play.

    Numbered so a folder sorts in story order, and stamped with where the
    scene starts, so an editor can find it in the episode: "Ben 003
    (4m07s).mp4", or with the video's name in a folder's pack.
    """
    names = []
    for number, cut in enumerate(kept(cuts), start=1):
        minutes, rest = divmod(int(cut.start), 60)
        at = f"{minutes}m{rest:02d}s"
        if videos is not None:
            at = f"{Path(videos[cut.video]).stem} {at}"
        names.append(f"{stem} {number:03d} ({at}).{export_format}")
    return names


def parse_length(text: str) -> float | None:
    """A length as typed: "90", "90s", "1:30" or "1m30s". None for empty.

    Raises:
        ValueError: If it is not a length, or not a positive one.
    """
    text = text.strip().lower().replace(" ", "")
    if not text:
        return None
    if ":" in text:
        minutes, _, rest = text.partition(":")
        value = int(minutes or 0) * 60 + float(rest or 0)
    elif "m" in text:
        minutes, _, rest = text.partition("m")
        value = int(minutes or 0) * 60 + float(rest.rstrip("s") or 0)
    else:
        value = float(text.rstrip("s"))
    if value <= 0:
        raise ValueError("a scene pack has to be longer than nothing")
    return value
