"""A scene pack: one video of someone's clearest shots, about as long as asked.

A scene pack is what an editor starts from when making a short for TikTok
or YouTube: one continuous video of one character, which they then cut to
music and put their own transitions on. So it has to be raw material that
cuts cleanly -- not a reel, which keeps everything a person is in, and not
a folder of three-second clips.

What makes it cut cleanly is that it is made of whole shots. Every piece
starts on the show's own cut into a shot and ends on its cut out of one, so
a join in the pack is a join the show already made, and an editor's cut
can land on any of them. Back-to-back shots the character holds are kept
together as one piece -- a conversation plays as the show edited it --
which also keeps the pack from strobing through a flash of every face.

Shots come from app/video/shots.py, found from the scan's own samples. A
shot qualifies when the character is in most of it; a run of qualifying
shots is a piece; pieces are taken clearest first until the pack is about
the length asked for, and played in story order. Plain cuts, no dissolves:
the transitions are the editor's to make.

Like the cut list it fills (app/video/cuts.py), this holds no footage and
opens nothing: clarity comes from the faces the scan already measured.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.faces.grouper import _observation_quality
from app.video.cuts import Cut, kept

# A shot counts as theirs when they are seen in at least this share of its
# samples. Below it the shot is mostly someone else -- the other side of the
# conversation -- which is the footage an editor has to cut around.
MIN_PRESENCE = 0.5

# A piece shorter than this is a flash, not a shot anyone can use.
MIN_PIECE_SECONDS = 1.5

# Pieces at least this long are taken before any shorter one, clearest
# first within each. Clearest-first alone favours quick close-ups: a 2:00
# pack of test_3's lead was 28 pieces, 12 of them 3s or under, which plays
# as a strobe. Taking these first made it 17 pieces, median 5.0s, none
# under 3s, for 6% of clarity (0.494 -> 0.464). Shorter pieces still fill
# a pack the longer ones cannot -- someone with little screen time.
PREFERRED_PIECE_SECONDS = 3.0

# How far from the length asked the pack may land, to end on a whole shot
# rather than mid-shot: this many seconds, or this share of the length if
# that is more. A 2:00 pack comes out between 1:54 and 2:06.
SLACK_SECONDS = 2.0
SLACK_SHARE = 0.05

# The shortest pack worth asking for. A bare "2" is two seconds, and almost
# certainly meant two minutes; refusing it says so instead of making an
# empty pack.
MIN_PACK_SECONDS = 5.0

# What the encode takes off each piece: it ends on a whole frame, and its
# sliver trimming (cutter.py) can still take a frame or two where a shot
# boundary was found late. Measured on whole-shot pieces; see
# Instructions.md 57.
EDGE_LOSS_SECONDS = 0.05


@dataclass(frozen=True)
class Seen:
    """One sampled face of the person: when, and how clearly."""

    time: float
    quality: float


@dataclass(frozen=True)
class ShotScore:
    """One shot, as far as a pack is concerned.

    `start` and `end` are the part of it known to be this shot: from its
    first sample to its last. The camera cut lies somewhere in the sampling
    interval either side, and a piece leaves that interval out rather than
    risk a frame of the neighbouring shot.
    """

    index: int
    start: float
    end: float
    presence: float
    score: float
    best: float | None

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Piece:
    """Back-to-back shots the character holds, played as one stretch."""

    video: int
    shots: tuple[ShotScore, ...]

    @property
    def start(self) -> float:
        return self.shots[0].start

    @property
    def end(self) -> float:
        return self.shots[-1].end

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def score(self) -> float:
        """The shots' scores, each weighted by how long it lasts."""
        total = sum(shot.seconds for shot in self.shots)
        if total <= 0:
            return 0.0
        return sum(shot.score * shot.seconds for shot in self.shots) / total


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


def score_shots(shots, seen: list[Seen], interval: float) -> list[ShotScore]:
    """Each shot's share of the character and how clearly it shows them.

    Args:
        shots: The video's shots (app.video.shots.Shot), in order.
        seen: The character's faces in that video.
        interval: The sampling interval the shots and faces were found at.

    A shot's samples are those from its first up to the next shot's first.
    Presence counts the moments the character is seen, not their faces --
    two of their cards in one frame are still one moment. Every shot is
    scored, even one a single sample long: it has no usable span, but it is
    still someone's shot, and whether it is theirs decides whether the
    shots either side of it are back to back (`find_pieces`).
    """
    scored = []
    for position, shot in enumerate(shots):
        following = shots[position + 1] if position + 1 < len(shots) else None
        samples_end = following.start_time if following is not None else shot.end_time
        # Its last sample: the one before the next shot's first. The video's
        # end is not a sample, so the last shot runs to it.
        known_end = following.earliest_start if following is not None else shot.end_time
        known_end = max(known_end, shot.start_time)
        faces = [face for face in seen if shot.start_time <= face.time < samples_end]
        samples = max(1, round((samples_end - shot.start_time) / interval))
        moments = len({round(face.time, 3) for face in faces})
        presence = min(1.0, moments / samples)
        if faces:
            quality = sum(face.quality for face in faces) / len(faces)
            best = max(faces, key=lambda face: face.quality).time
        else:
            quality, best = 0.0, None
        scored.append(ShotScore(shot.index, shot.start_time, known_end, presence, quality * presence, best))
    return scored


def find_pieces(
    scored: list[ShotScore],
    video: int = 0,
    within: list[tuple[float, float]] | None = None,
) -> list[Piece]:
    """Runs of back-to-back shots the character holds, as pieces.

    Args:
        scored: One video's shots, from `score_shots`.
        video: Which video they are from, for the cut list.
        within: Keep only pieces at least half inside these spans. The
            window passes the reel's plan, which a folder has already
            cleared of footage an earlier episode showed (repeats.py), so a
            recap is not packed twice.
    """
    runs: list[list[ShotScore]] = []
    for shot in scored:
        if shot.presence < MIN_PRESENCE:
            continue
        # Back to back means the very next shot. Decided by the shots' own
        # numbers, never by how close in time they are: a cutaway to
        # someone else can be shorter than any gap one could allow, and
        # would be packed inside the character's piece (found in review).
        if runs and shot.index == runs[-1][-1].index + 1:
            runs[-1].append(shot)
        else:
            runs.append([shot])

    pieces = []
    for run in runs:
        piece = Piece(video, tuple(run))
        if piece.seconds < MIN_PIECE_SECONDS:
            continue
        if within is not None and _inside(piece.start, piece.end, within) < 0.5 * piece.seconds:
            continue
        pieces.append(piece)
    return pieces


def _inside(start: float, end: float, spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, hi) - max(start, lo)) for lo, hi in spans)


def slack(seconds: float) -> float:
    """How far from `seconds` a pack may land to end on a whole shot."""
    return max(SLACK_SECONDS, SLACK_SHARE * seconds)


def choose(pieces: list[Piece], seconds: float, edge_loss: float = EDGE_LOSS_SECONDS) -> list[Cut]:
    """The clearest pieces, about `seconds` long once cut, as a cut list.

    Pieces of PREFERRED_PIECE_SECONDS or more are taken first, then
    shorter ones; within each, clearest first, while they fit. One
    too long to fit loses shots from whichever end is the weaker until it
    does, so it still starts and ends on the show's cuts; only a single
    shot longer than what is left is cut inside, around its clearest face.
    Taking stops once the pack is within the slack of the length.

    The list holds every piece in story order -- the ones not taken as
    dropped rows, so an editor can swap one in by hand -- with a shortened
    piece shown as changed from the whole run it came from.
    """
    # Never more than half the length either side, or a pack of a few
    # seconds would count as done before anything was taken.
    margin = min(slack(seconds), seconds / 2)
    ranked = sorted(
        range(len(pieces)),
        key=lambda i: (
            pieces[i].seconds < PREFERRED_PIECE_SECONDS,
            -pieces[i].score,
            pieces[i].video,
            pieces[i].start,
        ),
    )

    taken: dict[int, tuple[float, float]] = {}
    total = 0.0
    for i in ranked:
        if total >= seconds - margin:
            break
        room = seconds + margin - total
        span = _fitted(pieces[i], room + edge_loss, seconds - total + edge_loss)
        if span is None:
            continue
        taken[i] = span
        total += max(0.0, span[1] - span[0] - edge_loss)

    order = sorted(range(len(pieces)), key=lambda i: (pieces[i].video, pieces[i].start))
    return [
        Cut(
            video=pieces[i].video,
            start=taken[i][0] if i in taken else pieces[i].start,
            end=taken[i][1] if i in taken else pieces[i].end,
            planned_start=pieces[i].start,
            planned_end=pieces[i].end,
            dropped=i not in taken,
        )
        for i in order
    ]


def _fitted(piece: Piece, room: float, wanted: float) -> tuple[float, float] | None:
    """The piece, or as much of it as fits in `room`, on shot boundaries.

    A piece too long keeps the longest stretch of its whole shots that fits
    -- the clearer of two that tie -- so it still starts and ends where the
    show cuts. Only when no stretch of whole shots fits is one shot cut
    inside: the clearest that can hold what is wanted, around its best face.
    """
    if piece.seconds <= room:
        return piece.start, piece.end
    shots = piece.shots
    best_key, best_span = None, None
    for first in range(len(shots)):
        for last in range(first, len(shots)):
            span = shots[last].end - shots[first].start
            if span > room:
                break
            if span < MIN_PIECE_SECONDS:
                continue
            key = (span, Piece(piece.video, shots[first : last + 1]).score)
            if best_key is None or key > best_key:
                best_key, best_span = key, (shots[first].start, shots[last].end)
    if best_span is not None:
        return best_span

    length = min(room, max(wanted, MIN_PIECE_SECONDS))
    holding = [shot for shot in shots if shot.seconds >= length]
    if not holding:
        longest = max(shots, key=lambda shot: shot.seconds)
        if longest.seconds < MIN_PIECE_SECONDS:
            return None
        holding, length = [longest], longest.seconds
    shot = max(holding, key=lambda candidate: candidate.score)
    centre = shot.best if shot.best is not None else (shot.start + shot.end) / 2
    start = min(max(centre - length / 2, shot.start), shot.end - length)
    return start, start + length


def expected_seconds(cuts: list[Cut], edge_loss: float = EDGE_LOSS_SECONDS) -> float:
    """How long the kept pieces should come out, once cut."""
    return sum(max(0.0, cut.seconds - edge_loss) for cut in kept(cuts))


def available_seconds(pieces: list[Piece], edge_loss: float = EDGE_LOSS_SECONDS) -> float:
    """How long a pack could come out: every piece, whole, once cut."""
    return sum(max(0.0, piece.seconds - edge_loss) for piece in pieces)


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
