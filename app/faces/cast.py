"""The cast of a folder of videos: one entry per person, across every video.

Each video's scan already groups its own faces into cards. A folder needs
one more step -- deciding which card in episode 2 is the same person as
which card in episode 1 -- and it is taken with the rule `batch --person`
already uses to find someone in a video they were not named in, so the
window and the command line agree about who is who:

- **Clear matches link.** Two cards in different videos are the same person
  when each is the other's best match in its video, clears the mode's
  similarity floor, and is ahead of that video's runner-up by the match
  margin. Mutual, so a card cannot be claimed by two people.
- **Names link.** Cards a person gave the same name are one person whatever
  their faces score; naming is the stronger evidence.
- **Unclear matches ask.** A card with a plausible match that fails the rule
  -- too close to a runner-up, not mutual, or a second card from a video
  the person already has -- becomes a question, shown as two faces, never
  a guess. Answers are kept and override the scores.

Nothing here reads video; it works on the cards' average faces, so it is
cheap enough to redo after every answer.

Measured on the 22-minute episode split into two files: correct pairings
scored 0.92-0.97, and no other card reached 0.30 against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.faces.reference import DEFAULT_MATCH_MARGIN


@dataclass(frozen=True, order=True)
class CardRef:
    """One card: which video, and which person in that video's gallery."""

    video: int
    person: int


@dataclass(frozen=True)
class CastCard:
    """What linking needs to know about a card."""

    ref: CardRef
    embedding: np.ndarray | None
    name: str | None = None
    # Detections, for ordering the cast by how much a person is on screen.
    weight: int = 0


@dataclass
class CastMember:
    """One person across the folder."""

    cards: list[CardRef]
    name: str | None = None
    weight: int = 0

    @property
    def videos(self) -> list[int]:
        return sorted({card.video for card in self.cards})


@dataclass(frozen=True)
class SamePersonQuestion:
    """Two cards that might be one person, for someone to look at."""

    first: CardRef
    second: CardRef
    similarity: float

    @property
    def pair(self) -> frozenset[CardRef]:
        return frozenset((self.first, self.second))


@dataclass
class Answers:
    """What the person said about pairs of cards. Overrides every score."""

    same: set[frozenset[CardRef]] = field(default_factory=set)
    different: set[frozenset[CardRef]] = field(default_factory=set)

    def record(self, first: CardRef, second: CardRef, same: bool) -> None:
        pair = frozenset((first, second))
        (self.same if same else self.different).add(pair)
        (self.different if same else self.same).discard(pair)


class _Sets:
    """Union-find over cards, keeping each group's members."""

    def __init__(self, refs):
        self.parent = {ref: ref for ref in refs}
        self.members = {ref: {ref} for ref in refs}

    def find(self, ref):
        while self.parent[ref] != ref:
            self.parent[ref] = self.parent[self.parent[ref]]
            ref = self.parent[ref]
        return ref

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members.pop(rb)

    def group(self, ref):
        return self.members[self.find(ref)]


def _unit(vector) -> np.ndarray | None:
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def build_cast(
    cards: list[CastCard],
    minimum_similarity: float,
    margin: float = DEFAULT_MATCH_MARGIN,
    answers: Answers | None = None,
) -> tuple[list[CastMember], list[SamePersonQuestion]]:
    """Links cards across videos into people, and says what it could not decide.

    Args:
        cards: Every card from every video.
        minimum_similarity: The mode's grouping floor -- the number that
            already means "these faces are one person" in its embedding space.
        margin: How far ahead of a video's runner-up a match must be.
        answers: What the person has said about pairs so far.

    Returns:
        The cast, most on screen first, and the open questions, most likely
        first.
    """
    answers = answers or Answers()
    refs = [card.ref for card in cards]
    by_ref = {card.ref: card for card in cards}
    sets = _Sets(refs)

    def forbidden(a: CardRef, b: CardRef) -> bool:
        """Whether joining a's and b's groups contradicts something said."""
        group_a, group_b = sets.group(a), sets.group(b)
        if any(frozenset((x, y)) in answers.different for x in group_a for y in group_b):
            return True
        names_a = {by_ref[x].name.casefold() for x in group_a if by_ref[x].name}
        names_b = {by_ref[y].name.casefold() for y in group_b if by_ref[y].name}
        return bool(names_a and names_b and not names_a & names_b)

    # Names first: they decide what may never be joined.
    named: dict[str, CardRef] = {}
    for card in cards:
        if card.name:
            key = card.name.casefold()
            if key in named:
                sets.union(named[key], card.ref)
            else:
                named[key] = card.ref

    # Similarity between every pair of cards that have a face.
    faced = [card for card in cards if _unit(card.embedding) is not None]
    index = {card.ref: i for i, card in enumerate(faced)}
    if faced:
        matrix = np.stack([_unit(card.embedding) for card in faced])
        similarity = matrix @ matrix.T
    else:
        similarity = np.zeros((0, 0), dtype=np.float32)
    videos = sorted({card.ref.video for card in faced})
    in_video = {v: [c.ref for c in faced if c.ref.video == v] for v in videos}

    def ranked_in(ref: CardRef, video: int) -> list[tuple[float, CardRef]]:
        i = index[ref]
        return sorted(
            ((float(similarity[i, index[other]]), other) for other in in_video[video]),
            key=lambda item: item[0],
            reverse=True,
        )

    def clear_best(ref: CardRef, video: int) -> tuple[float, CardRef] | None:
        """ref's match in `video`, if it is clear by the batch rule."""
        ranked = ranked_in(ref, video)
        if not ranked:
            return None
        score, best = ranked[0]
        if score < minimum_similarity:
            return None
        if len(ranked) > 1 and score - ranked[1][0] < margin:
            return None
        return score, best

    # Clear, mutual matches, strongest first so a weak link cannot claim a
    # card before a strong one does.
    links = []
    for card in faced:
        for video in videos:
            if video == card.ref.video:
                continue
            found = clear_best(card.ref, video)
            if found is None:
                continue
            score, other = found
            back = clear_best(other, card.ref.video)
            if back is not None and back[1] == card.ref and card.ref < other:
                links.append((score, card.ref, other))
    links.sort(reverse=True)

    for score, a, b in links:
        if sets.find(a) == sets.find(b) or forbidden(a, b):
            continue
        group_a, group_b = sets.group(a), sets.group(b)
        # Two cards from one video in one person is a split card, which is
        # worth a question but not an assumption.
        if {x.video for x in group_a} & {y.video for y in group_b}:
            continue
        sets.union(a, b)

    # Answers last, because they override. Applied before the links, a
    # "yes" joining a split card to a person made that person look like it
    # already held a card from the split card's video, and the clear link
    # the answer depended on was then refused as a second one.
    for pair in answers.same:
        a, b = tuple(pair)
        if a in by_ref and b in by_ref and not forbidden(a, b):
            sets.union(a, b)

    # Everything plausible that was not linked is a question.
    questions: dict[frozenset, SamePersonQuestion] = {}
    for card in faced:
        for video in videos:
            # Only across videos. Within one, grouping has already kept two
            # cards apart -- often because both faces share a frame -- and a
            # split card still surfaces through its match in another video.
            if video == card.ref.video:
                continue
            ranked = ranked_in(card.ref, video)
            for score, other in ranked[:2]:
                if other == card.ref or score < minimum_similarity:
                    continue
                if sets.find(card.ref) == sets.find(other):
                    continue
                pair = frozenset((card.ref, other))
                if pair in answers.different or forbidden(card.ref, other):
                    continue
                groups = frozenset((sets.find(card.ref), sets.find(other)))
                existing = questions.get(groups)
                if existing is None or score > existing.similarity:
                    first, second = sorted((card.ref, other))
                    questions[groups] = SamePersonQuestion(first, second, score)

    members = []
    for root in {sets.find(ref) for ref in refs}:
        group = sorted(sets.group(root))
        name = next((by_ref[ref].name for ref in group if by_ref[ref].name), None)
        members.append(
            CastMember(
                cards=group,
                name=name,
                weight=sum(by_ref[ref].weight for ref in group),
            )
        )
    members.sort(key=lambda member: (-member.weight, member.cards[0]))
    return members, sorted(questions.values(), key=lambda q: -q.similarity)
