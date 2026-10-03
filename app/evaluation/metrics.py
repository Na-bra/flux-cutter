"""The measures, from one pipeline pass and that video's ground truth.

Counted in sampled faces (observations), which is the unit everything
downstream is built from, and in seconds where seconds are what a viewer
sees: a person's appearances, and the reel cut from their card.

**Which card is a person's.** The card holding most of their faces -- the
one someone picking them out of the gallery would choose. Their other cards
are splits; faces of theirs on someone else's card are contamination there.

**Contamination** is measured twice, because the two answer different
questions. In faces: of the faces on cards, how many belong to someone
other than the card's person. In seconds: of the reel cut from a person's
card, how much shows another named person and not them -- which is what
the viewer would actually see.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.evaluation.truth import NOT_A_FACE, NOT_PEOPLE, UNKNOWN, GroundTruth, Matcher
from app.faces.grouper import FaceIdentityGroup
from app.video.export import merge_for_export
from app.video.timeline import AppearanceInterval, build_appearance_intervals, merge_spans


@dataclass
class PersonResult:
    """How one person fared."""

    label: str
    name: str
    faces: int  # their faces the pipeline produced
    on_cards: int  # of those, on any card
    cards: int  # cards on which they are the majority
    main_card: int | None  # index of the card holding most of their faces
    on_main_card: int
    # Seconds: their true appearances, their main card's, and the overlap.
    true_seconds: float = 0.0
    card_seconds: float = 0.0
    shared_seconds: float = 0.0
    # The reel cut from their main card.
    reel_seconds: float = 0.0
    reel_shows_them: float = 0.0
    reel_shows_others: float = 0.0
    contaminated_by: dict[str, int] = field(default_factory=dict)

    @property
    def completeness(self) -> float:
        """Share of their faces on their main card."""
        return self.on_main_card / self.faces if self.faces else 0.0

    @property
    def timing_overlap(self) -> float:
        """Intersection over union of their true and card appearances."""
        union = self.true_seconds + self.card_seconds - self.shared_seconds
        return self.shared_seconds / union if union > 0 else 0.0

    @property
    def reel_purity(self) -> float:
        return self.reel_shows_them / self.reel_seconds if self.reel_seconds else 0.0


@dataclass
class CardResult:
    """One card, and who is really on it."""

    index: int
    size: int
    person: str | None  # the majority named person, if any
    counts: dict[str, int]

    @property
    def foreign(self) -> int:
        """Faces of a named person other than the card's."""
        return sum(n for label, n in self.counts.items() if label not in NOT_PEOPLE and label != self.person)

    @property
    def not_faces(self) -> int:
        return self.counts.get(NOT_A_FACE, 0)

    @property
    def unlabelled(self) -> int:
        return self.counts.get(None, 0)


@dataclass
class Evaluation:
    """Everything measured on one video."""

    video: str
    faces: int  # observations the pipeline produced
    matched: int  # of those, found in the truth
    true_faces: int  # matched and labelled a face
    not_faces: int
    checked_found: int  # faces found in the frames checked by eye
    checked_missed: int  # faces visible there and not found
    cards: list[CardResult]
    people: list[PersonResult]
    reviewed: bool

    @property
    def coverage(self) -> float:
        """Share of the pipeline's faces the truth labels."""
        return self.matched / self.faces if self.faces else 0.0

    @property
    def precision(self) -> float:
        return self.true_faces / self.matched if self.matched else 0.0

    @property
    def recall(self) -> float | None:
        seen = self.checked_found + self.checked_missed
        return self.checked_found / seen if seen else None

    @property
    def contamination(self) -> float:
        """Of the named faces on cards, the share on someone else's card."""
        named = sum(sum(n for l, n in c.counts.items() if l not in NOT_PEOPLE and l is not None) for c in self.cards)
        return sum(c.foreign for c in self.cards) / named if named else 0.0

    @property
    def contaminated_cards(self) -> list[CardResult]:
        return [card for card in self.cards if card.foreign]

    @property
    def splits(self) -> int:
        """Extra cards: one person on more than one card, counted per extra card."""
        return sum(max(0, p.cards - 1) for p in self.people)

    def summary(self, top: int = 12) -> str:
        recall = f"{self.recall:.1%}" if self.recall is not None else "not checked"
        lines = [
            f"{self.video}{'' if self.reviewed else '  (labels are a draft, not yet reviewed)'}",
            f"  truth covers {self.matched} of {self.faces} faces found ({self.coverage:.1%})",
            f"  detection: precision {self.precision:.1%} ({self.not_faces} not faces), recall {recall}"
            + (f" ({self.checked_missed} missed in {self.checked_found + self.checked_missed} checked)" if self.recall is not None else ""),
            f"  identity: contamination {self.contamination:.2%} of named faces on cards, "
            f"{len(self.contaminated_cards)} of {len(self.cards)} cards hold someone else; "
            f"{self.splits} extra cards from splits",
            "",
            f"  {'person':<16} {'faces':>6} {'cards':>5} {'main':>6} {'timing':>7} {'reel':>8} {'shows them':>10} {'others':>7}",
        ]
        for person in self.people[:top]:
            lines.append(
                f"  {person.name[:16]:<16} {person.faces:>6} {person.cards:>5} {person.completeness:>6.0%} "
                f"{person.timing_overlap:>7.0%} {person.reel_seconds:>7.0f}s {person.reel_purity:>10.0%} "
                f"{person.reel_shows_others:>6.1f}s"
            )
        worst = sorted(self.contaminated_cards, key=lambda c: -c.foreign)[:5]
        if worst:
            lines.append("")
            lines.append("  cards holding someone else:")
            for card in worst:
                others = ", ".join(f"{label} {n}" for label, n in sorted(card.counts.items(), key=lambda x: -x[1])
                                   if label not in NOT_PEOPLE and label is not None and label != card.person)
                lines.append(f"    card {card.index + 1} ({card.person}, {card.size} faces): {others}")
        return "\n".join(lines)


def _seconds(spans) -> float:
    return sum(end - start for start, end in spans)


def _intersect(a, b) -> list[tuple[float, float]]:
    """Overlap of two sorted, non-overlapping span lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract(a, b) -> list[tuple[float, float]]:
    """a less b, both sorted and non-overlapping."""
    out = []
    for start, end in a:
        cursor = start
        for b_start, b_end in b:
            if b_end <= cursor or b_start >= end:
                continue
            if b_start > cursor:
                out.append((cursor, b_start))
            cursor = max(cursor, b_end)
        if cursor < end:
            out.append((cursor, end))
    return out


def _spans(observations, duration: float, interval: float) -> list[tuple[float, float]]:
    if not observations:
        return []
    group = FaceIdentityGroup(group_id=-1, observations=list(observations))
    return [(i.start_time, i.end_time) for i in build_appearance_intervals(group, duration, interval)]


def evaluate(
    cards: list[list],
    unassigned: list,
    truth: GroundTruth,
    duration: float,
    interval: float,
) -> Evaluation:
    """Measures one pass of the pipeline against the video's truth.

    Args:
        cards: Each card's observations, in gallery order.
        unassigned: Observations that made no card.
    """
    matcher = Matcher(truth)
    card_labels = [[matcher.label(o) for o in card] for card in cards]
    loose_labels = [matcher.label(o) for o in unassigned]
    every = [label for labels in card_labels for label in labels] + loose_labels

    card_results = []
    for index, labels in enumerate(card_labels):
        counts = Counter(labels)
        named = Counter({label: n for label, n in counts.items() if label is not None and label not in NOT_PEOPLE})
        person = named.most_common(1)[0][0] if named else None
        card_results.append(CardResult(index=index, size=len(labels), person=person, counts=dict(counts)))

    # Each named person's faces, wherever they ended up.
    faces_of: dict[str, list] = {}
    for card, labels in zip(cards, card_labels):
        for observation, label in zip(card, labels):
            if label is not None and label not in NOT_PEOPLE:
                faces_of.setdefault(label, []).append(observation)
    for observation, label in zip(unassigned, loose_labels):
        if label is not None and label not in NOT_PEOPLE:
            faces_of.setdefault(label, []).append(observation)
    true_spans = {label: _spans(obs, duration, interval) for label, obs in faces_of.items()}

    people = []
    for label, observations in faces_of.items():
        per_card = [c.counts.get(label, 0) for c in card_results]
        main = max(range(len(per_card)), key=lambda i: per_card[i]) if any(per_card) else None
        result = PersonResult(
            label=label,
            name=truth.name_of(label),
            faces=len(observations),
            on_cards=sum(per_card),
            cards=sum(1 for c in card_results if c.person == label),
            main_card=main,
            on_main_card=per_card[main] if main is not None else 0,
            true_seconds=_seconds(true_spans[label]),
        )
        if main is not None:
            card_spans = _spans(cards[main], duration, interval)
            result.card_seconds = _seconds(card_spans)
            result.shared_seconds = _seconds(_intersect(true_spans[label], card_spans))
            # The reel exactly as an export of that card would cut it.
            reel = [
                (s.start_time, s.end_time)
                for s in merge_for_export([AppearanceInterval(a, b) for a, b in card_spans], duration)
            ]
            others = merge_spans([span for other, spans in true_spans.items() if other != label for span in spans])
            result.reel_seconds = _seconds(reel)
            result.reel_shows_them = _seconds(_intersect(reel, true_spans[label]))
            result.reel_shows_others = _seconds(_intersect(_subtract(reel, true_spans[label]), others))
            card = card_results[main]
            result.contaminated_by = {
                other: n for other, n in card.counts.items()
                if other is not None and other not in NOT_PEOPLE and other != label
            }
        people.append(result)
    people.sort(key=lambda p: -p.faces)

    found_at: dict[float, int] = Counter()
    every_observation = [o for card in cards for o in card] + list(unassigned)
    for observation, label in zip(every_observation, every):
        if label is not None and label != NOT_A_FACE:
            found_at[round(float(observation.source_timestamp), 3)] += 1
    checked_found = sum(found_at.get(round(frame.t, 3), 0) for frame in truth.checked)
    checked_missed = sum(len(frame.missed) for frame in truth.checked)

    matched = sum(1 for label in every if label is not None)
    return Evaluation(
        video=truth.video,
        faces=len(every),
        matched=matched,
        true_faces=sum(1 for label in every if label is not None and label != NOT_A_FACE),
        not_faces=sum(1 for label in every if label == NOT_A_FACE),
        checked_found=checked_found,
        checked_missed=checked_missed,
        cards=card_results,
        people=people,
        reviewed=truth.reviewed,
    )


__all__ = ["Evaluation", "PersonResult", "CardResult", "evaluate", "UNKNOWN"]
