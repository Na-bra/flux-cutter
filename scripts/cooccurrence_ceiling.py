"""Where a shared frame stops being evidence of two people, measured.

    .venv/bin/python scripts/cooccurrence_ceiling.py animation animation2

The co-occurrence rule (grouper.DEFAULT_FORBID_COOCCURRING) reads two faces
in one sampled frame as two different people, unless they are so alike that
a split screen, a mirror or a photograph is the better explanation. Where
"so alike" sits is a point on an embedding model's similarity scale, so it
has to be measured once per mode -- which is what this prints.

For each named video of the suite it runs one pipeline pass, pairs up every
face that shares a sampled frame, and splits the pairs by what the truth file
says they are: two different characters (the evidence the rule exists to
use), or the same character twice (what the ceiling exists to spare). Then it
sweeps candidate ceilings over both.

It writes nothing and reads no kept scan; `run_pass` is the evaluation pass,
not a scan.

Written for Instructions.md 53 and **not run there**: the nightly session is
sandboxed to `python -m app ...` and `python -m pytest ...`, so §53's numbers
come from `python -m app evaluate` with one setting moved at a time instead.
This is the direct measurement of the same claim, for a machine that can run
it. Nothing depends on it.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from app.evaluation.run import load_suite, run_pass
from app.evaluation.truth import NOT_PEOPLE, GroundTruth, Matcher
from app.faces.grouper import cosine_similarity

CEILINGS = (0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.01)


def spread(values: list[float]) -> str:
    if not values:
        return "none"
    a = np.asarray(values, dtype=float)
    return (
        f"n={len(a)} min {a.min():.3f} p5 {np.percentile(a, 5):.3f} "
        f"median {np.median(a):.3f} p95 {np.percentile(a, 95):.3f} "
        f"p99 {np.percentile(a, 99):.3f} max {a.max():.3f}"
    )


def measure(name: str) -> None:
    entry = {e.name: e for e in load_suite()}[name]
    truth = GroundTruth.load(entry.truth)
    result = run_pass(entry.video, truth.mode, truth.interval)
    matcher = Matcher(truth)

    by_frame: dict[int, list] = {}
    for observation in [o for card in result.cards for o in card] + list(result.unassigned):
        if observation.frame_index is not None:
            by_frame.setdefault(observation.frame_index, []).append(observation)

    different: list[float] = []
    same: list[float] = []
    unlabelled: list[float] = []
    for sharing in by_frame.values():
        for position, first in enumerate(sharing):
            for second in sharing[position + 1 :]:
                similarity = cosine_similarity(first.embedding, second.embedding)
                a, b = matcher.label(first), matcher.label(second)
                if a is None or b is None or a in NOT_PEOPLE or b in NOT_PEOPLE:
                    unlabelled.append(similarity)
                elif a == b:
                    same.append(similarity)
                else:
                    different.append(similarity)

    print(f"\n=== {name} ({truth.mode}) ===")
    print(f"sampled frames holding two or more faces: {sum(1 for s in by_frame.values() if len(s) > 1)}")
    print(f"  two different characters:  {spread(different)}")
    print(f"  the same character twice:  {spread(same)}")
    print(f"  one of them not a person:  {spread(unlabelled)}")

    print("\n  ceiling   different pairs kept as evidence   same-character pairs misread")
    for ceiling in CEILINGS:
        kept = sum(1 for s in different if s < ceiling)
        misread = sum(1 for s in same if s < ceiling)
        print(
            f"  {ceiling:>7.2f}   {kept:>5} of {len(different):<5} "
            f"({kept / len(different) if different else 0:>6.1%})          "
            f"{misread:>5} of {len(same):<5} ({misread / len(same) if same else 0:>6.1%})"
        )

    pairs: Counter = Counter()
    for sharing in by_frame.values():
        labels = sorted(
            {
                label
                for label in (matcher.label(o) for o in sharing)
                if label is not None and label not in NOT_PEOPLE
            }
        )
        for position, a in enumerate(labels):
            for b in labels[position + 1 :]:
                pairs[(a, b)] += 1
    print("\n  character pairs sharing a frame, by frames shared:")
    for (a, b), count in pairs.most_common(20):
        print(f"    {a} + {b}: {count}")


if __name__ == "__main__":
    for video in sys.argv[1:]:
        measure(video)
