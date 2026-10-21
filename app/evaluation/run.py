"""Running the pipeline for evaluation, and the suite of videos it runs over.

A pass here is the window's scan, setting for setting (worker._scan_footage),
with two differences that matter for measuring:

- it never reads or writes kept scans. A kept scan carries the user's own
  corrections -- merges, splits, names -- and measuring those would measure
  the user, not the pipeline;
- it keeps the faces that made no card. A kept scan drops them, and they
  are where a person missing from a reel went.

The suite (evaluation/suite.json) names each video, where it is on this
machine, and its truth file. The videos themselves are not in the
repository; a video that is not here is skipped and said so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.metrics import Evaluation, evaluate
from app.evaluation.truth import GroundTruth

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "evaluation" / "suite.json"


@dataclass
class Pass:
    """One pipeline pass: its cards, in gallery order, and its leftovers."""

    cards: list[list]
    unassigned: list
    duration: float
    interval: float
    groups: list = None


def run_pass(video: Path, mode: str, interval: float) -> Pass:
    """The pipeline over one video, exactly as a window scan runs it."""
    from app.faces.grouper import auto_min_detections
    from app.main import run_identity_pipeline
    from app.ui.worker import ScanSettings
    from app.video.frames import extract_frames
    from app.video.loader import get_video_info, load_video

    settings = ScanSettings.for_mode(mode, sample_interval=interval)
    with load_video(video) as container:
        duration = get_video_info(container)["duration"]
        if duration is None and container.duration:
            duration = container.duration / 1_000_000
        min_detections = (
            max(1, settings.min_detections)
            if settings.min_detections is not None
            else auto_min_detections(duration, settings.sample_interval)
        )
        result = run_identity_pipeline(
            extract_frames(container, sample_interval=settings.sample_interval),
            confidence_threshold=settings.confidence_threshold,
            padding_ratio=settings.padding_ratio,
            similarity_threshold=settings.similarity_threshold,
            margin_threshold=settings.margin_threshold,
            consolidation_threshold=settings.consolidation_threshold,
            min_confidence=settings.min_confidence,
            min_face_size=settings.min_face_size,
            min_group_eye_span=settings.min_group_eye_span,
            forbid_cooccurring=settings.forbid_cooccurring,
            cooccurrence_similarity_ceiling=settings.cooccurrence_similarity_ceiling,
            mode=settings.mode,
            min_detections=min_detections,
        )
    groups = sorted(result.grouper.groups, key=lambda g: -g.size)
    return Pass(
        cards=[list(group.observations) for group in groups],
        unassigned=list(result.grouper.unassigned),
        duration=duration or result.last_timestamp,
        interval=interval,
        groups=groups,
    )


def _track_report(groups, truth) -> None:
    """TEMPORARY diagnostic: every card's tracks, and where each one fits best."""
    import os
    from collections import Counter

    import numpy as np

    from app.evaluation.truth import NOT_PEOPLE, Matcher

    if not (ROOT / "output" / "track-report").exists():
        return
    del os

    matcher = Matcher(truth)
    cards = sorted(groups, key=lambda g: -g.size)

    def total(observations):
        return np.stack([o.embedding for o in observations]).astype(np.float64).sum(axis=0)

    def unit(vector):
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def labels(observations):
        counts = Counter(matcher.label(o) for o in observations)
        return " ".join(
            f"{label}:{n}" for label, n in sorted(counts.items(), key=lambda x: -x[1])
        )

    card_sums = [total(card.observations) for card in cards]
    print(f"\n===== track report: {truth.video} =====")
    for index, card in enumerate(cards):
        tracks = card.tracks
        print(f"\ncard {index + 1}  {card.size} faces  {len(tracks)} tracks  [{labels(card.observations)}]")
        if len(tracks) < 2:
            continue
        for track in sorted(tracks, key=len):
            track_sum = total(track)
            rest_sum = card_sums[index] - track_sum
            rest_size = card.size - len(track)
            own_link = float(track_sum @ rest_sum) / (len(track) * rest_size)
            own_centroid = float(unit(track_sum) @ unit(rest_sum))
            rivals = []
            for other, other_card in enumerate(cards):
                if other == index:
                    continue
                link = float(track_sum @ card_sums[other]) / (len(track) * other_card.size)
                centroid = float(unit(track_sum) @ unit(card_sums[other]))
                rivals.append((centroid, link, other + 1))
            rivals.sort(reverse=True)
            best = rivals[0]
            start = min(o.source_timestamp for o in track)
            end = max(o.source_timestamp for o in track)
            print(
                f"    track {len(track):>4} faces  {start:>7.1f}-{end:>7.1f}s  "
                f"own link {own_link:+.3f} cent {own_centroid:+.3f}  |  "
                f"best card {best[2]:>2} cent {best[0]:+.3f} link {best[1]:+.3f}  "
                f"[{labels(track)}]"
            )


@dataclass(frozen=True)
class SuiteEntry:
    name: str
    video: Path
    truth: Path


def load_suite(path: Path = SUITE) -> list[SuiteEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for item in data["videos"]:
        video = Path(item["video"]).expanduser()
        if not video.is_absolute():
            video = ROOT / video
        entries.append(SuiteEntry(item["name"], video, ROOT / item["truth"]))
    return entries


def evaluate_entry(entry: SuiteEntry) -> Evaluation:
    truth = GroundTruth.load(entry.truth)
    result = run_pass(entry.video, truth.mode, truth.interval)
    _track_report(result.groups, truth)
    return evaluate(result.cards, result.unassigned, truth, result.duration, result.interval)


def why_not(entry: SuiteEntry) -> str | None:
    """Why this entry cannot be measured here, or None if it can."""
    if not entry.video.exists():
        return f"{entry.video} is not on this machine"
    if not entry.truth.exists():
        return f"no truth at {entry.truth}"
    size = GroundTruth.load(entry.truth).size
    if size and entry.video.stat().st_size != size:
        # Another encode of the same episode has other frames, so other
        # boxes: measuring it against this truth would measure the mismatch.
        return f"{entry.video.name} is not the file its truth was labelled on"
    return None


def as_record(entry: SuiteEntry, result: Evaluation) -> dict:
    """The headline numbers, for keeping beside earlier runs."""
    return {
        "name": entry.name,
        "reviewed": result.reviewed,
        "faces": result.faces,
        "coverage": round(result.coverage, 4),
        "precision": round(result.precision, 4),
        "recall": None if result.recall is None else round(result.recall, 4),
        "contamination": round(result.contamination, 4),
        "contaminated_cards": len(result.contaminated_cards),
        "cards": len(result.cards),
        "splits": result.splits,
        "people": {
            p.label: {
                "faces": p.faces,
                "completeness": round(p.completeness, 4),
                "timing": round(p.timing_overlap, 4),
                "reel_purity": round(p.reel_purity, 4),
                "reel_shows_others": round(p.reel_shows_others, 2),
            }
            for p in result.people
        },
    }
