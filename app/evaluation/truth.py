"""Ground truth: every face a scan produced on one video, and who it is.

Kept as JSON beside the code (evaluation/truth/), never with the video,
which is not ours to redistribute. A face is identified the way the scan
produced it -- the sampled frame's timestamp and the detector's box -- so a
later run of the same pipeline finds its faces again exactly, and a changed
pipeline finds most of them again by overlap (`match`) and reports how many
it could not (`coverage`), which says when the truth needs topping up.

Labels are person ids shared across videos -- "henry" is the same person in
every Henry Danger file, which is what cross-episode matching is measured
by -- plus two that are not people:

- NOT_A_FACE: the detector fired on a poster, a pattern, a hand;
- UNKNOWN: a face, but not one that can be told apart -- too small, turned
  away, blurred. It counts for detection and is left out of identity.

Recall needs the faces the pipeline did *not* find, which no scan holds, so
some sampled frames are checked by eye and the faces in them it missed are
recorded (`CheckedFrame`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRUTH_FORMAT = 1
NOT_A_FACE = "not-a-face"
UNKNOWN = "unknown"
NOT_PEOPLE = (NOT_A_FACE, UNKNOWN)

# Two observations are one face when they are in the same sampled frame and
# their boxes overlap this much. The same pipeline reproduces boxes exactly;
# this is for one that has changed a little.
MATCH_SECONDS = 0.02
MATCH_OVERLAP = 0.5


@dataclass(frozen=True)
class TruthFace:
    """One face a scan produced, and who it is."""

    t: float
    box: tuple[int, int, int, int]
    label: str


@dataclass(frozen=True)
class CheckedFrame:
    """A sampled frame looked over by eye for faces the scan missed.

    `missed` holds a label for each face visible in it that the pipeline did
    not find. The ones it did find are already among the truth's faces.
    """

    t: float
    missed: tuple[str, ...] = ()


@dataclass
class GroundTruth:
    """Everything known to be true about one video's faces."""

    video: str
    size: int
    duration: float
    mode: str
    interval: float
    people: dict[str, str] = field(default_factory=dict)
    faces: list[TruthFace] = field(default_factory=list)
    checked: list[CheckedFrame] = field(default_factory=list)
    # Who labelled it, and whether a person has been over it: a draft is
    # evidence, not proof, and the report says which it is resting on.
    labelled_by: str = "draft"
    reviewed: bool = False
    notes: str = ""

    def name_of(self, label: str) -> str:
        return self.people.get(label, label)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["format"] = TRUTH_FORMAT
        faces = [[round(f.t, 4), list(f.box), f.label] for f in self.faces]
        checked = [[round(c.t, 4), list(c.missed)] for c in self.checked]
        data["faces"] = data["checked"] = []
        # One face to a line: thousands of them, reviewed and corrected as diffs.
        text = json.dumps(data, indent=1, ensure_ascii=False)
        for key, rows in (("faces", faces), ("checked", checked)):
            lines = ",\n".join("  " + json.dumps(row, ensure_ascii=False) for row in rows)
            text = text.replace(f'"{key}": []', f'"{key}": [\n{lines}\n ]' if rows else f'"{key}": []', 1)
        path.write_text(text + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "GroundTruth":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.pop("format", None) != TRUTH_FORMAT:
            raise ValueError(f"{path} is not ground truth this version can read")
        data["faces"] = [TruthFace(t, tuple(box), label) for t, box, label in data["faces"]]
        data["checked"] = [CheckedFrame(t, tuple(missed)) for t, missed in data["checked"]]
        return cls(**data)


def overlap(a, b) -> float:
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inner = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inner
    return inner / union if union > 0 else 0.0


def box_of(observation) -> tuple[int, int, int, int]:
    box = observation.detection.box
    return (int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max))


class Matcher:
    """Finds the truth face for an observation, by frame and box."""

    def __init__(self, truth: GroundTruth):
        self._by_time: dict[int, list[TruthFace]] = {}
        for face in truth.faces:
            self._by_time.setdefault(self._bucket(face.t), []).append(face)

    @staticmethod
    def _bucket(t: float) -> int:
        return int(round(t / MATCH_SECONDS))

    def label(self, observation) -> str | None:
        """The observation's truth label, or None if the truth has no such face."""
        t, box = float(observation.source_timestamp), box_of(observation)
        best, best_overlap = None, MATCH_OVERLAP
        for bucket in (self._bucket(t) - 1, self._bucket(t), self._bucket(t) + 1):
            for face in self._by_time.get(bucket, ()):
                if abs(face.t - t) > MATCH_SECONDS:
                    continue
                if face.box == box:
                    return face.label
                score = overlap(face.box, box)
                if score >= best_overlap:
                    best, best_overlap = face.label, score
        return best
