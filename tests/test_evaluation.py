"""The evaluation measures, on cases small enough to know the answer to."""

import numpy as np
import pytest

from app.evaluation.metrics import evaluate
from app.evaluation.review import apply_corrections, runs_of
from app.evaluation.truth import NOT_A_FACE, UNKNOWN, CheckedFrame, GroundTruth, Matcher, TruthFace
from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceObservation


def face(t, x=0):
    return FaceObservation(
        embedding=np.ones(4, np.float32) / 2,
        detection=FaceDetection(box=BoundingBox(x, 0, x + 50, 50), confidence=0.9),
        face_crop=None,
        source_timestamp=t,
    )


def truth_for(*runs, checked=()):
    """runs: (label, start, count, x) -- `count` faces 0.5s apart."""
    truth = GroundTruth(video="v.mp4", size=1, duration=60.0, mode="live", interval=0.5,
                        people={"henry": "Henry", "ray": "Ray"}, checked=list(checked))
    observations = {}
    for label, start, count, x in runs:
        for k in range(count):
            t = start + 0.5 * k
            truth.faces.append(TruthFace(t, (x, 0, x + 50, 50), label))
            observations.setdefault(label, []).append(face(t, x))
    return truth, observations


def test_two_people_on_their_own_cards_is_clean():
    truth, seen = truth_for(("henry", 0, 20, 0), ("ray", 20, 20, 100))

    result = evaluate([seen["henry"], seen["ray"]], [], truth, 60.0, 0.5)

    assert result.coverage == 1.0 and result.precision == 1.0
    assert result.contamination == 0.0 and result.splits == 0
    henry = next(p for p in result.people if p.label == "henry")
    assert henry.completeness == 1.0
    assert henry.timing_overlap == pytest.approx(1.0)
    assert henry.reel_shows_others == 0.0
    assert henry.reel_purity > 0.8  # the reel pads its cuts a little


def test_someone_else_on_a_card_is_contamination_in_faces_and_in_the_reel():
    """The failure that matters most: Ray's faces on Henry's card put Ray in
    Henry's reel."""
    truth, seen = truth_for(("henry", 0, 20, 0), ("ray", 30, 4, 100), ("ray", 40, 20, 100))
    henry_card = seen["henry"] + seen["ray"][:4]
    ray_card = seen["ray"][4:]

    result = evaluate([henry_card, ray_card], [], truth, 60.0, 0.5)

    assert result.contamination == pytest.approx(4 / 44)
    assert [c.index for c in result.contaminated_cards] == [0]
    henry = next(p for p in result.people if p.label == "henry")
    assert henry.contaminated_by == {"ray": 4}
    assert henry.reel_shows_others > 1.5  # Ray's two seconds, padded


def test_one_person_on_two_cards_is_a_split():
    truth, seen = truth_for(("henry", 0, 30, 0))

    result = evaluate([seen["henry"][:18], seen["henry"][18:]], [], truth, 60.0, 0.5)

    assert result.splits == 1
    assert result.contamination == 0.0
    henry = result.people[0]
    assert henry.cards == 2
    assert henry.completeness == pytest.approx(18 / 30)


def test_faces_that_made_no_card_count_against_completeness_only():
    truth, seen = truth_for(("henry", 0, 20, 0))

    result = evaluate([seen["henry"][:15]], seen["henry"][15:], truth, 60.0, 0.5)

    henry = result.people[0]
    assert henry.faces == 20 and henry.on_cards == 15 and henry.completeness == 0.75


def test_not_faces_lower_precision_and_unknown_faces_do_not():
    truth, seen = truth_for(("henry", 0, 8, 0), (NOT_A_FACE, 10, 2, 300), (UNKNOWN, 12, 2, 400))
    card = seen["henry"] + seen[NOT_A_FACE] + seen[UNKNOWN]

    result = evaluate([card], [], truth, 60.0, 0.5)

    assert result.precision == pytest.approx(10 / 12)
    assert result.contamination == 0.0
    assert result.cards[0].person == "henry"


def test_recall_counts_the_faces_missed_in_checked_frames():
    truth, seen = truth_for(("henry", 0, 4, 0), ("ray", 0, 4, 100),
                            checked=[CheckedFrame(0.0, ("ray",)), CheckedFrame(1.0, ())])

    result = evaluate([seen["henry"], seen["ray"]], [], truth, 60.0, 0.5)

    # At 0.0: two found, one missed. At 1.0: two found.
    assert (result.checked_found, result.checked_missed) == (4, 1)
    assert result.recall == pytest.approx(0.8)


def test_faces_the_truth_does_not_know_lower_coverage_not_scores():
    truth, seen = truth_for(("henry", 0, 10, 0))

    result = evaluate([seen["henry"] + [face(30.0, 500)]], [], truth, 60.0, 0.5)

    assert result.coverage == pytest.approx(10 / 11)
    assert result.precision == 1.0
    assert result.cards[0].unlabelled == 1


def test_a_box_that_moved_slightly_still_matches():
    """A changed pipeline reproduces most boxes nearly, not exactly."""
    truth, _ = truth_for(("henry", 0, 1, 0))
    nudged = face(0.004, 3)

    assert Matcher(truth).label(nudged) == "henry"
    assert Matcher(truth).label(face(0.5, 0)) is None


def test_ground_truth_survives_a_round_trip(tmp_path):
    truth, _ = truth_for(("henry", 0, 3, 0), checked=[CheckedFrame(1.0, ("ray", UNKNOWN))])
    truth.notes = "drafted from a scan"

    again = GroundTruth.load(truth.save(tmp_path / "v.json"))

    assert again == truth


def test_the_summary_names_what_went_wrong():
    truth, seen = truth_for(("henry", 0, 20, 0), ("ray", 30, 4, 100), ("ray", 40, 20, 100))
    result = evaluate([seen["henry"] + seen["ray"][:4], seen["ray"][4:]], [], truth, 60.0, 0.5)

    text = result.summary()

    assert "draft" in text
    assert "card 1 (henry, 24 faces): ray 4" in text


def test_a_run_is_one_persons_faces_in_consecutive_frames():
    truth, _ = truth_for(("henry", 0, 4, 0), ("ray", 0, 2, 100), ("henry", 10, 2, 0))

    runs = runs_of(truth)

    assert [(r.label, len(r.faces)) for r in runs] == [("henry", 4), ("ray", 2), ("henry", 2)]


def test_corrections_relabel_faces_name_people_and_mark_the_truth_reviewed(tmp_path):
    truth, _ = truth_for(("henry", 0, 4, 0))
    path = truth.save(tmp_path / "v.json")
    moved = [[f.t, list(f.box), "jasper"] for f in truth.faces[2:]]

    changed, again = apply_corrections(path, {"people": {"henry": "Henry Hart"}, "faces": moved, "reviewed": True})

    assert changed == 2
    assert [f.label for f in GroundTruth.load(path).faces] == ["henry", "henry", "jasper", "jasper"]
    assert again.people["henry"] == "Henry Hart" and "jasper" in again.people
    assert again.reviewed
