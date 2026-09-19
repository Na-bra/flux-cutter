"""The people library: names remembered across every video.

Faces here are unit vectors chosen so the scores are known; the floor is
live action's 0.35 and the margin the batch rule's 0.05.
"""

import numpy as np
import pytest

from app import scans
from app.faces import library
from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.faces.library import KnownPerson, known, suggest
from app.scans import CachedScan

SPACE = "arcface-w600k-r50"
FLOOR = 0.35


def face(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def card(vector, name=None, space=SPACE, group_id=0):
    vector = np.asarray(vector, dtype=np.float32)
    observation = FaceObservation(
        embedding=vector,
        detection=FaceDetection(box=BoundingBox(0, 0, 10, 10), confidence=0.9),
        face_crop=None,
        source_timestamp=0.0, embedding_space=space,
    )
    return FaceIdentityGroup(
        group_id=group_id, observations=[observation],
        representative_embedding=vector, name=name,
    )


A, B, C = face(1, 0, 0, 0), face(0, 1, 0, 0), face(0, 0, 1, 0)


# ------------------------------------------------------------ remembering


def test_a_named_card_is_remembered_under_its_name():
    library.remember("scan-1", [card(A, "Jamie Lee"), card(B)])

    people = known()
    assert [p.name for p in people] == ["Jamie Lee"]
    assert people[0].videos == 1
    np.testing.assert_allclose(people[0].faces[0], A, atol=1e-5)


def test_one_face_is_kept_per_video_named_in():
    library.remember("scan-1", [card(A, "Jamie")])
    library.remember("scan-2", [card(face(1, 0.2, 0, 0), "jamie")])

    jamie = library.find("JAMIE")
    assert jamie.videos == 2
    assert jamie.faces.shape == (2, 4)


def test_renaming_moves_the_face_and_clearing_forgets_it():
    library.remember("scan-1", [card(A, "Jamie")])
    library.remember("scan-1", [card(A, "Sam")])
    assert [p.name for p in known()] == ["Sam"]

    library.remember("scan-1", [card(A)])
    assert known() == []


def test_a_scan_pruned_away_still_taught_the_library():
    """Faces outlive the scan they came from; only renaming removes them."""
    library.remember("scan-1", [card(A, "Jamie")])
    library.remember("scan-2", [card(B)])

    assert library.find("Jamie") is not None


def test_saving_a_scan_updates_the_library():
    """Naming reaches the library through the scan, with nothing of its own."""
    scans.save("key-1", CachedScan(groups=[card(A, "Jamie", group_id=1)], frame_count=1))

    assert library.find("Jamie") is not None


def test_names_given_before_the_library_existed_are_found():
    scans.save("old", CachedScan(groups=[card(A, "Jamie", group_id=1)], frame_count=1))
    library._path().unlink()

    assert library.find("Jamie") is not None


def test_a_damaged_library_is_ignored_not_fatal():
    library.library_dir().mkdir(parents=True, exist_ok=True)
    library._path().write_text("{not json")

    assert known() == []
    library.remember("scan-1", [card(A, "Jamie")])
    assert library.find("Jamie") is not None


# ----------------------------------------------------------- suggesting


def person(name, *faces):
    return KnownPerson(name=name, space=SPACE, faces=np.stack(faces), videos=len(faces))


def test_a_card_that_looks_like_someone_named_is_suggested():
    found = suggest([card(B), card(face(1, 0.1, 0, 0))], FLOOR, people=[person("Jamie", A)])

    assert [(s.card, s.name) for s in found] == [(1, "Jamie")]
    assert found[0].similarity > 0.99


def test_nothing_is_suggested_for_strangers():
    assert suggest([card(B), card(C)], FLOOR, people=[person("Jamie", A)]) == []


def test_two_cards_equally_like_someone_get_no_suggestion():
    """Which of them is Jamie is a question, not a guess."""
    groups = [card(face(1, 0.1, 0, 0)), card(face(1, 0, 0.1, 0))]
    assert suggest(groups, FLOOR, people=[person("Jamie", A)]) == []


def test_a_card_equally_like_two_people_gets_no_suggestion():
    groups = [card(face(1, 1, 0, 0))]
    assert suggest(groups, FLOOR, people=[person("Jamie", A), person("Sam", B)]) == []


def test_any_face_the_person_was_seen_with_can_match():
    """Kept per video so a changed look still matches the episode it was in."""
    later_look = face(0, 0, 1, 0.1)
    found = suggest([card(later_look)], FLOOR, people=[person("Jamie", A, C)])

    assert [s.name for s in found] == ["Jamie"]


def test_a_name_already_on_a_card_here_is_not_suggested_again():
    groups = [card(A, "Jamie"), card(face(1, 0.1, 0, 0))]
    assert suggest(groups, FLOOR, people=[person("Jamie", A)]) == []


def test_named_cards_are_not_given_suggestions():
    assert suggest([card(A, "Someone else")], FLOOR, people=[person("Jamie", A)]) == []


def test_people_from_another_mode_are_never_suggested():
    library.remember("scan-1", [card(A, "Jamie", space="ccip")])
    assert suggest([card(A)], FLOOR) == []


def test_forgetting_removes_a_name_and_its_faces():
    library.remember("scan-1", [card(A, "Jamie"), card(B, "Sam")])

    assert library.forget("jamie") == 1
    assert [p.name for p in known()] == ["Sam"]
    assert library.forget("nobody") == 0


def test_the_people_command_lists_and_forgets(monkeypatch, capsys):
    import sys
    from app import __main__ as cli

    library.remember("scan-1", [card(A, "Jamie Lee")])

    monkeypatch.setattr(sys, "argv", ["app", "people"])
    cli.main()
    assert "Jamie Lee" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["app", "people", "forget", "Jamie Lee"])
    cli.main()
    assert "Forgot Jamie Lee." in capsys.readouterr().out
    assert known() == []
