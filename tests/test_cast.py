"""Linking cards across a folder of videos into one cast.

Faces here are unit vectors picked so the scores are known exactly; the
floor is live action's 0.35 and the margin is the batch rule's 0.05.
"""

import numpy as np
import pytest

from app.faces.cast import Answers, CardRef, CastCard, build_cast

FLOOR = 0.35


def face(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def card(video, person, vector, name=None, weight=10):
    return CastCard(CardRef(video, person), vector, name, weight)


def who(members):
    """Each person as the set of (video, person) cards they hold."""
    return {frozenset((c.video, c.person) for c in m.cards) for m in members}


A = face(1, 0, 0, 0)
B = face(0, 1, 0, 0)
C = face(0, 0, 1, 0)


def test_the_same_face_in_three_videos_is_one_person():
    members, questions = build_cast(
        [card(0, 0, A), card(1, 0, A), card(2, 0, A), card(0, 1, B), card(1, 1, B)],
        FLOOR,
    )

    assert who(members) == {
        frozenset({(0, 0), (1, 0), (2, 0)}),
        frozenset({(0, 1), (1, 1)}),
    }
    assert questions == []


def test_strangers_stay_apart_and_nobody_is_asked_about_them():
    members, questions = build_cast([card(0, 0, A), card(1, 0, B)], FLOOR)

    assert who(members) == {frozenset({(0, 0)}), frozenset({(1, 0)})}
    assert questions == []


def test_two_equally_good_matches_are_asked_about_not_guessed():
    """One face in video 0; two in video 1 that score within the margin."""
    near_a = face(0.9, 0.1, 0, 0)
    also_near_a = face(0.9, 0, 0.1, 0)
    members, questions = build_cast(
        [card(0, 0, A), card(1, 0, near_a), card(1, 1, also_near_a)], FLOOR
    )

    assert all(len(m.cards) == 1 for m in members)
    assert {q.pair for q in questions} >= {
        frozenset({CardRef(0, 0), CardRef(1, 0)})
    } or {q.pair for q in questions} >= {frozenset({CardRef(0, 0), CardRef(1, 1)})}
    assert questions and questions[0].similarity >= FLOOR


def test_a_match_that_is_not_mutual_is_not_a_link():
    """Video 0's second card likes video 1's card best, but that card likes
    video 0's first card better -- so only the mutual pair links."""
    best = face(1, 0.05, 0, 0)
    second_best = face(1, 0.4, 0, 0)
    members, questions = build_cast(
        [card(0, 0, best), card(0, 1, second_best), card(1, 0, A)], FLOOR
    )

    assert frozenset({(0, 0), (1, 0)}) in who(members)
    assert frozenset({(0, 1)}) in who(members)
    # The runner-up is still worth asking about: it may be a split card.
    assert frozenset({CardRef(0, 1), CardRef(1, 0)}) in {q.pair for q in questions}


def test_a_name_joins_cards_whatever_their_faces_score():
    members, _ = build_cast(
        [card(0, 0, A, "Jamie"), card(1, 0, B, "jamie"), card(1, 1, A)], FLOOR
    )

    jamie = next(m for m in members if m.name)
    assert {(c.video, c.person) for c in jamie.cards} == {(0, 0), (1, 0)}
    # The unnamed card with Jamie's face from video 1 is not added: video 1
    # already has a Jamie, so that is a question, not a link.
    assert frozenset({(1, 1)}) in who(members)


def test_different_names_are_never_joined_or_asked_about():
    members, questions = build_cast(
        [card(0, 0, A, "Jamie"), card(1, 0, A, "Sam")], FLOOR
    )

    assert who(members) == {frozenset({(0, 0)}), frozenset({(1, 0)})}
    assert questions == []


def test_saying_yes_joins_even_a_weak_pair():
    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 0), same=True)

    members, _ = build_cast([card(0, 0, A), card(1, 0, B)], FLOOR, answers=answers)

    assert who(members) == {frozenset({(0, 0), (1, 0)})}


def test_saying_no_keeps_even_a_strong_pair_apart_and_stops_asking():
    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 0), same=False)

    members, questions = build_cast([card(0, 0, A), card(1, 0, A)], FLOOR, answers=answers)

    assert who(members) == {frozenset({(0, 0)}), frozenset({(1, 0)})}
    assert questions == []


def test_changing_an_answer_replaces_it():
    answers = Answers()
    answers.record(CardRef(0, 0), CardRef(1, 0), same=False)
    answers.record(CardRef(0, 0), CardRef(1, 0), same=True)

    members, _ = build_cast([card(0, 0, A), card(1, 0, A)], FLOOR, answers=answers)

    assert who(members) == {frozenset({(0, 0), (1, 0)})}


# A second card for the same actor in one video. Grouping merges any two
# cards whose faces score above its consolidation threshold (0.375 in live
# action), so a split that survives it is a looser likeness than the main
# card -- 0.6 here, against the main card's 1.0.
SPLIT = face(0.6, 0.8, 0, 0)


def test_a_split_card_becomes_a_question_not_a_second_link():
    """Grouping split one actor into two cards in video 0. Both match video
    1's card; the stronger links, and the weaker is put to the person."""
    members, questions = build_cast(
        [card(0, 0, A), card(0, 1, SPLIT), card(1, 0, A)], FLOOR
    )

    assert frozenset({(0, 0), (1, 0)}) in who(members)
    assert frozenset({(0, 1)}) in who(members)
    assert any(q.pair == frozenset({CardRef(0, 1), CardRef(1, 0)}) for q in questions)


def test_a_yes_to_a_split_card_puts_both_cards_in_one_person():
    answers = Answers()
    answers.record(CardRef(0, 1), CardRef(1, 0), same=True)

    members, questions = build_cast(
        [card(0, 0, A), card(0, 1, SPLIT), card(1, 0, A)],
        FLOOR,
        answers=answers,
    )

    assert who(members) == {frozenset({(0, 0), (0, 1), (1, 0)})}
    assert questions == []


def test_the_cast_is_ordered_by_screen_time_and_keeps_names():
    members, _ = build_cast(
        [
            card(0, 0, A, weight=5),
            card(1, 0, A, weight=5),
            card(0, 1, B, name="Sam", weight=100),
        ],
        FLOOR,
    )

    assert members[0].name == "Sam"
    assert members[1].weight == 10


def test_a_card_with_no_face_is_its_own_person():
    members, questions = build_cast([card(0, 0, None), card(1, 0, A)], FLOOR)

    assert len(members) == 2
    assert questions == []


def test_questions_come_most_likely_first():
    members, questions = build_cast(
        [
            card(0, 0, A),
            card(1, 0, face(1, 0.9, 0, 0)),
            card(1, 1, face(1, 0.95, 0, 0)),
            card(0, 1, C),
            card(1, 2, face(0.5, 0, 1, 0)),
            card(1, 3, face(0.5, 0, 0.95, 0)),
        ],
        FLOOR,
    )

    scores = [q.similarity for q in questions]
    assert scores == sorted(scores, reverse=True)
    assert len(questions) >= 2
