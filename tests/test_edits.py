"""Correcting the identities a scan produced."""

import numpy as np
import pytest

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.edits import (
    EditError,
    discard_groups,
    merge_groups,
    rename_group,
    split_group,
)
from app.faces.grouper import FaceIdentityGroup, FaceObservation


def observation(timestamp, direction=0):
    vector = np.zeros(8, dtype=np.float32)
    vector[direction] = 1.0
    return FaceObservation(
        embedding=vector,
        detection=FaceDetection(box=BoundingBox(0, 0, 40, 50), confidence=0.9),
        face_crop=np.full((5, 4, 3), 9, dtype=np.uint8),
        source_timestamp=float(timestamp),
        frame_index=int(timestamp),
        embedding_space="arcface-w600k-r50",
    )


def group(tracks, direction=0):
    """A group built from tracks, the way clustering builds one."""
    observations = [
        observation(t, direction) for track in tracks for t in track
    ]
    return FaceIdentityGroup(
        group_id=0,
        observations=observations,
        unit_sizes=[len(track) for track in tracks],
    )


@pytest.fixture
def gallery():
    """Three people: 5, 3 and 2 detections, largest first."""
    return [
        group([[0, 1, 2], [10, 11]], direction=0),
        group([[20, 21, 22]], direction=1),
        group([[30], [40]], direction=2),
    ]


# ------------------------------------------------------------------- tracks


def test_a_group_remembers_the_tracks_it_was_built_from():
    built = group([[0, 1, 2], [10, 11]])

    assert [len(track) for track in built.tracks] == [3, 2]
    assert [o.source_timestamp for o in built.tracks[1]] == [10.0, 11.0]


def test_a_group_built_by_hand_is_one_track():
    """Nothing recorded means nothing to split along, not zero tracks."""
    bare = FaceIdentityGroup(group_id=1, observations=[observation(1)])

    assert len(bare.tracks) == 1


# -------------------------------------------------------------------- merge


def test_merging_two_cards_makes_one_person(gallery):
    merged = merge_groups(gallery, [1, 2])

    assert len(merged) == 2
    assert [g.size for g in merged] == [5, 5]
    assert sorted(o.source_timestamp for o in merged[1].observations) == [
        20.0,
        21.0,
        22.0,
        30.0,
        40.0,
    ]


def test_a_merged_card_can_still_be_split_back_apart(gallery):
    """The tracks concatenate, so the correction is not one-way."""
    merged = merge_groups(gallery, [1, 2])
    combined = next(g for g in merged if g.size == 5 and g.unit_sizes == [3, 1, 1])

    assert [len(t) for t in combined.tracks] == [3, 1, 1]


def test_a_merged_card_gets_its_own_face_and_centroid(gallery):
    merged = merge_groups(gallery, [0, 1])
    biggest = merged[0]

    assert biggest.representative_observation in biggest.observations
    assert biggest.representative_embedding is not None
    assert biggest.representative_embedding.shape == (8,)


def test_merging_needs_two_people(gallery):
    with pytest.raises(EditError, match="two people or more"):
        merge_groups(gallery, [1])
    with pytest.raises(EditError, match="two people or more"):
        merge_groups(gallery, [1, 1])


def test_merging_somebody_who_is_not_there(gallery):
    with pytest.raises(EditError, match="no person #9"):
        merge_groups(gallery, [0, 8])


# -------------------------------------------------------------------- split


def test_splitting_moves_the_chosen_tracks_into_a_new_card(gallery):
    split = split_group(gallery, 0, [1])

    assert len(split) == 4
    assert sorted(g.size for g in split) == [2, 2, 3, 3]
    peeled = next(g for g in split if [o.source_timestamp for o in g.observations] == [10.0, 11.0])
    assert peeled.unit_sizes == [2]


def test_what_stays_behind_keeps_its_own_tracks(gallery):
    split = split_group(gallery, 0, [1])

    stayed = next(
        g for g in split if [o.source_timestamp for o in g.observations] == [0.0, 1.0, 2.0]
    )
    assert stayed.unit_sizes == [3]


def test_a_person_seen_in_one_unbroken_run_cannot_be_split(gallery):
    with pytest.raises(EditError, match="single unbroken track"):
        split_group(gallery, 1, [0])


def test_moving_every_track_out_is_not_a_split(gallery):
    with pytest.raises(EditError, match="nobody behind"):
        split_group(gallery, 0, [0, 1])


def test_splitting_needs_a_track(gallery):
    with pytest.raises(EditError, match="at least one track"):
        split_group(gallery, 0, [])


def test_splitting_along_a_track_that_is_not_there(gallery):
    with pytest.raises(EditError, match="no track 5"):
        split_group(gallery, 0, [5])


# ------------------------------------------------------------------ discard


def test_discarding_removes_a_card(gallery):
    kept = discard_groups(gallery, [1])

    assert [g.size for g in kept] == [5, 2]


def test_discarding_several_at_once(gallery):
    assert [g.size for g in discard_groups(gallery, [1, 2])] == [5]


def test_discarding_everyone_is_refused(gallery):
    with pytest.raises(EditError, match="discard everyone"):
        discard_groups(gallery, [0, 1, 2])


# ------------------------------------------------------------------ ordering


def test_every_edit_leaves_the_gallery_ordered_largest_first(gallery):
    for edited in (
        merge_groups(gallery, [1, 2]),
        split_group(gallery, 0, [1]),
        discard_groups(gallery, [0]),
    ):
        # All three are computed before this loop runs, so they also have
        # to be independent of each other -- see the numbering test below.
        sizes = [g.size for g in edited]
        assert sizes == sorted(sizes, reverse=True), sizes
        assert [g.group_id for g in edited] == list(range(1, len(edited) + 1))


def test_an_edit_does_not_disturb_the_gallery_it_was_given(gallery):
    """The caller keeps its own list; an edit returns a new one."""
    before = [g.size for g in gallery]

    discard_groups(gallery, [1])

    assert [g.size for g in gallery] == before


def test_two_edits_of_the_same_gallery_do_not_renumber_each_other():
    """Groups survive an edit untouched, so numbering has to produce new
    objects: stamping an id onto a shared one reached into the caller's
    gallery and into any earlier edit still holding it."""
    start = [
        group([[0, 1, 2]], direction=0),
        group([[20, 21]], direction=1),
        group([[30]], direction=2),
    ]

    first = discard_groups(start, [2])
    second = discard_groups(start, [0])

    assert [g.group_id for g in first] == [1, 2]
    assert [g.group_id for g in second] == [1, 2]
    assert [g.group_id for g in start] == [0, 0, 0]


def test_an_edit_never_loses_a_card_that_was_not_touched():
    """A group with no cover picture is dropped by the gallery, so an edit
    that left one that way would make an untouched person disappear."""
    bare = FaceIdentityGroup(
        group_id=0,
        observations=[observation(1), observation(2)],
        unit_sizes=[2],
    )
    other = group([[10, 11, 12]], direction=1)
    third = group([[20]], direction=2)

    kept = discard_groups([bare, other, third], [2])

    assert len(kept) == 2
    for card in kept:
        assert card.representative_observation is not None
        assert card.representative_embedding is not None


# ------------------------------------------------------------------- names


def test_naming_a_card_leaves_the_gallery_in_the_same_order(gallery):
    """Renaming changes no membership, so the card must not move out from
    under the cursor that just named it."""
    named = rename_group(gallery, 1, "Jamie")

    assert [g.size for g in named] == [g.size for g in gallery]
    assert named[1].name == "Jamie"
    assert named[0].name is None


def test_a_name_is_tidied_before_it_is_kept(gallery):
    assert rename_group(gallery, 0, "  Jamie   Lee ")[0].name == "Jamie Lee"


def test_an_empty_name_clears_it(gallery):
    named = rename_group(gallery, 0, "Jamie")

    assert rename_group(named, 0, "   ")[0].name is None


def test_a_name_that_could_not_be_a_filename_is_refused(gallery):
    with pytest.raises(EditError, match="cannot contain"):
        rename_group(gallery, 0, "Jamie/Lee")


def test_a_very_long_name_is_refused(gallery):
    with pytest.raises(EditError, match="60 characters"):
        rename_group(gallery, 0, "x" * 61)


def test_renaming_somebody_who_is_not_there(gallery):
    with pytest.raises(EditError, match="no person #9"):
        rename_group(gallery, 8, "Jamie")


def test_merging_keeps_the_name_of_whoever_contributed_most(gallery):
    """Merging the stray half of an actor into the named one is the whole
    point, so losing the name would punish the correction."""
    named = rename_group(gallery, 0, "Jamie")

    merged = merge_groups(named, [0, 2])

    assert merged[0].name == "Jamie"


def test_merging_takes_a_name_from_wherever_there_is_one(gallery):
    named = rename_group(gallery, 2, "Jamie")

    merged = merge_groups(named, [0, 2])

    assert merged[0].name == "Jamie"


def test_the_name_stays_with_who_is_left_behind_after_a_split(gallery):
    """Splitting says "those shots are somebody else", so the person
    keeping the name is the one the user did not point at."""
    named = rename_group(gallery, 0, "Jamie")

    split = split_group(named, 0, [1])

    stayed = next(g for g in split if len(g.observations) == 3 and g.name)
    peeled = next(
        g for g in split if [o.source_timestamp for o in g.observations] == [10.0, 11.0]
    )
    assert stayed.name == "Jamie"
    assert peeled.name is None


def test_a_name_survives_the_renumbering_that_follows_a_discard(gallery):
    """This is why a name belongs to the person and not to the card's
    position: the position changes on every correction."""
    named = rename_group(gallery, 2, "Jamie")

    kept = discard_groups(named, [0])

    assert [g.name for g in kept] == [None, "Jamie"]
    assert kept[1].group_id == 2


def test_naming_cannot_lose_a_card_either():
    """Renaming skips the reordering the other edits go through, so the
    guarantee that no edit drops a card has to be applied there too."""
    bare = FaceIdentityGroup(
        group_id=1, observations=[observation(1), observation(2)], unit_sizes=[2]
    )
    other = FaceIdentityGroup(group_id=2, observations=[observation(9)], unit_sizes=[1])

    named = rename_group([bare, other], 0, "Jamie")

    assert len(named) == 2
    assert named[0].name == "Jamie"
    for card in named:
        assert card.representative_observation is not None
        assert card.representative_embedding is not None
