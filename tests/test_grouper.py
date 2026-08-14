import numpy as np
import pytest

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import (
    FaceObservation,
    IdentityGrouper,
    cosine_similarity,
)


def make_observation(
    vector,
    confidence: float = 0.9,
    face_size: int = 200,
    timestamp: float = 0.0,
) -> FaceObservation:
    """Builds a FaceObservation from a raw (not necessarily normalized) vector."""
    embedding = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    box = BoundingBox(x_min=0, y_min=0, x_max=face_size, y_max=face_size)
    detection = FaceDetection(box=box, confidence=confidence)
    face_crop = np.zeros((4, 4, 3), dtype=np.uint8)
    return FaceObservation(
        embedding=embedding,
        detection=detection,
        face_crop=face_crop,
        source_timestamp=timestamp,
    )


def test_same_person_groups_together():
    """Near-identical embeddings should be assigned to one group."""
    grouper = IdentityGrouper(similarity_threshold=0.5, margin_threshold=0.05)

    first = make_observation([1.0, 0.0, 0.0, 0.0], timestamp=0.0)
    second = make_observation([0.97, 0.1, 0.1, 0.1], timestamp=1.0)
    third = make_observation([0.95, 0.12, 0.1, 0.12], timestamp=2.0)

    group_id_1 = grouper.add(first)
    group_id_2 = grouper.add(second)
    group_id_3 = grouper.add(third)

    assert group_id_1 == group_id_2 == group_id_3
    assert len(grouper.groups) == 1
    assert grouper.groups[0].size == 3


def test_different_people_stay_separate():
    """Clearly different (orthogonal) embeddings should form separate groups."""
    grouper = IdentityGrouper(similarity_threshold=0.5, margin_threshold=0.05)

    person_a = make_observation([1.0, 0.0, 0.0, 0.0], timestamp=0.0)
    person_b = make_observation([0.0, 1.0, 0.0, 0.0], timestamp=1.0)
    person_c = make_observation([0.0, 0.0, 1.0, 0.0], timestamp=2.0)

    group_id_a = grouper.add(person_a)
    group_id_b = grouper.add(person_b)
    group_id_c = grouper.add(person_c)

    assert len({group_id_a, group_id_b, group_id_c}) == 3
    assert len(grouper.groups) == 3
    assert all(group.size == 1 for group in grouper.groups)


def test_ambiguous_match_creates_new_group_instead_of_forcing_one():
    """An observation equidistant between two groups should not be forced into either."""
    grouper = IdentityGrouper(similarity_threshold=0.5, margin_threshold=0.05)

    grouper.add(make_observation([1.0, 0.0, 0.0, 0.0], timestamp=0.0))
    grouper.add(make_observation([0.0, 1.0, 0.0, 0.0], timestamp=1.0))

    # Equidistant (cosine ~0.707) from both existing group centroids, well
    # above the similarity floor for each but with zero margin between them.
    ambiguous = make_observation([1.0, 1.0, 0.0, 0.0], timestamp=2.0)
    group_id = grouper.add(ambiguous)

    assert len(grouper.groups) == 3
    assert group_id not in {grouper.groups[0].group_id, grouper.groups[1].group_id}


def test_a_confident_but_not_top_match_still_requires_margin():
    """Best match clears the threshold alone but is too close to the runner-up to merge."""
    grouper = IdentityGrouper(similarity_threshold=0.5, margin_threshold=0.1)

    grouper.add(make_observation([1.0, 0.0, 0.0], timestamp=0.0))
    grouper.add(make_observation([0.0, 1.0, 0.0], timestamp=1.0))

    # cosine ~0.724 vs group A, ~0.690 vs group B: both individually clear
    # the 0.5 floor, but the gap between them (~0.034) is well under the
    # required 0.1 margin, so this must not be forced into either group.
    candidate = make_observation([1.05, 1.0, 0.0], timestamp=2.0)
    group_id = grouper.add(candidate)

    assert len(grouper.groups) == 3
    assert group_id not in {grouper.groups[0].group_id, grouper.groups[1].group_id}


def test_multiple_people_multiple_groups_interleaved():
    """Detections belonging to different people, interleaved in time, still separate cleanly."""
    grouper = IdentityGrouper(similarity_threshold=0.5, margin_threshold=0.05)

    person_a_1 = make_observation([1.0, 0.0, 0.0, 0.0], timestamp=0.0)
    person_b_1 = make_observation([0.0, 1.0, 0.0, 0.0], timestamp=1.0)
    person_a_2 = make_observation([0.96, 0.1, 0.05, 0.0], timestamp=2.0)
    person_b_2 = make_observation([0.05, 0.97, 0.1, 0.0], timestamp=3.0)

    id_a1 = grouper.add(person_a_1)
    id_b1 = grouper.add(person_b_1)
    id_a2 = grouper.add(person_a_2)
    id_b2 = grouper.add(person_b_2)

    assert id_a1 == id_a2
    assert id_b1 == id_b2
    assert id_a1 != id_b1
    assert len(grouper.groups) == 2


def test_empty_input_produces_no_groups():
    """No observations added should mean no groups."""
    grouper = IdentityGrouper()
    assert grouper.groups == []
    assert grouper.unassigned == []


def test_invalid_embedding_is_handled_safely():
    """Malformed embeddings must not crash the grouper; they're set aside."""
    grouper = IdentityGrouper()

    nan_observation = make_observation([1.0, 0.0, 0.0, 0.0])
    nan_observation = FaceObservation(
        embedding=np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
        detection=nan_observation.detection,
        face_crop=nan_observation.face_crop,
        source_timestamp=0.0,
    )

    wrong_shape_observation = FaceObservation(
        embedding=np.zeros((4, 4), dtype=np.float32),
        detection=nan_observation.detection,
        face_crop=nan_observation.face_crop,
        source_timestamp=1.0,
    )

    assert grouper.add(nan_observation) is None
    assert grouper.add(wrong_shape_observation) is None
    assert grouper.groups == []
    assert len(grouper.unassigned) == 2


def test_low_confidence_detection_left_unassigned():
    """A detection below the confidence floor should not seed or join a group."""
    grouper = IdentityGrouper(min_confidence=0.7)

    observation = make_observation([1.0, 0.0, 0.0, 0.0], confidence=0.5)
    group_id = grouper.add(observation)

    assert group_id is None
    assert grouper.groups == []
    assert grouper.unassigned == [observation]


def test_tiny_face_left_unassigned():
    """A face box smaller than the minimum size should not seed or join a group."""
    grouper = IdentityGrouper(min_face_size=40)

    observation = make_observation([1.0, 0.0, 0.0, 0.0], face_size=20)
    group_id = grouper.add(observation)

    assert group_id is None
    assert grouper.groups == []
    assert grouper.unassigned == [observation]


def test_representative_observation_prefers_higher_quality():
    """A later, higher-quality (bigger/more confident) observation should replace the representative."""
    grouper = IdentityGrouper(similarity_threshold=0.5, margin_threshold=0.05)

    low_quality = make_observation([1.0, 0.0, 0.0, 0.0], confidence=0.75, face_size=60, timestamp=0.0)
    high_quality = make_observation([0.98, 0.1, 0.05, 0.0], confidence=0.97, face_size=300, timestamp=1.0)

    grouper.add(low_quality)
    grouper.add(high_quality)

    assert len(grouper.groups) == 1
    assert grouper.groups[0].representative_observation is high_quality


def test_constructor_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        IdentityGrouper(similarity_threshold=1.5)

    with pytest.raises(ValueError):
        IdentityGrouper(margin_threshold=-0.1)


def test_cosine_similarity_of_identical_normalized_vectors_is_one():
    vector = np.array([0.6, 0.8], dtype=np.float32)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)
