"""Choosing a person with a photograph rather than a gallery click."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.faces.reference import (
    DEFAULT_MATCH_MARGIN,
    ReferenceError,
    ReferenceFace,
    load_image,
    load_reference_face,
    match_reference,
    score_groups,
)

SPACE = "arcface-w600k-r50"


def unit(*values) -> np.ndarray:
    """A normalized embedding, so cosine similarity is the dot product."""
    vector = np.array(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


def group(embedding, group_id=1, space=SPACE, size=1) -> FaceIdentityGroup:
    observations = [
        FaceObservation(
            embedding=embedding,
            detection=None,
            face_crop=np.zeros((2, 2, 3), dtype=np.uint8),
            source_timestamp=float(index),
            embedding_space=space,
        )
        for index in range(size)
    ]
    return FaceIdentityGroup(
        group_id=group_id,
        observations=observations,
        representative_embedding=embedding,
    )


def reference(embedding, space=SPACE, name="photo.jpg") -> ReferenceFace:
    return ReferenceFace(
        embedding=embedding,
        embedding_space=space,
        detection=None,
        source=Path(name),
        face_count=1,
    )


# --------------------------------------------------------------- reading it


@pytest.mark.runs_without_assets
def test_a_missing_reference_is_refused_by_name(tmp_path):
    with pytest.raises(ReferenceError, match="does not exist"):
        load_image(tmp_path / "nobody.jpg")


@pytest.mark.runs_without_assets
def test_a_video_is_not_a_reference_photo(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not an image")

    with pytest.raises(ReferenceError, match="Unsupported reference image format"):
        load_image(path)


@pytest.mark.runs_without_assets
def test_a_file_that_is_not_really_an_image_is_refused(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"still not an image")

    with pytest.raises(ReferenceError, match="Could not read"):
        load_image(path)


@pytest.mark.runs_without_assets
def test_a_photo_is_read_as_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (8, 6), color=128).save(path)

    image = load_image(path)

    assert image.shape == (6, 8, 3)
    assert image.dtype == np.uint8


# ---------------------------------------------------------------- scoring it


@pytest.mark.runs_without_assets
def test_the_closest_identity_wins():
    groups = [
        group(unit(1, 0, 0), group_id=1),
        group(unit(0, 1, 0), group_id=2),
        group(unit(0.9, 0.1, 0), group_id=3),
    ]

    match = match_reference(reference(unit(0, 1, 0)), groups)

    assert match.index == 1
    assert match.similarity == pytest.approx(1.0)
    assert match.margin > DEFAULT_MATCH_MARGIN


@pytest.mark.runs_without_assets
def test_a_person_who_is_not_in_the_video_is_reported_as_absent():
    groups = [group(unit(1, 0, 0)), group(unit(0, 1, 0))]

    with pytest.raises(ReferenceError, match="No identity in this video matches"):
        match_reference(reference(unit(0, 0, 1)), groups)


@pytest.mark.runs_without_assets
def test_two_equally_good_matches_are_not_guessed_between():
    """A confident wrong answer here costs minutes of encoding the wrong face."""
    groups = [group(unit(1, 0, 0), group_id=1), group(unit(1, 0.02, 0), group_id=2)]

    with pytest.raises(ReferenceError, match="about equally well"):
        match_reference(reference(unit(1, 0.01, 0)), groups)


@pytest.mark.runs_without_assets
def test_one_identity_has_no_runner_up_to_be_ambiguous_against():
    match = match_reference(reference(unit(1, 0, 0)), [group(unit(1, 0, 0))])

    assert match.index == 0
    assert match.runner_up_similarity is None
    assert match.margin is None


@pytest.mark.runs_without_assets
def test_a_video_with_no_identities_says_so():
    with pytest.raises(ReferenceError, match="no identities"):
        match_reference(reference(unit(1, 0, 0)), [])


@pytest.mark.runs_without_assets
def test_the_floor_can_be_lowered_for_a_hard_photo():
    groups = [group(unit(1, 0, 0)), group(unit(0, 1, 0))]
    # Scores 0.32 against the first identity: under the 0.35 default
    # floor, over a lowered one.
    photo = reference(unit(0.32, 0.0, 0.9474))

    with pytest.raises(ReferenceError):
        match_reference(photo, groups)

    assert match_reference(photo, groups, minimum_similarity=0.3).index == 0


@pytest.mark.runs_without_assets
def test_embeddings_from_two_different_models_are_never_compared():
    """An ArcFace vector and a CCIP one have a cosine similarity, and it
    means nothing. Matching across them would answer confidently and wrongly."""
    groups = [group(unit(1, 0, 0), space="ccip-caformer-24")]

    with pytest.raises(ReferenceError, match="same mode"):
        score_groups(reference(unit(1, 0, 0)), groups)


@pytest.mark.runs_without_assets
def test_an_identity_with_no_centroid_is_scored_out_of_contention():
    groups = [group(unit(1, 0, 0)), group(unit(1, 0, 0), group_id=2)]
    groups[0].representative_embedding = None

    assert score_groups(reference(unit(1, 0, 0)), groups) == [-1.0, pytest.approx(1.0)]


# ------------------------------------------------ against the real detector


def test_a_photo_with_nobody_in_it_is_refused(tmp_path):
    path = tmp_path / "wall.jpg"
    Image.new("RGB", (256, 256), color=(120, 120, 120)).save(path)

    with pytest.raises(ReferenceError, match="No face found"):
        load_reference_face(path)
