"""Tests for content-mode selection.

The property worth protecting is that the mode is a *choice*: nothing in the
pipeline may inspect a video and pick for the user, and nothing may quietly
apply one mode's numbers to the other's models.
"""

import numpy as np
import pytest

from app import modes
from app.faces.grouper import (
    FaceObservation,
    IdentityGrouper,
    MixedEmbeddingSpaces,
)
from app.faces.detector import BoundingBox, FaceDetection


# ----------------------------------------------------------- the registry


def test_both_modes_are_registered():
    assert modes.mode_ids() == [modes.LIVE, modes.ANIMATION]


def test_live_action_is_the_default():
    """A fresh install must behave exactly as FluxCutter always has."""
    assert modes.DEFAULT_MODE == modes.LIVE


def test_every_mode_has_usable_metadata():
    for mode_id in modes.mode_ids():
        spec = modes.get_mode(mode_id)
        assert spec.id == mode_id
        assert spec.display_name and spec.summary
        assert spec.embedding_space
        assert spec.detector_model and spec.embedder_model
        assert callable(spec.build_detector) and callable(spec.build_embedder)


def test_an_unknown_mode_names_the_valid_ones():
    """This is reachable from a command line, so the message has to help."""
    with pytest.raises(KeyError, match="live"):
        modes.get_mode("cartoon")


def test_the_two_modes_use_different_embedding_spaces():
    """The whole basis for refusing to compare their embeddings."""
    live = modes.get_mode(modes.LIVE).embedding_space
    animation = modes.get_mode(modes.ANIMATION).embedding_space
    assert live != animation


def test_the_two_modes_use_different_models():
    live = modes.get_mode(modes.LIVE)
    animation = modes.get_mode(modes.ANIMATION)
    assert live.detector_model.filename != animation.detector_model.filename
    assert live.embedder_model.filename != animation.embedder_model.filename


def test_live_action_thresholds_are_unchanged():
    """Pins the existing pipeline: this mode is not a place to experiment."""
    grouping = modes.get_mode(modes.LIVE).grouping
    assert grouping.similarity_threshold == 0.35
    assert grouping.consolidation_threshold == 0.375
    assert grouping.contradiction_floor == 0.25
    assert modes.get_mode(modes.LIVE).detection.confidence_threshold == 0.6


def test_animation_thresholds_are_independent_of_live_action():
    """CCIP similarity is on a different scale; sharing numbers would merge
    the entire cast into one character."""
    live = modes.get_mode(modes.LIVE).grouping
    animation = modes.get_mode(modes.ANIMATION).grouping
    assert animation.similarity_threshold != live.similarity_threshold
    assert animation.contradiction_floor != live.contradiction_floor
    assert animation.similarity_threshold > live.similarity_threshold


def test_animation_does_not_ask_for_landmark_geometry():
    """Its detector produces no landmarks, so the non-face filter is inert.

    Set to zero deliberately rather than left holding a live-action number
    that silently never fires.
    """
    assert modes.get_mode(modes.ANIMATION).grouping.min_group_eye_span == 0.0


# ------------------------------------------------------------ availability


def test_availability_reports_what_is_missing(monkeypatch):
    # Both axes are pinned. Stubbing only the weights leaves the runtime
    # check real, so the answer would depend on whether the machine running
    # the tests happens to have the optional onnxruntime installed -- which
    # is how this passed locally and failed on CI.
    monkeypatch.setattr(modes, "missing_requirements", lambda spec: ())
    monkeypatch.setattr(modes, "find_model", lambda spec: None)
    state = modes.availability(modes.ANIMATION)
    assert not state.usable
    assert state.download_megabytes > 100
    assert "MB" in state.describe()


def test_a_missing_runtime_is_reported_as_such(monkeypatch):
    monkeypatch.setattr(modes, "missing_requirements", lambda spec: ("onnxruntime",))
    state = modes.availability(modes.ANIMATION)
    assert not state.usable and not state.installable
    assert "pip install onnxruntime" in state.describe()


def test_live_action_needs_no_extra_runtime():
    """It must never become unusable because an optional package is absent."""
    assert modes.get_mode(modes.LIVE).extra_requirements == ()
    assert modes.missing_requirements(modes.get_mode(modes.LIVE)) == ()


# ------------------------------------------------- models are built lazily


def test_importing_modes_loads_no_model(monkeypatch):
    """Selecting a mode must not cost the load of the one not selected.

    The two sets of weights are close to a gigabyte together, so building
    them eagerly is the difference between running and not running on a
    modest machine.
    """
    import sys

    # Neither backend module should be imported merely by importing modes.
    for module in ("app.faces.anime",):
        sys.modules.pop(module, None)
    import importlib

    importlib.reload(modes)
    assert "app.faces.anime" not in sys.modules


def test_each_mode_builds_its_own_detector(monkeypatch):
    """Switching the mode switches the implementation, not just a label."""
    built = []
    monkeypatch.setattr(modes, "_live_detector", lambda **kw: built.append("live"))
    monkeypatch.setattr(modes, "_anime_detector", lambda **kw: built.append("anime"))
    # Rebuild the specs so they pick up the patched factories.
    live = modes.MODES[modes.LIVE]
    animation = modes.MODES[modes.ANIMATION]
    assert live.build_detector is not animation.build_detector
    assert live.build_embedder is not animation.build_embedder


# --------------------------------------------- embeddings cannot be mixed


def observation(vector, space):
    embedding = np.array(vector, dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return FaceObservation(
        embedding=embedding,
        detection=FaceDetection(
            box=BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200), confidence=0.9
        ),
        face_crop=np.zeros((4, 4, 3), np.uint8),
        source_timestamp=0.0,
        embedding_space=space,
    )


def test_two_embedding_spaces_cannot_reach_one_grouper():
    """A live-action vector and an animation vector are not comparable.

    Cosine similarity between them is a perfectly well-formed number, which
    is exactly why this has to be refused rather than left to a threshold.
    """
    grouper = IdentityGrouper(min_detections=1)
    grouper.add(observation([1.0, 0.0, 0.0], "arcface-w600k-r50"))

    with pytest.raises(MixedEmbeddingSpaces, match="not comparable"):
        grouper.add(observation([0.0, 1.0, 0.0], "ccip-caformer-24"))


def test_one_space_groups_normally():
    grouper = IdentityGrouper(min_detections=1)
    grouper.add(observation([1.0, 0.0, 0.0], "ccip-caformer-24"))
    grouper.add(observation([0.99, 0.14, 0.0], "ccip-caformer-24"))
    assert len(grouper.groups) == 1


def test_unlabelled_observations_stay_usable():
    """Hand-built observations name no space and must not be locked out."""
    grouper = IdentityGrouper(min_detections=1)
    grouper.add(observation([1.0, 0.0, 0.0], None))
    grouper.add(observation([0.99, 0.14, 0.0], None))
    assert len(grouper.groups) == 1


def test_a_track_is_checked_too():
    """add_track is the path the real pipeline uses."""

    class _Track:
        def __init__(self, observations):
            self.observations = observations

    grouper = IdentityGrouper(min_detections=1)
    grouper.add(observation([1.0, 0.0, 0.0], "arcface-w600k-r50"))
    with pytest.raises(MixedEmbeddingSpaces):
        grouper.add_track(_Track([observation([0.0, 1.0, 0.0], "ccip-caformer-24")]))
