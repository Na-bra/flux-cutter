import numpy as np
import pytest

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceObservation, IdentityGrouper
from app.faces.tracker import FaceTracker


def make_observation(
    vector,
    box: tuple[int, int, int, int],
    confidence: float = 0.9,
    timestamp: float = 0.0,
    frame_index: int = 0,
) -> FaceObservation:
    """Builds a FaceObservation at an explicit box from a raw vector."""
    embedding = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    x_min, y_min, x_max, y_max = box
    detection = FaceDetection(
        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        confidence=confidence,
    )
    return FaceObservation(
        embedding=embedding,
        detection=detection,
        face_crop=np.zeros((4, 4, 3), dtype=np.uint8),
        source_timestamp=timestamp,
        frame_index=frame_index,
    )


def test_overlapping_detections_link_into_one_track():
    """A face barely moving between frames should stay a single track."""
    tracker = FaceTracker()
    for index in range(4):
        offset = index * 5
        tracker.add_frame(
            index,
            [make_observation([1.0, 0.0, 0.0], (offset, 0, 100 + offset, 100),
                              timestamp=index * 0.25, frame_index=index)],
        )

    tracks = tracker.finish()
    assert len(tracks) == 1
    assert tracks[0].size == 4
    assert tracks[0].start_time == 0.0
    assert tracks[0].end_time == pytest.approx(0.75)


def test_disjoint_detections_start_separate_tracks():
    """Boxes that do not overlap belong to different faces."""
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100))])
    tracker.add_frame(1, [make_observation([1.0, 0.0, 0.0], (500, 500, 600, 600),
                                           timestamp=0.25, frame_index=1)])

    tracks = tracker.finish()
    assert len(tracks) == 2
    assert all(track.size == 1 for track in tracks)


def test_two_faces_in_one_frame_track_independently():
    """Concurrent faces must not steal each other's detections."""
    tracker = FaceTracker()
    for index in range(3):
        tracker.add_frame(
            index,
            [
                make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100),
                                 timestamp=index * 0.25, frame_index=index),
                make_observation([0.0, 1.0, 0.0], (400, 0, 500, 100),
                                 timestamp=index * 0.25, frame_index=index),
            ],
        )

    tracks = tracker.finish()
    assert len(tracks) == 2
    assert sorted(track.size for track in tracks) == [3, 3]


def test_contradiction_veto_breaks_track_on_shot_cut():
    """Overlapping boxes with an unrelated face must not link.

    This is the shot-cut case: the camera cuts and a different person lands
    in the same part of the frame, so IoU alone would happily merge two
    identities.
    """
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100))])
    tracker.add_frame(1, [make_observation([0.0, 1.0, 0.0], (5, 0, 105, 100),
                                           timestamp=0.25, frame_index=1)])

    tracks = tracker.finish()
    assert len(tracks) == 2, "an unrelated face reusing the same screen position must not link"


def test_track_survives_one_missed_frame():
    """A single dropped detection should not split a track."""
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100))])
    tracker.add_frame(1, [])
    tracker.add_frame(2, [make_observation([1.0, 0.0, 0.0], (5, 0, 105, 100),
                                           timestamp=0.5, frame_index=2)])

    tracks = tracker.finish()
    assert len(tracks) == 1
    assert tracks[0].size == 2


def test_track_closes_after_long_gap():
    """Beyond the gap allowance the face is treated as a new appearance."""
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100))])
    tracker.add_frame(5, [make_observation([1.0, 0.0, 0.0], (5, 0, 105, 100),
                                           timestamp=1.25, frame_index=5)])

    tracks = tracker.finish()
    assert len(tracks) == 2


def test_track_embedding_is_the_normalized_mean():
    """The averaged embedding is what makes tracking worth doing."""
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100))])
    tracker.add_frame(1, [make_observation([1.0, 1.0, 0.0], (5, 0, 105, 100),
                                           timestamp=0.25, frame_index=1)])

    track = tracker.finish()[0]
    assert track.size == 2
    assert float(np.linalg.norm(track.embedding)) == pytest.approx(1.0, abs=1e-5)
    # Mean of (1,0,0) and (0.707,0.707,0) leans toward x but carries y.
    assert track.embedding[0] > track.embedding[1] > 0.0


def test_representative_observation_is_highest_confidence():
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100),
                                           confidence=0.75)])
    tracker.add_frame(1, [make_observation([1.0, 0.0, 0.0], (5, 0, 105, 100),
                                           confidence=0.95, timestamp=0.25, frame_index=1)])

    track = tracker.finish()[0]
    assert track.representative_observation.detection.confidence == pytest.approx(0.95)


def test_tracks_are_returned_in_chronological_order():
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 100, 100))])
    tracker.add_frame(1, [make_observation([0.0, 1.0, 0.0], (500, 500, 600, 600),
                                           timestamp=0.25, frame_index=1)])
    tracker.add_frame(2, [make_observation([0.0, 0.0, 1.0], (900, 900, 1000, 1000),
                                           timestamp=0.5, frame_index=2)])

    tracks = tracker.finish()
    assert [track.start_time for track in tracks] == [0.0, 0.25, 0.5]


def test_tracker_rejects_invalid_settings():
    with pytest.raises(ValueError):
        FaceTracker(iou_threshold=1.5)
    with pytest.raises(ValueError):
        FaceTracker(max_frame_gap=-1)
    with pytest.raises(ValueError):
        FaceTracker(contradiction_floor=2.0)


def test_grouping_a_track_files_every_observation():
    """A grouped track contributes all its frames, not just its average.

    The timeline stage reads per-observation timestamps, so a track must not
    collapse into a single representative point when it joins a group.
    """
    tracker = FaceTracker()
    for index in range(3):
        tracker.add_frame(
            index,
            [make_observation([1.0, 0.0, 0.0], (index * 5, 0, 100 + index * 5, 100),
                              timestamp=index * 0.25, frame_index=index)],
        )
    track = tracker.finish()[0]

    grouper = IdentityGrouper()
    group_id = grouper.add_track(track)

    assert group_id is not None
    assert len(grouper.groups) == 1
    assert grouper.groups[0].size == 3
    assert sorted(o.source_timestamp for o in grouper.groups[0].observations) == [0.0, 0.25, 0.5]


def test_grouping_drops_unreliable_frames_from_a_track():
    """Low-confidence frames must not drag a track's averaged identity around."""
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 200, 200),
                                           confidence=0.95)])
    tracker.add_frame(1, [make_observation([1.0, 0.0, 0.0], (5, 0, 205, 200),
                                           confidence=0.2, timestamp=0.25, frame_index=1)])
    track = tracker.finish()[0]
    assert track.size == 2

    grouper = IdentityGrouper(min_confidence=0.7)
    grouper.add_track(track)

    assert grouper.groups[0].size == 1
    assert grouper.groups[0].observations[0].detection.confidence == pytest.approx(0.95)


def test_wholly_unreliable_track_is_left_unassigned():
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.0, 0.0], (0, 0, 200, 200),
                                           confidence=0.2)])
    track = tracker.finish()[0]

    grouper = IdentityGrouper(min_confidence=0.7)

    assert grouper.add_track(track) is None
    assert grouper.groups == []
    assert len(grouper.unassigned) == 1


def test_two_tracks_of_one_person_merge_into_a_single_identity():
    """A split track is recoverable: grouping is what puts it back together."""
    tracker = FaceTracker()
    tracker.add_frame(0, [make_observation([1.0, 0.02, 0.0], (0, 0, 200, 200))])
    tracker.add_frame(9, [make_observation([1.0, 0.0, 0.05], (900, 900, 1100, 1100),
                                           timestamp=2.25, frame_index=9)])
    tracks = tracker.finish()
    assert len(tracks) == 2, "far-apart boxes should not have been linked by IoU"

    grouper = IdentityGrouper()
    for track in tracks:
        grouper.add_track(track)

    assert len(grouper.groups) == 1
    assert grouper.groups[0].size == 2
