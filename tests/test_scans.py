"""Keeping a scan so the same video is not scanned twice."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from app import scans
from app.faces.detector import BoundingBox, FaceDetection, FaceLandmarks
from app.faces.grouper import FaceIdentityGroup, FaceObservation


def observation(timestamp=0.0, frame_index=0, embedding=None, landmarks=True):
    return FaceObservation(
        embedding=(
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if embedding is None
            else np.asarray(embedding, dtype=np.float32)
        ),
        detection=FaceDetection(
            box=BoundingBox(10, 20, 110, 140),
            confidence=0.91,
            landmarks=(
                FaceLandmarks((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))
                if landmarks
                else None
            ),
        ),
        face_crop=np.full((6, 5, 3), 7, dtype=np.uint8),
        source_timestamp=timestamp,
        frame_index=frame_index,
        sharpness=42.5,
        embedding_space="arcface-w600k-r50",
    )


def group(group_id=1, size=3, representative=1):
    observations = [observation(timestamp=float(i), frame_index=i) for i in range(size)]
    built = FaceIdentityGroup(group_id=group_id, observations=observations)
    built.representative_embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    built.representative_observation = observations[representative]
    return built


def a_scan(**overrides):
    defaults = dict(
        groups=[group(1), group(2, size=2, representative=0)],
        unassigned_count=4,
        total_detections=5,
        track_count=3,
        frame_count=120,
        last_timestamp=119.0,
        embedding_time=1.5,
        grouping_time=0.25,
        video_duration=120.0,
        min_detections=7,
        created_at=time.time(),
        scan_seconds=53.1,
    )
    defaults.update(overrides)
    return scans.CachedScan(**defaults)


SETTINGS = dict(
    sample_interval=0.5,
    confidence_threshold=0.6,
    padding_ratio=0.08,
    similarity_threshold=0.35,
    margin_threshold=0.0,
    consolidation_threshold=0.375,
    min_confidence=0.7,
    min_face_size=40,
    min_group_eye_span=0.15,
    mode="live",
    forbid_cooccurring=True,
    cooccurrence_similarity_ceiling=0.5,
    min_detections=None,
)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"pretend footage")
    return path


# ------------------------------------------------------------------ the key


def test_the_same_video_and_settings_give_the_same_key(video):
    assert scans.cache_key(video, **SETTINGS) == scans.cache_key(video, **SETTINGS)


def test_every_setting_that_changes_the_answer_changes_the_key(video):
    """A key that ignored a threshold would serve one scan's answer for
    another scan's question, which is worse than no cache at all."""
    base = scans.cache_key(video, **SETTINGS)

    changes = {
        "sample_interval": 1.0,
        "confidence_threshold": 0.7,
        "padding_ratio": 0.1,
        "similarity_threshold": 0.4,
        "margin_threshold": 0.05,
        "consolidation_threshold": 0.5,
        "min_confidence": 0.8,
        "min_face_size": 50,
        "min_group_eye_span": 0.2,
        "mode": "animation",
        "forbid_cooccurring": False,
        "cooccurrence_similarity_ceiling": 0.6,
        "min_detections": 5,
    }
    for name, value in changes.items():
        assert scans.cache_key(video, **{**SETTINGS, name: value}) != base, name


def test_editing_the_video_changes_the_key(video):
    before = scans.cache_key(video, **SETTINGS)
    video.write_bytes(b"pretend footage, but longer")

    assert scans.cache_key(video, **SETTINGS) != before


def test_a_release_invalidates_kept_scans(video, monkeypatch):
    """Grouping thresholds have been retuned against real footage more than
    once. A scan from before such a change is not old, it is wrong."""
    before = scans.cache_key(video, **SETTINGS)
    monkeypatch.setattr("app.__version__", "99.0.0")

    assert scans.cache_key(video, **SETTINGS) != before


def test_a_video_that_cannot_be_stated_never_matches(tmp_path):
    missing = tmp_path / "gone.mp4"

    assert scans.cache_key(missing, **SETTINGS) != scans.cache_key(missing, **SETTINGS)


# ------------------------------------------------------------- round trip


def test_a_kept_scan_comes_back_the_same():
    original = a_scan()
    scans.save("key", original)

    back = scans.load("key")

    assert back is not None
    assert back.unassigned_count == original.unassigned_count
    assert back.total_detections == original.total_detections
    assert back.track_count == original.track_count
    assert back.frame_count == original.frame_count
    assert back.video_duration == original.video_duration
    assert back.min_detections == original.min_detections
    assert back.scan_seconds == pytest.approx(original.scan_seconds)
    assert [g.group_id for g in back.groups] == [1, 2]


def test_every_observation_survives_intact():
    scans.save("key", a_scan())

    group_back = scans.load("key").groups[0]
    original = a_scan().groups[0]

    assert len(group_back.observations) == len(original.observations)
    for kept, made in zip(group_back.observations, original.observations):
        np.testing.assert_array_equal(kept.embedding, made.embedding)
        assert kept.detection.box == made.detection.box
        assert kept.detection.confidence == pytest.approx(made.detection.confidence)
        assert kept.detection.landmarks.as_tuple() == made.detection.landmarks.as_tuple()
        assert kept.source_timestamp == made.source_timestamp
        assert kept.frame_index == made.frame_index
        assert kept.sharpness == pytest.approx(made.sharpness)
        assert kept.embedding_space == made.embedding_space


def test_the_representative_keeps_its_crop_and_stays_the_group_s_own():
    """The gallery draws its thumbnail from the representative's crop, and
    only that one is stored -- so it has to be the object in the list."""
    scans.save("key", a_scan())

    group_back = scans.load("key").groups[0]

    assert group_back.representative_observation is group_back.observations[1]
    np.testing.assert_array_equal(
        group_back.representative_observation.face_crop,
        np.full((6, 5, 3), 7, dtype=np.uint8),
    )
    np.testing.assert_allclose(
        group_back.representative_embedding, [0.0, 1.0, 0.0]
    )


def test_crops_that_nothing_reads_are_not_stored():
    """Every non-representative crop is dropped, which is what keeps a
    22-minute episode's scan to a few megabytes."""
    scans.save("key", a_scan())

    group_back = scans.load("key").groups[0]

    assert [o.face_crop is None for o in group_back.observations] == [True, False, True]


def test_a_group_with_no_representative_still_round_trips():
    bare = FaceIdentityGroup(group_id=9, observations=[observation()])
    scans.save("key", a_scan(groups=[bare]))

    group_back = scans.load("key").groups[0]

    assert group_back.group_id == 9
    assert group_back.representative_observation is None
    assert group_back.representative_embedding is None


def test_landmarks_may_be_absent():
    """The animation detector returns none, and none are invented."""
    without = FaceIdentityGroup(group_id=1, observations=[observation(landmarks=False)])
    scans.save("key", a_scan(groups=[without]))

    assert scans.load("key").groups[0].observations[0].detection.landmarks is None


def test_a_scan_with_no_groups_round_trips():
    scans.save("key", a_scan(groups=[]))

    assert scans.load("key").groups == []


# --------------------------------------------------------------- failing


def test_a_missing_entry_is_a_miss_not_an_error():
    assert scans.load("never-stored") is None


def test_an_unreadable_entry_is_a_miss_and_removes_itself():
    """Every caller's fallback is to do the work, so a corrupt file must
    cost a rescan rather than a failed scan."""
    scans.save("key", a_scan())
    path = scans.scan_cache_dir() / "key.npz"
    path.write_bytes(b"not an npz at all")

    assert scans.load("key") is None
    assert not path.exists()


def test_an_entry_from_another_layout_is_ignored():
    scans.save("key", a_scan())
    path = scans.scan_cache_dir() / "key.npz"
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    manifest = json.loads(str(arrays["manifest"]))
    manifest["version"] = scans.CACHE_VERSION + 1
    arrays["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **arrays)

    assert scans.load("key") is None


def test_an_interrupted_write_leaves_the_previous_scan_intact(monkeypatch):
    scans.save("key", a_scan(frame_count=1))

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(scans.np, "savez_compressed", explode)
    with pytest.raises(OSError):
        scans.save("key", a_scan(frame_count=999))

    assert scans.load("key").frame_count == 1


# -------------------------------------------------------------- managing


def test_listing_reports_what_is_kept():
    scans.save("one", a_scan())
    scans.save("two", a_scan())

    kept = scans.entries()

    assert len(kept) == 2
    assert scans.total_bytes() == sum(entry.size_bytes for entry in kept)


def test_clearing_removes_everything():
    scans.save("one", a_scan())
    scans.save("two", a_scan())

    assert scans.clear() == 2
    assert scans.entries() == []
    assert scans.load("one") is None


def test_the_oldest_scans_go_when_the_cache_outgrows_its_limit():
    for name in ("old", "middle", "new"):
        scans.save(name, a_scan())
        time.sleep(0.01)

    removed = scans.prune(max_bytes=1)

    assert removed == 3
    assert scans.entries() == []


def test_pruning_keeps_what_fits():
    scans.save("only", a_scan())
    size = scans.total_bytes()

    assert scans.prune(max_bytes=size) == 0
    assert len(scans.entries()) == 1


def test_listing_an_absent_cache_is_empty_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUXCUTTER_SCAN_DIR", str(tmp_path / "never-made"))

    assert scans.entries() == []
    assert scans.total_bytes() == 0
    assert scans.clear() == 0
