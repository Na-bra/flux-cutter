from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.faces.detector import FaceDetector
from app.faces.embedder import EMBEDDING_DIMENSIONS, FaceEmbedder
from app.video.frames import extract_frames
from app.video.loader import load_video

VIDEO_PATH = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test.mp4"
DETECTOR_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "face_detection_yunet_2026may.onnx"
EMBEDDER_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "models"
    / "face_recognition_arcface_w600k_r50.onnx"
)


@pytest.fixture(scope="module")
def detector():
    if not DETECTOR_MODEL_PATH.is_file():
        pytest.skip(f"YuNet model not found at '{DETECTOR_MODEL_PATH}'.")

    detector_instance = FaceDetector(model_path=DETECTOR_MODEL_PATH, confidence_threshold=0.6)
    try:
        yield detector_instance
    finally:
        detector_instance.close()


@pytest.fixture(scope="module")
def embedder():
    if not EMBEDDER_MODEL_PATH.is_file():
        pytest.skip(
            f"ArcFace model not found at '{EMBEDDER_MODEL_PATH}'. Extract w600k_r50.onnx from the InsightFace "
            "buffalo_l bundle before running embedder tests."
        )

    embedder_instance = FaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
    try:
        yield embedder_instance
    finally:
        embedder_instance.close()


@pytest.fixture(scope="module")
def sampled_faces(detector):
    """Real (timestamp, frame, detection) triples sampled across the test video."""
    with load_video(VIDEO_PATH) as container:
        frames = extract_frames(container, sample_interval=1.0)

    faces = []
    for timestamp, frame in frames:
        for detection in detector.detect(frame):
            faces.append((timestamp, frame, detection))

    if len(faces) < 2:
        pytest.skip("Not enough real detections in the sample video to validate the embedder.")

    return faces


def test_embed_produces_l2_normalized_vector(embedder, sampled_faces):
    """Tests that a real face crop embeds into a unit-norm 512-d vector."""
    _, frame, detection = sampled_faces[0]
    embedding = embedder.embed(frame, detection)

    assert embedding.shape == (EMBEDDING_DIMENSIONS,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)


def test_embed_same_face_is_self_similar(embedder, sampled_faces):
    """Tests that embedding the same real detection twice yields near-identical vectors."""
    _, frame, detection = sampled_faces[0]

    embedding_a = embedder.embed(frame, detection)
    embedding_b = embedder.embed(frame, detection)

    assert float(np.dot(embedding_a, embedding_b)) == pytest.approx(1.0, abs=1e-4)


def test_embed_different_real_faces_are_dissimilar(embedder, sampled_faces):
    """Tests that two detections sampled far apart in the trailer aren't near-identical.

    This isn't a guarantee of different identity (the trailer could re-show
    the same actor), but it validates the embedder produces a real,
    non-degenerate identity signal rather than a constant/near-constant
    vector, which cosine similarity near 1.0 for every pair would indicate.
    """
    _, frame_a, detection_a = sampled_faces[0]
    _, frame_b, detection_b = sampled_faces[-1]

    embedding_a = embedder.embed(frame_a, detection_a)
    embedding_b = embedder.embed(frame_b, detection_b)

    assert float(np.dot(embedding_a, embedding_b)) < 0.95


def test_embed_rejects_invalid_frame(embedder, sampled_faces):
    """Tests that a malformed frame raises rather than crashing into cv2 internals."""
    _, _, detection = sampled_faces[0]

    with pytest.raises(ValueError):
        embedder.embed(None, detection)

    with pytest.raises(ValueError):
        embedder.embed(np.zeros((10, 10), dtype=np.uint8), detection)


def test_embed_requires_landmarks(embedder, sampled_faces):
    """Tests that a detection without landmarks can't be aligned/embedded."""
    _, frame, detection = sampled_faces[0]
    detection_without_landmarks = replace(detection, landmarks=None)

    with pytest.raises(ValueError):
        embedder.embed(frame, detection_without_landmarks)


def test_align_produces_canonical_arcface_crop(embedder, sampled_faces):
    """Tests that alignment emits the 112x112 pose ArcFace was trained on."""
    _, frame, detection = sampled_faces[0]
    aligned = FaceEmbedder.align(frame, detection)

    assert aligned.shape == (112, 112, 3)


def test_align_is_deterministic(embedder, sampled_faces):
    """Tests that aligning the same face twice is bit-identical.

    This guards a deliberate implementation choice: the similarity
    transform is solved in closed form (Umeyama) rather than with
    cv2.estimateAffinePartial2D, whose RANSAC/LMEDS sampling can return a
    slightly different matrix per call. Non-deterministic alignment would
    make the same face embed to two different vectors between runs, which
    identity grouping compares by cosine similarity.
    """
    _, frame, detection = sampled_faces[0]

    first = FaceEmbedder.align(frame, detection)
    second = FaceEmbedder.align(frame, detection)

    assert np.array_equal(first, second)


def test_align_requires_landmarks(embedder, sampled_faces):
    """Tests that alignment refuses a detection with no landmarks."""
    _, frame, detection = sampled_faces[0]
    detection_without_landmarks = replace(detection, landmarks=None)

    with pytest.raises(ValueError):
        FaceEmbedder.align(frame, detection_without_landmarks)
