"""Tests for the animation backend.

Split by cost: the parts that need neither the runtime nor 195 MB of weights
are checked always, and the ones that actually run a model are marked so a
clone without them still gets a green, honest run.
"""

from pathlib import Path

import numpy as np
import pytest

from app.faces import anime
from app.faces.detector import BoundingBox, FaceDetection

MODELS = Path(__file__).resolve().parents[1] / "assets" / "models"
VIDEO = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "animation.mp4"

needs_runtime = pytest.mark.skipif(
    not anime.onnxruntime_available(), reason="onnxruntime is not installed"
)
needs_weights = pytest.mark.skipif(
    not (MODELS / "anime_face_detection_v1.1_s.onnx").is_file(),
    reason="animation weights are not installed",
)


def test_the_animation_embedding_space_is_named():
    """Every vector this module makes is stamped, or the grouper cannot
    tell it apart from an ArcFace one."""
    assert anime.ANIME_EMBEDDING_SPACE
    from app.faces.embedder import LIVE_EMBEDDING_SPACE

    assert anime.ANIME_EMBEDDING_SPACE != LIVE_EMBEDDING_SPACE


def test_detection_settings_are_animation_specific():
    """Defaults here must not be the live-action ones."""
    settings = anime.AnimeDetectorSettings()
    assert settings.confidence_threshold == 0.30
    assert settings.min_face_size < 40


def test_a_clear_error_when_the_runtime_is_missing(monkeypatch):
    """The failure has to name the package and the fix."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(anime.AnimeModelUnavailable, match="pip install onnxruntime"):
        anime._session(Path("unused.onnx"))


def test_a_missing_weights_file_is_reported_clearly():
    with pytest.raises(FileNotFoundError, match="not found"):
        anime._resolve(Path("/nowhere/anime.onnx"), "anime_detector")


def test_letterboxing_preserves_aspect_ratio():
    """A squashed frame changes every face's proportions before the model
    ever sees it."""
    tall = np.zeros((200, 100, 3), np.uint8)
    canvas, scale, pad_x, pad_y = anime._letterbox(tall, 640)
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(3.2)
    assert pad_x > 0 and pad_y == 0


# ------------------------------------------------------- with real models


@needs_runtime
@needs_weights
def test_the_detector_finds_characters_in_real_footage():
    from app.video.frames import extract_frames
    from app.video.loader import load_video

    # Sampled at 10s over the first few minutes rather than a handful of
    # widely spaced frames: this episode opens on several minutes of
    # vehicles and explosions, so a sparse sample can legitimately contain
    # no characters at all and would make this test flap.
    detector = anime.AnimeFaceDetector()
    found = 0
    with load_video(VIDEO) as container:
        for index, (_, frame) in enumerate(extract_frames(container, sample_interval=10.0)):
            found += len(detector.detect(frame))
            if index >= 24:
                break
    detector.close()
    assert found > 0, "the animation detector found nothing in animated footage"


@needs_runtime
@needs_weights
def test_the_detector_reports_no_landmarks_rather_than_inventing_them():
    """Alignment built on invented points would be worse than none."""
    from app.video.frames import extract_frames
    from app.video.loader import load_video

    detector = anime.AnimeFaceDetector()
    seen = []
    with load_video(VIDEO) as container:
        for index, (_, frame) in enumerate(extract_frames(container, sample_interval=10.0)):
            seen.extend(detector.detect(frame))
            if seen or index >= 24:
                break
    detector.close()
    assert seen, "expected at least one detection"
    assert all(d.landmarks is None for d in seen)


@needs_runtime
@needs_weights
def test_embeddings_are_unit_length_and_stamped():
    embedder = anime.AnimeFaceEmbedder()
    frame = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    detection = FaceDetection(
        box=BoundingBox(x_min=100, y_min=100, x_max=260, y_max=260), confidence=0.8
    )
    (result,) = embedder.embed_batch(frame, [detection])
    embedder.close()

    assert result is not None
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)
    assert result.embedding_space == anime.ANIME_EMBEDDING_SPACE


@needs_runtime
@needs_weights
def test_an_unusable_crop_yields_none_in_its_own_slot():
    """Same positional contract as the live-action embedder."""
    embedder = anime.AnimeFaceEmbedder()
    frame = np.zeros((100, 100, 3), np.uint8)
    offscreen = FaceDetection(
        box=BoundingBox(x_min=99, y_min=99, x_max=100, y_max=100), confidence=0.5
    )
    good = FaceDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=90, y_max=90), confidence=0.9
    )
    results = embedder.embed_batch(frame, [offscreen, good])
    embedder.close()

    assert len(results) == 2
    assert results[1] is not None
