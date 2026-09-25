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
        anime._open_session(Path("unused.onnx"), {})


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


# ----------------------------------------------------------------- Core ML
#
# Both models reach Apple's accelerators through Core ML, with their free
# dimensions pinned so Core ML will take the whole graph (see
# anime._open_session). The CPU stays the fallback, and the two must agree.

coreml_here = pytest.mark.skipif(
    not anime._coreml_is_worth_trying(), reason="Core ML is not available on this machine"
)


def _backend(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FLUXCUTTER_EMBED_BACKEND", raising=False)
    else:
        monkeypatch.setenv("FLUXCUTTER_EMBED_BACKEND", value)


def test_an_unknown_backend_is_refused(monkeypatch):
    _backend(monkeypatch, "gpu-please")
    with pytest.raises(ValueError, match="unknown embedding backend"):
        anime._open_session(Path("unused.onnx"), {})


def test_asking_for_core_ml_where_there_is_none_says_so(monkeypatch):
    """Asked for by name, a missing accelerator is an error rather than a
    silent CPU run someone would then mistake for Core ML's speed."""
    _backend(monkeypatch, "coreml")
    monkeypatch.setattr(anime, "_coreml_is_worth_trying", lambda: False)
    if not anime.onnxruntime_available():
        pytest.skip("onnxruntime is not installed")
    with pytest.raises(RuntimeError, match="Core ML was requested"):
        anime._open_session(Path("unused.onnx"), {})


@needs_runtime
@needs_weights
@pytest.mark.parametrize("value", ["cpu", "opencv"])
def test_the_cpu_can_be_chosen_with_live_mode_s_switch(monkeypatch, value):
    """One setting opts a machine out of Core ML in both modes; live mode's
    other backend is cv2.dnn, which cannot load these graphs, so it means
    the CPU here."""
    _backend(monkeypatch, value)
    embedder = anime.AnimeFaceEmbedder()

    assert embedder._session.get_providers() == ["CPUExecutionProvider"]


@needs_runtime
@needs_weights
@coreml_here
def test_both_models_run_on_core_ml_by_default(monkeypatch):
    _backend(monkeypatch, None)

    assert anime.AnimeFaceDetector()._session.get_providers()[0] == "CoreMLExecutionProvider"
    assert anime.AnimeFaceEmbedder()._session.get_providers()[0] == "CoreMLExecutionProvider"


def _crops_from_footage(count=6):
    """Real drawn faces from the test footage, with their frame."""
    from app.video.frames import extract_frames
    from app.video.loader import load_video

    detector = anime.AnimeFaceDetector()
    with load_video(VIDEO) as container:
        for _, frame in extract_frames(container, sample_interval=5.0):
            found = detector.detect(frame)
            if len(found) >= 1:
                return frame, found[:count]
    pytest.skip("no faces found in the footage")


@needs_runtime
@needs_weights
@coreml_here
@pytest.mark.skipif(not VIDEO.is_file(), reason="no animation footage")
def test_core_ml_and_the_cpu_embed_a_face_the_same(monkeypatch):
    """Measured over 64 real faces: worst cosine agreement 1.0000. Unlike
    live mode's Core ML path, these come out the same vectors."""
    frame, detections = _crops_from_footage()
    _backend(monkeypatch, "cpu")
    on_cpu = anime.AnimeFaceEmbedder().embed_batch(frame, detections)
    _backend(monkeypatch, "coreml")
    on_coreml = anime.AnimeFaceEmbedder().embed_batch(frame, detections)

    for a, b in zip(on_cpu, on_coreml):
        assert float(np.dot(a.embedding, b.embedding)) > 0.9999


@needs_runtime
@needs_weights
@coreml_here
@pytest.mark.skipif(not VIDEO.is_file(), reason="no animation footage")
def test_core_ml_and_the_cpu_find_the_same_faces(monkeypatch):
    from app.video.frames import extract_frames
    from app.video.loader import load_video

    with load_video(VIDEO) as container:
        frames = [frame for _, frame in extract_frames(container, sample_interval=20.0)]
    _backend(monkeypatch, "cpu")
    on_cpu = [anime.AnimeFaceDetector().detect(frame) for frame in frames]
    _backend(monkeypatch, "coreml")
    on_coreml = [anime.AnimeFaceDetector().detect(frame) for frame in frames]

    assert sum(len(found) for found in on_cpu) > 0
    for a, b in zip(on_cpu, on_coreml):
        assert [d.box for d in a] == [d.box for d in b]


@needs_runtime
@needs_weights
def test_several_faces_embed_as_they_would_one_at_a_time():
    """The session takes one face per call now; a frame with several must
    still come back in order, each the vector it would have alone."""
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    boxes = [
        FaceDetection(box=BoundingBox(x_min=20 + 180 * k, y_min=60, x_max=160 + 180 * k, y_max=200), confidence=0.8)
        for k in range(3)
    ]
    embedder = anime.AnimeFaceEmbedder()

    together = embedder.embed_batch(frame, boxes)
    alone = [embedder.embed_batch(frame, [box])[0] for box in boxes]

    for a, b in zip(together, alone):
        assert np.allclose(a.embedding, b.embedding, atol=1e-6)
