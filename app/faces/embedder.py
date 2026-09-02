import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.models import MODELS, ModelDownloadError, ensure_model_cli

from app.faces.detector import FaceDetection
from app.faces.quality import sharpness as crop_sharpness


# Which library runs the forward pass. The weights, the preprocessing and
# the output are the same either way -- this only decides who multiplies
# the matrices.
#
# On Apple silicon the CoreML execution provider runs this model roughly
# 4-5x faster than cv2.dnn (5.8 ms against 25-32 ms per face, measured over
# 3111 real faces), because it reaches the Neural Engine. It is not simply
# better everywhere: onnxruntime's own CPU provider measured 109 ms per
# face on the same machine, so the win is CoreML specifically, not
# onnxruntime, and cv2.dnn stays the default off macOS.
#
# The vectors are not bit-identical -- CoreML computes in lower precision
# internally -- so this was checked rather than assumed. Against cv2.dnn
# over those 3111 faces the median agreement is 0.99974 cosine and the
# worst is 0.9843, and identity separation is unchanged on real footage:
# AUC 0.99883 against 0.99881, false merges at 95% recall 0.48% against
# 0.51%. Same embedding space, so galleries built under one backend stay
# comparable under the other.
COREML_BACKEND = "coreml"
OPENCV_BACKEND = "opencv"

# Set FLUXCUTTER_EMBED_BACKEND to force one, which is how the two are
# compared and how a machine with a broken CoreML stack can opt out.
BACKEND_OVERRIDE_VARIABLE = "FLUXCUTTER_EMBED_BACKEND"

# How many faces CoreML is handed at once. See _forward_coreml: the value
# matters far more than it looks, because the shape must never change.
COREML_BATCH = 2


def _coreml_is_worth_trying() -> bool:
    """Whether to attempt CoreML at all, before paying to load anything."""
    if sys.platform != "darwin":
        return False
    try:
        import onnxruntime
    except ImportError:
        return False
    return "CoreMLExecutionProvider" in onnxruntime.get_available_providers()


def _open_coreml_session(model_path: Path):
    """A CoreML-backed session, or None if this machine cannot give one.

    Returns None rather than raising: an unavailable accelerator is a
    reason to use the other backend, not a reason to fail a scan. It also
    verifies that CoreML actually took the graph, because onnxruntime
    silently falls back to its own CPU provider -- which is *slower* here
    than the cv2.dnn path this would be replacing.
    """
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception:
        return None
    if "CoreMLExecutionProvider" not in session.get_providers():
        return None
    return session


EMBEDDING_DIMENSIONS = 512

# The identifier stamped on every vector this module produces.
LIVE_EMBEDDING_SPACE = "arcface-w600k-r50"


@dataclass(frozen=True)
class EmbeddedFace:
    """One face's identity vector, with how usable the crop it came from was.

    Sharpness rides along with the embedding because this is the only place
    the aligned crop exists. Measuring it anywhere else would mean either
    warping the face a second time or measuring the wrong image -- the raw
    box crop, whose Laplacian variance depends on how big the face happened
    to be in frame (app/faces/quality.py).
    """

    embedding: np.ndarray
    sharpness: float
    # Which model's space this vector lives in. Two embeddings are only
    # comparable when these match: cosine similarity between an ArcFace
    # vector and a CCIP one is a number, and it means nothing.
    embedding_space: str = "arcface-w600k-r50"


# Canonical ArcFace 5-point reference template, in pixel coordinates on the
# 112x112 aligned crop. Point order is the same order YuNet emits its
# landmarks in (right eye, left eye, nose tip, right mouth corner, left
# mouth corner), which is also the order OpenCV's own SFace alignCrop
# consumed them in -- so the mapping below is the one the pipeline was
# already relying on before ArcFace replaced SFace.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def _similarity_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping source onto destination.

    Returns the 2x3 matrix cv2.warpAffine expects.

    This is deliberately a closed-form fit rather than
    cv2.estimateAffinePartial2D: that estimator samples (RANSAC/LMEDS) and
    so can return a slightly different matrix for identical input, which
    would make the same face embed to two different vectors between runs.
    Identity grouping compares embeddings by cosine similarity, so that
    jitter is not acceptable here.

    Restricting the fit to rotation + uniform scale + translation (rather
    than a full affine) is what keeps the face's aspect ratio intact;
    letting it shear or stretch independently per axis would distort the
    very geometry the embedding is meant to describe.
    """
    point_count = source.shape[0]
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean

    covariance = destination_centered.T @ source_centered / point_count
    unitary_u, singular_values, unitary_vt = np.linalg.svd(covariance)

    # Guard against the SVD returning a reflection instead of a rotation.
    reflection_fix = np.ones(2)
    if np.linalg.det(unitary_u) * np.linalg.det(unitary_vt) < 0:
        reflection_fix[-1] = -1.0

    rotation = unitary_u @ np.diag(reflection_fix) @ unitary_vt
    source_variance = (source_centered**2).sum() / point_count
    if source_variance == 0.0:
        raise ValueError("degenerate landmarks: all five points are identical")

    scale = float((singular_values * reflection_fix).sum() / source_variance)
    translation = destination_mean - scale * (rotation @ source_mean)

    return np.hstack([scale * rotation, translation.reshape(2, 1)]).astype(np.float64)


class FaceEmbedder:
    """Extracts identity embeddings from detected faces using ArcFace.

    ArcFace (InsightFace's w600k_r50: ResNet50 trained on WebFace600K)
    replaced OpenCV Zoo's SFace here because identity grouping was losing
    genuine same-person matches on the hard frames -- profile angles,
    motion blur, harsh key light -- where SFace's margin between "same
    person" and "different person" is narrowest. ArcFace's additive-angular
    -margin loss separates those cases substantially better.

    It still runs through cv2.dnn on CPU, so this stays a zero-extra-
    dependency swap: no torch, no onnxruntime, no second framework. The
    two things SFace provided for free had to be reimplemented:
    landmark alignment (cv2.FaceRecognizerSF.alignCrop is SFace-specific)
    and output normalization.

    Embeddings are 512-dimensional and L2-normalized, so cosine similarity
    between two of them is a plain dot product.
    """

    DEFAULT_MODEL_FILENAME = "face_recognition_arcface_w600k_r50.onnx"
    DEFAULT_MODEL_SOURCE = (
        "the 'buffalo_l' bundle at "
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip "
        "(extract w600k_r50.onnx and rename it)"
    )

    INPUT_SIZE = (112, 112)
    # ArcFace expects each channel scaled to roughly [-1, 1] as (x - 127.5) / 127.5.
    INPUT_MEAN = 127.5
    INPUT_STD = 127.5

    def __init__(self, model_path: str | Path | None = None, backend: str | None = None):
        """
        Initializes the embedder.

        Args:
            model_path: Path to the ArcFace ONNX model. If omitted, uses the
                repository-local model in assets/models.
            backend: "coreml" or "opencv" to force one, or None to pick the
                faster one available here. A forced backend that cannot be
                opened raises rather than falling back, because someone who
                names a backend is measuring something.
        """
        self.model_path = self._resolve_model_path(model_path)
        self._session = None
        self._net = None

        requested = backend or os.environ.get(BACKEND_OVERRIDE_VARIABLE) or None
        if requested not in (None, COREML_BACKEND, OPENCV_BACKEND):
            raise ValueError(
                f"unknown embedding backend {requested!r}; "
                f"expected {COREML_BACKEND!r} or {OPENCV_BACKEND!r}"
            )

        if requested == COREML_BACKEND:
            self._session = _open_coreml_session(self.model_path)
            if self._session is None:
                raise RuntimeError("CoreML was requested but is not available here")
        elif requested is None and _coreml_is_worth_trying():
            self._session = _open_coreml_session(self.model_path)

        if self._session is not None:
            self.backend = COREML_BACKEND
            self._input_name = self._session.get_inputs()[0].name
        else:
            self.backend = OPENCV_BACKEND
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        """One forward pass, whichever backend is in use."""
        if self._session is not None:
            return self._forward_coreml(blob)
        self._net.setInput(blob)
        return self._net.forward()

    def _forward_coreml(self, blob: np.ndarray) -> np.ndarray:
        """CoreML, fed a batch of exactly COREML_BATCH faces every time.

        This model declares its output as {1, 512}, so CoreML treats each
        new batch size as a new graph and recompiles. Frames hold 1, 2, 3,
        5... faces, so a naive call per frame recompiles almost every
        frame: measured at 818 ms per face against 8.7 ms when the shape is
        held constant, a 94x difference that made the "faster" backend far
        slower than the one it replaced.

        So the batch shape never varies. Short batches are padded with
        zeroes and the padding is sliced back off; longer ones are split.
        COREML_BATCH is 2 because that measured fastest over the real
        distribution of faces per frame -- 11.6s for the test video against
        13.1s at 1 and 36.9s at 8, where the padding waste outweighs the
        larger batch.
        """
        rows = blob.shape[0]
        outputs = []
        for start in range(0, rows, COREML_BATCH):
            chunk = blob[start : start + COREML_BATCH]
            if chunk.shape[0] < COREML_BATCH:
                padded = np.zeros(
                    (COREML_BATCH,) + chunk.shape[1:], dtype=chunk.dtype
                )
                padded[: chunk.shape[0]] = chunk
                result = self._session.run(None, {self._input_name: padded})[0]
                outputs.append(result[: chunk.shape[0]])
            else:
                outputs.append(self._session.run(None, {self._input_name: chunk})[0])
        return np.concatenate(outputs, axis=0)

    @classmethod
    def _default_model_path(cls) -> Path:
        """Where the model is, fetching it on first use if it is not here.

        The download happens at this point rather than at import: it is the
        moment the model is actually needed, so a run that never touches
        embedding never pays for it. See app/models.py.
        """
        return ensure_model_cli(MODELS["embedder"])

    @classmethod
    def _resolve_model_path(cls, model_path: str | Path | None) -> Path:
        # An explicitly passed path is used as given and never downloaded --
        # someone naming a file means that file, and silently substituting a
        # different one would be worse than failing.
        if model_path is not None:
            resolved_path = Path(model_path)
            if resolved_path.is_file():
                return resolved_path
            raise FileNotFoundError(
                f"ArcFace model not found at '{resolved_path}'."
            )

        try:
            return cls._default_model_path()
        except ModelDownloadError as error:
            raise FileNotFoundError(str(error)) from error

    @staticmethod
    def _landmark_array(detection: FaceDetection) -> np.ndarray:
        """The detection's five landmarks as a 5x2 array, in YuNet's own order."""
        if detection.landmarks is None:
            raise ValueError("FaceDetection has no landmarks; cannot align it for embedding")

        return np.array(detection.landmarks.as_tuple(), dtype=np.float64).reshape(5, 2)

    @classmethod
    def align(cls, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Warps a detected face to the canonical 112x112 ArcFace pose.

        Alignment is driven by YuNet's landmarks rather than the bounding
        box, because ArcFace is trained on crops whose eyes/nose/mouth sit
        at fixed positions; feeding it an unaligned box crop measurably
        degrades the embedding.
        """
        landmarks = cls._landmark_array(detection)
        matrix = _similarity_transform(landmarks, ARCFACE_TEMPLATE_112)
        return cv2.warpAffine(frame, matrix, cls.INPUT_SIZE, flags=cv2.INTER_LINEAR, borderValue=0.0)

    def _normalize(self, raw_feature: np.ndarray) -> np.ndarray:
        """L2-normalizes one raw ArcFace output so cosine similarity is a dot product."""
        norm = float(np.linalg.norm(raw_feature))
        if norm == 0.0:
            raise ValueError("ArcFace produced a degenerate all-zero embedding")
        return (raw_feature / norm).astype(np.float32)

    def embed_batch(
        self, frame: np.ndarray, detections: list[FaceDetection]
    ) -> list["EmbeddedFace | None"]:
        """
        Embeds several faces from one frame in a single forward pass.

        Identical in result to calling embed() per face -- verified bit-exact,
        not merely close -- but markedly cheaper, because the per-call overhead
        of the DNN backend is paid once instead of once per face. Measured at
        roughly 1.34x throughput, plateauing around a batch of 8; this footage
        averages 4.6 faces per frame, which lands in the useful part of that
        curve.

        Batching is per frame rather than across frames on purpose: frames
        arrive from a generator that deliberately holds only one at a time, and
        buffering several to fill a larger batch would trade back the memory
        that streaming was introduced to reclaim.

        Returns:
            One EmbeddedFace per input detection, in the same order. An entry
            is None where that face could not be aligned or embedded (no
            landmarks, or degenerate geometry), so callers can drop individual
            failures without losing the correspondence to their detections.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be an RGB image with three channels")
        if not detections:
            return []

        results: list[EmbeddedFace | None] = [None] * len(detections)
        aligned_faces: list[np.ndarray] = []
        source_positions: list[int] = []

        for position, detection in enumerate(detections):
            try:
                aligned_faces.append(self.align(frame, detection))
                source_positions.append(position)
            except ValueError:
                continue

        if not aligned_faces:
            return results

        blob = cv2.dnn.blobFromImages(
            aligned_faces,
            scalefactor=1.0 / self.INPUT_STD,
            size=self.INPUT_SIZE,
            mean=(self.INPUT_MEAN, self.INPUT_MEAN, self.INPUT_MEAN),
            swapRB=False,
        )
        raw_features = self._forward(blob)

        for row, position in enumerate(source_positions):
            try:
                results[position] = EmbeddedFace(
                    embedding=self._normalize(raw_features[row]),
                    sharpness=crop_sharpness(aligned_faces[row]),
                    embedding_space=LIVE_EMBEDDING_SPACE,
                )
            except ValueError:
                results[position] = None

        return results

    def embed(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Aligns and embeds a detected face directly from its source frame.

        Args:
            frame: The RGB frame the detection came from (not a pre-made crop).
            detection: A FaceDetection produced by FaceDetector, with landmarks.

        Returns:
            A 512-dimensional, L2-normalized identity embedding.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be an RGB image with three channels")

        aligned = self.align(frame, detection)

        # The frame is already RGB and ArcFace wants RGB, so no channel swap.
        blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0 / self.INPUT_STD,
            size=self.INPUT_SIZE,
            mean=(self.INPUT_MEAN, self.INPUT_MEAN, self.INPUT_MEAN),
            swapRB=False,
        )
        return self._normalize(self._forward(blob).flatten())

    def close(self):
        """Cleans up the network resources."""
        self._net = None
        self._session = None
        return None
