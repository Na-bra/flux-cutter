from pathlib import Path

import cv2
import numpy as np

from app.faces.detector import FaceDetection

EMBEDDING_DIMENSIONS = 512

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

    def __init__(self, model_path: str | Path | None = None):
        """
        Initializes the embedder.

        Args:
            model_path: Path to the ArcFace ONNX model. If omitted, uses the
                repository-local model in assets/models.
        """
        self.model_path = self._resolve_model_path(model_path)
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @classmethod
    def _default_model_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "assets" / "models" / cls.DEFAULT_MODEL_FILENAME

    @classmethod
    def _resolve_model_path(cls, model_path: str | Path | None) -> Path:
        resolved_path = Path(model_path) if model_path is not None else cls._default_model_path()
        if resolved_path.is_file():
            return resolved_path

        raise FileNotFoundError(
            f"ArcFace model not found at '{resolved_path}'. Download the official model from {cls.DEFAULT_MODEL_SOURCE}."
        )

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
        self._net.setInput(blob)
        raw_feature = self._net.forward().flatten()

        norm = float(np.linalg.norm(raw_feature))
        if norm == 0.0:
            raise ValueError("ArcFace produced a degenerate all-zero embedding")

        return (raw_feature / norm).astype(np.float32)

    def close(self):
        """Cleans up the network resources."""
        self._net = None
        return None
