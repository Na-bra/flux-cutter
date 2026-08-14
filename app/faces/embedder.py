from pathlib import Path

import cv2
import numpy as np

from app.faces.detector import FaceDetection

EMBEDDING_DIMENSIONS = 128


class FaceEmbedder:
    """Extracts identity embeddings from detected faces using OpenCV SFace.

    SFace (face_recognition_sface_2021dec.onnx) is the OpenCV Zoo
    face-recognition model built as YuNet's counterpart: it consumes the
    same 5-point landmarks YuNet already outputs, runs on CPU through the
    cv2.dnn backend with no dependency beyond opencv-python (already
    required for detection), and ships official cosine/L2 similarity
    thresholds validated on standard face-verification benchmarks. That
    made it the natural choice over pulling in a second framework
    (e.g. dlib/face_recognition, insightface+onnxruntime) for a prototype
    stage that is meant to stay CPU-only and cross-platform.
    """

    DEFAULT_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"
    DEFAULT_MODEL_SOURCE = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    )

    def __init__(self, model_path: str | Path | None = None):
        """
        Initializes the embedder.

        Args:
            model_path: Path to the SFace ONNX model. If omitted, uses the
                repository-local model in assets/models.
        """
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise ImportError(
                "This OpenCV build does not expose cv2.FaceRecognizerSF. "
                "Install a full OpenCV package with DNN support."
            )

        self.model_path = self._resolve_model_path(model_path)
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")

    @classmethod
    def _default_model_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "assets" / "models" / cls.DEFAULT_MODEL_FILENAME

    @classmethod
    def _resolve_model_path(cls, model_path: str | Path | None) -> Path:
        resolved_path = Path(model_path) if model_path is not None else cls._default_model_path()
        if resolved_path.is_file():
            return resolved_path

        raise FileNotFoundError(
            f"SFace model not found at '{resolved_path}'. Download the official model from {cls.DEFAULT_MODEL_SOURCE}."
        )

    @staticmethod
    def _to_yunet_row(detection: FaceDetection) -> np.ndarray:
        """Rebuild the 1x15 detection row cv2.FaceRecognizerSF.alignCrop expects."""
        if detection.landmarks is None:
            raise ValueError("FaceDetection has no landmarks; cannot align it for embedding")

        box = detection.box
        row = [
            float(box.x_min),
            float(box.y_min),
            float(box.x_max - box.x_min),
            float(box.y_max - box.y_min),
            *detection.landmarks.as_tuple(),
            float(detection.confidence),
        ]
        return np.array([row], dtype=np.float32)

    def embed(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Aligns and embeds a detected face directly from its source frame.

        Alignment uses YuNet's own landmarks (via cv2's alignCrop) rather
        than a padded box crop, since SFace's accuracy depends on the
        eyes/nose/mouth being registered to a canonical pose before the
        embedding network runs.

        Args:
            frame: The RGB frame the detection came from (not a pre-made crop).
            detection: A FaceDetection produced by FaceDetector, with landmarks.

        Returns:
            A 128-dimensional, L2-normalized identity embedding.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be an RGB image with three channels")

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_row = self._to_yunet_row(detection)
        aligned = self._recognizer.alignCrop(bgr_frame, face_row)
        raw_feature = self._recognizer.feature(aligned)[0]

        norm = float(np.linalg.norm(raw_feature))
        if norm == 0.0:
            raise ValueError("SFace produced a degenerate all-zero embedding")

        return (raw_feature / norm).astype(np.float32)

    def close(self):
        """Cleans up the recognizer resources."""
        self._recognizer = None
        return None
