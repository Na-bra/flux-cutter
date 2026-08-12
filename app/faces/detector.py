from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Represents a bounding box with absolute pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class FaceDetection:
    """Contains the data for a single detected face."""

    box: BoundingBox
    confidence: float


class FaceDetector:
    """A lightweight face detector backed by OpenCV Haar cascades."""

    @staticmethod
    def _resolve_cascade_path() -> Path:
        """Find the Haar cascade XML in the environment or repository."""
        candidates = []

        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / "assets" / "models" / "haarcascade_frontalface_default.xml")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "OpenCV face cascade not found. Checked: " + searched + "."
        )

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initializes the detector.

        Args:
            min_detection_confidence: Minimum confidence threshold used to accept a
                detection. OpenCV Haar cascades do not return a confidence score,
                so the score is approximated from detection size and filtered
                against this threshold.
        """
        self.min_detection_confidence = float(min_detection_confidence)

        cascade_path = self._resolve_cascade_path()
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise FileNotFoundError(f"Failed to load OpenCV face cascade from {cascade_path}.")

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detects faces in a single RGB image.

        Args:
            image: An RGB image as a NumPy array.

        Returns:
            A list of FaceDetection objects for each face found.
        """
        if image is None:
            return []

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        height, width = gray.shape[:2]
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        detections = []
        for x, y, w, h in faces:
            area = w * h
            frame_area = max(width * height, 1)
            confidence = min(0.99, 0.5 + (area / frame_area) * 10.0)
            confidence = max(0.0, min(0.99, confidence))

            if confidence < self.min_detection_confidence:
                continue

            x_min = max(0, int(x))
            y_min = max(0, int(y))
            x_max = min(width, int(x + w))
            y_max = min(height, int(y + h))

            detections.append(
                FaceDetection(
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    confidence=float(confidence),
                )
            )

        return detections

    def close(self):
        """Cleans up the detector resources."""
        self._cascade = None