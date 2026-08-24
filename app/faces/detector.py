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


def intersection_over_union(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Overlap ratio between two boxes, in [0.0, 1.0].

    Lives next to BoundingBox rather than in any one consumer because both
    the gallery's duplicate-suppression and the tracker's frame-to-frame
    linking need it.
    """
    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)

    intersection_width = max(0, x_max - x_min)
    intersection_height = max(0, y_max - y_min)
    intersection_area = intersection_width * intersection_height

    area_a = (box_a.x_max - box_a.x_min) * (box_a.y_max - box_a.y_min)
    area_b = (box_b.x_max - box_b.x_min) * (box_b.y_max - box_b.y_min)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


@dataclass(frozen=True)
class FaceLandmarks:
    """Five facial landmark points in absolute pixel coordinates.

    Order matches the YuNet output: right eye, left eye, nose tip,
    right mouth corner, left mouth corner.
    """

    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]

    def as_tuple(self) -> tuple[float, ...]:
        """Flatten to (x, y) * 5, the layout cv2.FaceRecognizerSF expects."""
        return (
            *self.right_eye,
            *self.left_eye,
            *self.nose_tip,
            *self.mouth_right,
            *self.mouth_left,
        )


@dataclass(frozen=True)
class FaceDetection:
    """Contains the data for a single detected face."""

    box: BoundingBox
    confidence: float
    landmarks: FaceLandmarks | None = None


class FaceDetector:
    """A lightweight face detector backed by OpenCV YuNet on CPU."""

    DEFAULT_MODEL_FILENAME = "face_detection_yunet_2026may.onnx"
    DEFAULT_MODEL_SOURCE = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2026may.onnx"
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        """
        Initializes the detector.

        Args:
            model_path: Path to the YuNet ONNX model. If omitted, uses the
                repository-local model in assets/models.
            input_size: Detection resolution as (width, height). The detector
                resizes frames with letterboxing to this size before inference.
            confidence_threshold: Minimum confidence threshold used to accept
                a detection.
            nms_threshold: Non-max suppression threshold.
            top_k: Maximum number of candidate boxes to keep before NMS.
        """
        if not hasattr(cv2, "FaceDetectorYN"):
            raise ImportError(
                "This OpenCV build does not expose cv2.FaceDetectorYN. "
                "Install a full OpenCV package with DNN support."
            )

        self.input_size = self._normalize_input_size(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.model_path = self._resolve_model_path(model_path)
        self._detector = self._create_detector()

    @staticmethod
    def _normalize_input_size(input_size: tuple[int, int]) -> tuple[int, int]:
        width, height = input_size
        if width <= 0 or height <= 0:
            raise ValueError("input_size must contain positive width and height")
        return int(width), int(height)

    @classmethod
    def _default_model_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "assets" / "models" / cls.DEFAULT_MODEL_FILENAME

    @classmethod
    def _resolve_model_path(cls, model_path: str | Path | None) -> Path:
        resolved_path = Path(model_path) if model_path is not None else cls._default_model_path()
        if resolved_path.is_file():
            return resolved_path

        raise FileNotFoundError(
            f"YuNet model not found at '{resolved_path}'. Download the official model from {cls.DEFAULT_MODEL_SOURCE}."
        )

    def _create_detector(self):
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU

        try:
            return cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=self.input_size,
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
                backend_id=backend_id,
                target_id=target_id,
            )
        except cv2.error as exc:
            raise ImportError(
                "OpenCV could not initialize the YuNet face detector. "
                "Make sure the installed cv2 package includes DNN support."
            ) from exc

    @staticmethod
    def _letterbox(image: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
        """Resize an image to fit inside target_size while preserving aspect ratio."""
        target_width, target_height = target_size
        image_height, image_width = image.shape[:2]

        scale = min(target_width / image_width, target_height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))

        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_height, target_width, 3), dtype=image.dtype)

        pad_x = (target_width - resized_width) // 2
        pad_y = (target_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _map_point(x: float, y: float, scale: float, pad_x: int, pad_y: int) -> tuple[float, float]:
        return ((x - pad_x) / scale, (y - pad_y) / scale)

    @classmethod
    def _map_landmarks(
        cls, face: np.ndarray, scale: float, pad_x: int, pad_y: int
    ) -> FaceLandmarks:
        points = [cls._map_point(face[i], face[i + 1], scale, pad_x, pad_y) for i in range(4, 14, 2)]
        return FaceLandmarks(
            right_eye=points[0],
            left_eye=points[1],
            nose_tip=points[2],
            mouth_right=points[3],
            mouth_left=points[4],
        )

    @classmethod
    def _map_detection(
        cls,
        face: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        frame_width: int,
        frame_height: int,
    ) -> FaceDetection | None:
        x_min, y_min, width, height = face[:4]
        confidence = float(face[14])

        mapped_x_min = int(round((x_min - pad_x) / scale))
        mapped_y_min = int(round((y_min - pad_y) / scale))
        mapped_x_max = int(round((x_min + width - pad_x) / scale))
        mapped_y_max = int(round((y_min + height - pad_y) / scale))

        x_min_clamped = max(0, min(frame_width, mapped_x_min))
        y_min_clamped = max(0, min(frame_height, mapped_y_min))
        x_max_clamped = max(0, min(frame_width, mapped_x_max))
        y_max_clamped = max(0, min(frame_height, mapped_y_max))

        if x_min_clamped >= x_max_clamped or y_min_clamped >= y_max_clamped:
            return None

        return FaceDetection(
            box=BoundingBox(
                x_min=x_min_clamped,
                y_min=y_min_clamped,
                x_max=x_max_clamped,
                y_max=y_max_clamped,
            ),
            confidence=confidence,
            landmarks=cls._map_landmarks(face, scale, pad_x, pad_y),
        )

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detects faces in a single RGB image.

        Args:
            image: An RGB image as a NumPy array.

        Returns:
            A list of FaceDetection objects for each face found.
        """
        if image is None or image.ndim != 3 or image.shape[2] < 3:
            return []

        frame_height, frame_width = image.shape[:2]
        # YuNet is an OpenCV DNN model and was trained on BGR input. Frames
        # arrive here as RGB (extract_frames decodes rgb24), so the channels
        # must be swapped before inference. Feeding RGB straight through still
        # finds most faces -- detection is largely structural -- but it costs
        # recall and, more importantly, landmark precision, which the
        # embedder's ArcFace alignment depends on for a usable identity
        # embedding.
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detection_image, scale, pad_x, pad_y = self._letterbox(bgr_image, self.input_size)
        self._detector.setInputSize(self.input_size)

        _, faces = self._detector.detect(detection_image)
        if faces is None:
            return []

        detections = []
        for face in faces:
            confidence = float(face[14])
            if confidence < self.confidence_threshold:
                continue

            detection = self._map_detection(face, scale, pad_x, pad_y, frame_width, frame_height)
            if detection is not None:
                detections.append(detection)

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    def close(self):
        """Cleans up the detector resources."""
        self._detector = None
        return None