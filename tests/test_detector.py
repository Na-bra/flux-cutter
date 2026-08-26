from pathlib import Path

import cv2
import numpy as np
import pytest

from app.faces.detector import FaceDetector
from app.video.frames import extract_frames
from app.video.loader import load_video

VIDEO_PATH = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test.mp4"
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "face_detection_yunet_2026may.onnx"


@pytest.fixture(scope="module")
def detector():
    """Fixture to create a single detector instance for all tests."""
    if not MODEL_PATH.is_file():
        pytest.skip(
            f"YuNet model not found at '{MODEL_PATH}'. Download it from the OpenCV Zoo before running detector tests."
        )

    detector_instance = FaceDetector(model_path=MODEL_PATH, input_size=(640, 640), confidence_threshold=0.6)
    try:
        yield detector_instance
    finally:
        detector_instance.close()


@pytest.fixture(scope="module")
def wide_detector():
    """Fixture for a wider input size used in multiple-face validation."""
    if not MODEL_PATH.is_file():
        pytest.skip(
            f"YuNet model not found at '{MODEL_PATH}'. Download it from the OpenCV Zoo before running detector tests."
        )

    detector_instance = FaceDetector(model_path=MODEL_PATH, input_size=(1280, 1280), confidence_threshold=0.5)
    try:
        yield detector_instance
    finally:
        detector_instance.close()


def _bbox_is_valid(image, detection):
    height, width = image.shape[:2]
    assert 0 <= detection.box.x_min < detection.box.x_max <= width
    assert 0 <= detection.box.y_min < detection.box.y_max <= height
    assert 0.0 <= detection.confidence <= 1.0


def _find_first_detection(detector_instance, sample_interval=1.0):
    with load_video(VIDEO_PATH) as container:
        for timestamp, frame_image in extract_frames(container, sample_interval=sample_interval):
            detections = detector_instance.detect(frame_image)
            if detections:
                return timestamp, frame_image, detections

    raise AssertionError("No face detected in the sample video.")


def test_detector_finds_face_in_video_frame(detector):
    """Tests that the detector finds a known face in the test video."""
    _, found_frame, found_detections = _find_first_detection(detector)

    assert len(found_detections) > 0
    face = found_detections[0]
    assert face.confidence >= 0.6
    _bbox_is_valid(found_frame, face)


def test_detector_finds_no_face_in_blank_frame(detector):
    """Tests that the detector finds no faces in a blank image."""
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_image)
    assert len(detections) == 0


def test_detector_rejects_invalid_input(detector):
    """Tests that invalid inputs return no detections."""
    assert detector.detect(None) == []
    assert detector.detect(np.zeros((480, 640), dtype=np.uint8)) == []


def test_detector_handles_high_confidence_threshold():
    """Tests that a very high confidence threshold filters out detections."""
    if not MODEL_PATH.is_file():
        pytest.skip(
            f"YuNet model not found at '{MODEL_PATH}'. Download it from the OpenCV Zoo before running detector tests."
        )

    high_conf_detector = FaceDetector(model_path=MODEL_PATH, input_size=(640, 640), confidence_threshold=0.99)
    try:
        with load_video(VIDEO_PATH) as container:
            _, image = next(iter(extract_frames(container, sample_interval=1.0)))

        detections = high_conf_detector.detect(image)
        assert len(detections) == 0
    finally:
        high_conf_detector.close()


def test_detector_detects_multiple_faces_from_real_frames(wide_detector):
    """Tests that two real face crops can be detected together on one canvas."""
    first_face_image = None
    first_face_detection = None

    with load_video(VIDEO_PATH) as container:
        for _, frame_image in extract_frames(container, sample_interval=1.0):
            detections = wide_detector.detect(frame_image)
            if detections:
                first_face_image = frame_image
                first_face_detection = detections[0]
                break

    assert first_face_image is not None and first_face_detection is not None

    face_box = first_face_detection.box
    face_crop = first_face_image[face_box.y_min:face_box.y_max, face_box.x_min:face_box.x_max]
    face_crop = np.ascontiguousarray(face_crop)

    target_height = 720
    scale = target_height / face_crop.shape[0]
    resized_width = int(round(face_crop.shape[1] * scale))
    resized_face = cv2.resize(face_crop, (resized_width, target_height))

    canvas_height = 1280
    canvas_width = 1280
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    first_x = 120
    second_x = canvas_width - resized_width - 120
    y_offset = 200

    canvas[y_offset:y_offset + target_height, first_x:first_x + resized_width] = resized_face
    canvas[y_offset:y_offset + target_height, second_x:second_x + resized_width] = resized_face

    detections = wide_detector.detect(canvas)
    assert len(detections) >= 2
    for detection in detections[:2]:
        _bbox_is_valid(canvas, detection)