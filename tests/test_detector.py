from pathlib import Path

import numpy as np
import pytest

from app.faces.detector import FaceDetector
from app.video.frames import extract_frames
from app.video.loader import load_video

VIDEO_PATH = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test.mp4"


@pytest.fixture(scope="module")
def detector():
    """Fixture to create a single detector instance for all tests."""
    d = FaceDetector(min_detection_confidence=0.3)
    yield d
    d.close()


def test_detector_finds_face_in_video_frame(detector):
    """Tests that the detector finds a known face in the test video."""
    with load_video(VIDEO_PATH) as container:
        frames = extract_frames(container, sample_interval=1.0)

        detections = []
        image = None
        for _, frame in frames:
            detections = detector.detect(frame)
            if detections:
                image = frame
                break

    assert image is not None
    assert len(detections) > 0
    face = detections[0]

    assert face.confidence > 0.3
    assert face.box.x_min > 0
    assert face.box.y_min > 0
    assert face.box.x_max < image.shape[1]
    assert face.box.y_max < image.shape[0]
    assert face.box.x_max > face.box.x_min


def test_detector_finds_no_face_in_blank_frame(detector):
    """Tests that the detector finds no faces in a blank image."""
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_image)
    assert len(detections) == 0


def test_detector_handles_high_confidence_threshold(detector):
    """Tests that a high confidence threshold filters out detections."""
    # Re-initialize with a threshold that is unlikely to be met
    high_conf_detector = FaceDetector(min_detection_confidence=0.99)
    with load_video(VIDEO_PATH) as container:
        _, image = extract_frames(container, sample_interval=30)[0]

    detections = high_conf_detector.detect(image)
    assert len(detections) == 0
    high_conf_detector.close()