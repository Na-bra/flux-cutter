"""Detection and embedding for animated footage.

Live action and animation are two pipelines, not one pipeline with a switch.
The reason is measured rather than assumed: on `animation.mp4` (Ben 10, 7
minutes) the live-action YuNet detector finds 0.26 faces per sampled frame
and most of what it does find is not a character, while the detector here
finds 0.45 and the montages show it catching Ben, Gwen and Grandpa Max
including the large foreground faces (Instructions.md 17).

Both models run under onnxruntime rather than cv2.dnn, which cannot load
either graph -- the detector's ONNX trips cv2's importer on a Concat node.
onnxruntime is an optional dependency: nothing in live-action mode imports
this module, so a user who never selects Animation never needs it, and the
frozen app does not carry it.

Two things here deliberately do NOT match the live-action pipeline:

- **No landmarks.** The detector returns boxes only. Rather than invent five
  points to keep the shape of FaceDetection tidy, `landmarks` stays None and
  the embedder below does not need them -- which is also why this embedder
  cannot be paired with ArcFace, and vice versa.
- **A different embedding space.** CCIP embeddings are not comparable with
  ArcFace's, and not merely because the numbers differ: they are not centred
  the same way. Two *different* characters score a median 0.567 here against
  ArcFace's 0.03 for two different people. Feeding one model's vectors to
  the other's thresholds would put every character in one group.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.debug import onnx_log_severity
from app.faces.detector import BoundingBox, FaceDetection
from app.faces.embedder import (
    BACKEND_OVERRIDE_VARIABLE,
    COREML_BACKEND,
    OPENCV_BACKEND,
    EmbeddedFace,
    _coreml_is_worth_trying,
)
from app.faces.quality import sharpness as crop_sharpness
from app.models import MODELS, ModelDownloadError, ensure_model_cli

# The identifier written onto every embedding this module produces. The
# grouper refuses to compare vectors carrying different ids, which is what
# makes mixing the two pipelines an error rather than a silent wrong answer.
ANIME_EMBEDDING_SPACE = "ccip-caformer-24"

DETECTOR_INPUT_SIZE = 640
EMBEDDER_INPUT_SIZE = 384

# ImageNet statistics, which is what CCIP was trained against.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AnimeModelUnavailable(RuntimeError):
    """Raised when animation mode cannot run on this machine."""


def onnxruntime_available() -> bool:
    """Whether the optional runtime animation mode needs is installed."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


# Core ML, and why the shapes are fixed first.
#
# Both models ran on onnxruntime's CPU provider, and a 23-minute episode took
# 14 minutes to scan: 109 ms a frame to find faces and 249 ms a face to
# embed them. Live mode reaches Apple's accelerators through Core ML
# (app/faces/embedder.py); these two models did not, for a reason that took
# measuring to see. Both declare their batch size -- and the detector its
# height and width -- as free dimensions, and Core ML cannot plan a graph
# whose shapes are unbounded:
#
#                                      embed a face   find faces in a frame
#     CPU (as before)                      249 ms            109 ms
#     Core ML, neural network              849 ms             86 ms
#     Core ML, ML Program                  refused            refused
#     Core ML, ML Program, shapes fixed     67 ms             25 ms
#
# The neural-network format takes the graph in pieces and hands the rest
# back to the CPU, and the embedder came out 3.4x *slower*. ML Program with
# the shapes pinned -- batch 1, and the 640x640 letterbox every frame is
# already fitted into -- runs whole: 3.7x faster to embed, 4.4x to detect.
# The shapes are fixed as the session loads (free dimension overrides), so
# the pinned model files are used as downloaded and nothing is rewritten.
#
# Unlike live mode's, these vectors come out the same: over 64 real faces
# the worst cosine agreement with the CPU was 1.0000, and the detector's
# raw output differed by at most 0.0016. A whole scan of each test video
# gives the same people either way (Instructions.md 47).
#
# The CPU provider stays the fallback, so animation mode still needs no
# accelerator: off macOS, without Core ML, or when onnxruntime quietly
# declines the graph, it runs exactly as it did.

def _open_session(model_path: Path, fixed: dict[str, int]):
    """A session for one model, on Core ML where it can be.

    Args:
        fixed: Free dimensions and the sizes they are always given here.
    """
    # Checked before anything is imported: a mistyped setting is a typo
    # whatever is installed, and saying onnxruntime is missing instead would
    # send someone after the wrong problem.
    requested = os.environ.get(BACKEND_OVERRIDE_VARIABLE) or None
    if requested not in (None, COREML_BACKEND, *CPU_BACKENDS):
        raise ValueError(
            f"unknown embedding backend {requested!r}; expected {COREML_BACKEND!r} "
            f"or one of {', '.join(repr(b) for b in CPU_BACKENDS)}"
        )

    try:
        import onnxruntime
    except ImportError as error:
        raise AnimeModelUnavailable(
            "Animation mode needs the onnxruntime package, which is not "
            "installed.\n    pip install onnxruntime"
        ) from error

    options = onnxruntime.SessionOptions()
    options.log_severity_level = onnx_log_severity()
    for name, size in fixed.items():
        options.add_free_dimension_override_by_name(name, size)

    if requested not in CPU_BACKENDS and _coreml_is_worth_trying():
        try:
            session = onnxruntime.InferenceSession(
                str(model_path),
                options,
                providers=[
                    ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
                    "CPUExecutionProvider",
                ],
            )
            # onnxruntime falls back silently when Core ML declines the
            # graph, so it is checked rather than assumed.
            if "CoreMLExecutionProvider" in session.get_providers():
                return session
        except Exception:
            pass
    if requested == COREML_BACKEND:
        raise RuntimeError("Core ML was requested but is not available here")

    return onnxruntime.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


# Live mode's switch, read the same way here, so one setting opts a machine
# out of Core ML in both modes. Live mode's other backend is cv2.dnn, which
# cannot load these graphs, so for animation "opencv" means the CPU.
CPU_BACKENDS = ("cpu", OPENCV_BACKEND)


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Fits an image into a square without distorting it.

    The same approach the live-action detector uses, for the same reason:
    squashing a 16:9 frame into a square changes every face's proportions.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_width) // 2, (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, pad_x, pad_y


@dataclass(frozen=True)
class AnimeDetectorSettings:
    """Detection knobs for animated footage, separate from the live ones.

    Separate because the numbers are not transferable: this detector scores
    on a different scale from YuNet, and a threshold tuned on one says
    nothing about the other. Changing these cannot affect live-action runs.
    """

    confidence_threshold: float = 0.30
    nms_threshold: float = 0.45
    min_face_size: int = 24


class AnimeFaceDetector:
    """Finds drawn faces, returning the same FaceDetection the rest of the app uses."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        settings: AnimeDetectorSettings | None = None,
    ):
        self.settings = settings or AnimeDetectorSettings()
        self.model_path = _resolve(model_path, "anime_detector")
        self._session = _open_session(
            self.model_path,
            {"batch": 1, "height": DETECTOR_INPUT_SIZE, "width": DETECTOR_INPUT_SIZE},
        )
        self._input = self._session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detects drawn faces in one RGB frame.

        Returns:
            FaceDetections with `landmarks` left as None -- this detector
            does not produce them, and inventing them would corrupt any
            alignment built on top.
        """
        if image is None or image.ndim != 3 or image.shape[2] < 3:
            return []

        frame_height, frame_width = image.shape[:2]
        canvas, scale, pad_x, pad_y = _letterbox(image, DETECTOR_INPUT_SIZE)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self._session.run(None, {self._input: blob})[0]

        # (1, 4 + classes, anchors) -> (anchors, 4 + classes)
        predictions = raw[0].T
        scores = predictions[:, 4:].max(axis=1)
        keep = scores >= self.settings.confidence_threshold
        predictions, scores = predictions[keep], scores[keep]
        if not len(predictions):
            return []

        boxes = []
        for (centre_x, centre_y, width, height), score in zip(predictions[:, :4], scores):
            x_min = (centre_x - width / 2 - pad_x) / scale
            y_min = (centre_y - height / 2 - pad_y) / scale
            x_max = (centre_x + width / 2 - pad_x) / scale
            y_max = (centre_y + height / 2 - pad_y) / scale
            boxes.append([x_min, y_min, x_max, y_max, float(score)])

        candidates = np.array(boxes)
        chosen = cv2.dnn.NMSBoxes(
            candidates[:, :4].tolist(),
            candidates[:, 4].tolist(),
            self.settings.confidence_threshold,
            self.settings.nms_threshold,
        )
        if len(chosen) == 0:
            return []

        detections = []
        for index in np.array(chosen).flatten():
            x_min, y_min, x_max, y_max, score = candidates[index]
            box = BoundingBox(
                x_min=max(0, min(frame_width, int(round(x_min)))),
                y_min=max(0, min(frame_height, int(round(y_min)))),
                x_max=max(0, min(frame_width, int(round(x_max)))),
                y_max=max(0, min(frame_height, int(round(y_max)))),
            )
            if box.x_max - box.x_min < self.settings.min_face_size:
                continue
            if box.y_max - box.y_min < self.settings.min_face_size:
                continue
            detections.append(
                FaceDetection(box=box, confidence=float(score), landmarks=None)
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    def close(self) -> None:
        """Releases the session, which is most of the model's memory."""
        self._session = None


@dataclass(frozen=True)
class AnimeEmbedderSettings:
    """Embedding knobs for animated footage."""

    # How much context around the detected face to include. CCIP is trained
    # on character images rather than tight face crops, so a little of the
    # hair and costume around the face is signal, not noise -- on this
    # footage hair colour is often what separates two characters.
    crop_padding_ratio: float = 0.35


class AnimeFaceEmbedder:
    """Embeds a drawn character, returning the same EmbeddedFace the app uses."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        settings: AnimeEmbedderSettings | None = None,
    ):
        self.settings = settings or AnimeEmbedderSettings()
        self.model_path = _resolve(model_path, "anime_embedder")
        self._session = _open_session(self.model_path, {"batch": 1})
        self._input = self._session.get_inputs()[0].name

    def crop(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray | None:
        """The padded character crop this model expects, or None if unusable."""
        box = detection.box
        height, width = frame.shape[:2]
        padding = self.settings.crop_padding_ratio * max(
            box.x_max - box.x_min, box.y_max - box.y_min
        )
        x_min = max(0, int(box.x_min - padding))
        y_min = max(0, int(box.y_min - padding))
        x_max = min(width, int(box.x_max + padding))
        y_max = min(height, int(box.y_max + padding))
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        return crop

    def embed_batch(
        self, frame: np.ndarray, detections: list[FaceDetection]
    ) -> list[EmbeddedFace | None]:
        """
        Embeds several characters from one frame.

        Returns:
            One entry per detection, in order, None where the crop was
            unusable -- the same contract as the live-action embedder, so
            callers do not branch on which pipeline they are in.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be an RGB image with three channels")
        if not detections:
            return []

        results: list[EmbeddedFace | None] = [None] * len(detections)
        crops, positions = [], []
        for position, detection in enumerate(detections):
            crop = self.crop(frame, detection)
            if crop is not None:
                crops.append(crop)
                positions.append(position)
        if not crops:
            return results

        batch = np.stack(
            [
                (
                    cv2.resize(
                        crop, (EMBEDDER_INPUT_SIZE, EMBEDDER_INPUT_SIZE),
                        interpolation=cv2.INTER_AREA,
                    ).astype(np.float32)
                    / 255.0
                    - _IMAGENET_MEAN
                )
                / _IMAGENET_STD
                for crop in crops
            ]
        ).transpose(0, 3, 1, 2)

        # One face at a time: the session's batch is fixed at 1 (see
        # _open_session), and on the CPU four at once was no faster anyway
        # (266 ms a face against 249).
        vectors = np.concatenate(
            [self._session.run(None, {self._input: batch[i : i + 1]})[0] for i in range(len(batch))]
        )
        for row, position in enumerate(positions):
            vector = vectors[row].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue
            results[position] = EmbeddedFace(
                embedding=vector / norm,
                sharpness=crop_sharpness(crops[row]),
                embedding_space=ANIME_EMBEDDING_SPACE,
            )
        return results

    def close(self) -> None:
        self._session = None


def _resolve(model_path: str | Path | None, key: str) -> Path:
    """Where the weights are, fetching them on first use like every other model."""
    if model_path is not None:
        resolved = Path(model_path)
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(f"{key} model not found at '{resolved}'.")
    try:
        return ensure_model_cli(MODELS[key])
    except ModelDownloadError as error:
        raise FileNotFoundError(str(error)) from error
