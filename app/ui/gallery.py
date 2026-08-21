from dataclasses import dataclass, field
from math import ceil

import cv2
import numpy as np

from app.faces.detector import BoundingBox, FaceDetection, intersection_over_union
from app.faces.grouper import FaceIdentityGroup


DEFAULT_PADDING_RATIO = 0.08
DEFAULT_THUMBNAIL_SIZE = (192, 192)
DEFAULT_MAX_ITEMS = 24
DEFAULT_MIN_TIMESTAMP_GAP_SECONDS = 1.0
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.6


@dataclass(frozen=True)
class GalleryItem:
    """Represents a single face thumbnail in the gallery."""

    thumbnail: np.ndarray
    source_timestamp: float
    detection_confidence: float
    bounding_box: BoundingBox
    frame_index: int | None = None


@dataclass
class FaceGallery:
    """Simple gallery state for representative face detections."""

    items: list[GalleryItem]
    source_detection_count: int = 0
    candidate_count: int = 0
    selected_index: int | None = None

    def select(self, index: int) -> GalleryItem:
        """Select and return a gallery item by index."""
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Gallery index out of range: {index}")

        self.selected_index = index
        return self.items[index]

    @property
    def selected_item(self) -> GalleryItem | None:
        """Return the currently selected gallery item, if any."""
        if self.selected_index is None:
            return None
        if self.selected_index < 0 or self.selected_index >= len(self.items):
            return None
        return self.items[self.selected_index]


@dataclass(frozen=True)
class _GalleryCandidate:
    """Internal candidate used before representative sampling."""

    face_crop: np.ndarray
    source_timestamp: float
    detection_confidence: float
    bounding_box: BoundingBox
    frame_index: int | None = None


def _validate_rgb_frame(frame: np.ndarray) -> None:
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame must be an RGB image with three channels")


def _clamp_face_box(
    detection: FaceDetection,
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> BoundingBox:
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    box = detection.box
    width = box.x_max - box.x_min
    height = box.y_max - box.y_min
    if width <= 0 or height <= 0:
        raise ValueError("FaceDetection box must have positive dimensions")

    pad_x = max(1, int(round(width * padding_ratio)))
    pad_y = max(1, int(round(height * padding_ratio)))

    x_min = max(0, box.x_min - pad_x)
    y_min = max(0, box.y_min - pad_y)
    x_max = min(frame_width, box.x_max + pad_x)
    y_max = min(frame_height, box.y_max + pad_y)

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Padded face box collapsed after clamping")

    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def crop_face(
    frame: np.ndarray,
    detection: FaceDetection,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> np.ndarray:
    """Crop a face from an RGB frame using a small padded bounding box."""
    _validate_rgb_frame(frame)

    frame_height, frame_width = frame.shape[:2]
    padded_box = _clamp_face_box(detection, frame_width, frame_height, padding_ratio)
    return frame[
        padded_box.y_min : padded_box.y_max,
        padded_box.x_min : padded_box.x_max,
    ].copy()


def generate_thumbnail(
    face_crop: np.ndarray,
    thumbnail_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> np.ndarray:
    """Generate a fixed-size thumbnail while preserving aspect ratio."""
    _validate_rgb_frame(face_crop)

    target_width, target_height = thumbnail_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("thumbnail_size must contain positive width and height")

    crop_height, crop_width = face_crop.shape[:2]
    scale = min(target_width / crop_width, target_height / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        face_crop,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros((target_height, target_width, 3), dtype=face_crop.dtype)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


def _select_representative_candidates(
    candidates: list[_GalleryCandidate],
    max_items: int,
    minimum_timestamp_gap_seconds: float,
    duplicate_iou_threshold: float,
) -> list[_GalleryCandidate]:
    selected: list[_GalleryCandidate] = []

    for candidate in sorted(
        candidates,
        key=lambda entry: (-entry.detection_confidence, entry.source_timestamp),
    ):
        is_duplicate = False
        for chosen in selected:
            if abs(candidate.source_timestamp - chosen.source_timestamp) > minimum_timestamp_gap_seconds:
                continue
            if intersection_over_union(candidate.bounding_box, chosen.bounding_box) >= duplicate_iou_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        selected.append(candidate)
        if len(selected) >= max_items:
            break

    return selected


def build_face_gallery(
    frame_detections: list[tuple[float, np.ndarray, list[FaceDetection]]],
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    thumbnail_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
    max_items: int = DEFAULT_MAX_ITEMS,
    minimum_timestamp_gap_seconds: float = DEFAULT_MIN_TIMESTAMP_GAP_SECONDS,
    duplicate_iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD,
) -> FaceGallery:
    """Build a representative face gallery from sampled frames and detections."""
    candidates: list[_GalleryCandidate] = []
    source_detection_count = 0

    for frame_index, (timestamp, frame, detections) in enumerate(frame_detections):
        source_detection_count += len(detections)
        for detection in detections:
            face_crop = crop_face(frame, detection, padding_ratio=padding_ratio)
            if face_crop.size == 0:
                continue

            candidates.append(
                _GalleryCandidate(
                    face_crop=face_crop,
                    source_timestamp=float(timestamp),
                    detection_confidence=float(detection.confidence),
                    bounding_box=detection.box,
                    frame_index=frame_index,
                )
            )

    selected_candidates = _select_representative_candidates(
        candidates,
        max_items=max_items,
        minimum_timestamp_gap_seconds=minimum_timestamp_gap_seconds,
        duplicate_iou_threshold=duplicate_iou_threshold,
    )

    items = [
        GalleryItem(
            thumbnail=generate_thumbnail(candidate.face_crop, thumbnail_size=thumbnail_size),
            source_timestamp=candidate.source_timestamp,
            detection_confidence=candidate.detection_confidence,
            bounding_box=candidate.bounding_box,
            frame_index=candidate.frame_index,
        )
        for candidate in selected_candidates
    ]

    return FaceGallery(
        items=items,
        source_detection_count=source_detection_count,
        candidate_count=len(candidates),
    )


def render_gallery_montage(
    items: list[GalleryItem],
    columns: int = 4,
    tile_padding: int = 12,
    label_height: int = 56,
    background_color: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """Render the gallery items into a simple image grid."""
    if columns <= 0:
        raise ValueError("columns must be greater than 0")

    tile_width, tile_height = DEFAULT_THUMBNAIL_SIZE
    cell_width = tile_width + tile_padding * 2
    cell_height = tile_height + label_height + tile_padding * 2

    if not items:
        return np.full((cell_height, cell_width, 3), background_color, dtype=np.uint8)

    rows = ceil(len(items) / columns)
    montage = np.full(
        (rows * cell_height, columns * cell_width, 3),
        background_color,
        dtype=np.uint8,
    )

    for index, item in enumerate(items):
        row = index // columns
        column = index % columns
        top = row * cell_height
        left = column * cell_width

        tile = montage[top : top + cell_height, left : left + cell_width]
        thumbnail_top = tile_padding
        thumbnail_left = tile_padding
        tile[
            thumbnail_top : thumbnail_top + tile_height,
            thumbnail_left : thumbnail_left + tile_width,
        ] = item.thumbnail

        cv2.rectangle(
            tile,
            (thumbnail_left, thumbnail_top),
            (thumbnail_left + tile_width - 1, thumbnail_top + tile_height - 1),
            (80, 80, 80),
            1,
        )

        timestamp_text = f"#{index + 1}  t={item.source_timestamp:.2f}s"
        confidence_text = f"conf={item.detection_confidence:.2f}"
        box_text = (
            f"[{item.bounding_box.x_min}, {item.bounding_box.y_min}]"
            f"-{item.bounding_box.x_max}, {item.bounding_box.y_max}"
        )

        text_x = tile_padding
        text_y = tile_padding + tile_height + 18
        cv2.putText(tile, timestamp_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        cv2.putText(tile, confidence_text, (text_x, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
        cv2.putText(tile, box_text, (text_x, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return montage


def save_gallery_montage(
    items: list[GalleryItem],
    output_path,
    columns: int = 4,
) -> np.ndarray:
    """Render the gallery montage and save it to disk."""
    montage = render_gallery_montage(items, columns=columns)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return montage


def format_selected_item(item: GalleryItem, index: int | None = None) -> str:
    """Format a selected gallery item for console output."""
    prefix = f"Selected gallery item #{index + 1}" if index is not None else "Selected gallery item"
    return (
        f"{prefix}:\n"
        f"  timestamp = {item.source_timestamp:.2f}\n"
        f"  confidence = {item.detection_confidence:.2f}\n"
        f"  box = ({item.bounding_box.x_min}, {item.bounding_box.y_min}, "
        f"{item.bounding_box.x_max}, {item.bounding_box.y_max})"
    )


@dataclass(frozen=True)
class PersonCard:
    """Represents one identity group as a single gallery entry."""

    group_id: int
    representative_thumbnail: np.ndarray
    detection_count: int
    best_confidence: float
    first_seen_timestamp: float
    last_seen_timestamp: float


@dataclass
class IdentityGallery:
    """Gallery state for identity groups, one card per detected person."""

    cards: list[PersonCard]
    # Parallel to `cards` (same order, same length) so a card index chosen
    # from the displayed gallery (e.g. via --select-index) can be mapped
    # straight back to the FaceIdentityGroup that produced it — needed by
    # later stages (e.g. appearance timestamps) that need the group's
    # underlying detections, not just its display summary.
    groups: list[FaceIdentityGroup] = field(default_factory=list)
    total_observations: int = 0
    unassigned_count: int = 0


def build_identity_gallery(
    groups: list[FaceIdentityGroup],
    unassigned_count: int = 0,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    thumbnail_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> IdentityGallery:
    """Build a person-level gallery from grouped face observations.

    Each group's representative face crop (already selected by the
    grouper for quality) becomes that person's card thumbnail.
    """
    cards = []
    selected_groups = []
    for group in sorted(groups, key=lambda g: g.size, reverse=True):
        representative = group.representative_observation
        if representative is None:
            continue

        timestamps = [observation.source_timestamp for observation in group.observations]
        cards.append(
            PersonCard(
                group_id=group.group_id,
                representative_thumbnail=generate_thumbnail(
                    representative.face_crop, thumbnail_size=thumbnail_size
                ),
                detection_count=group.size,
                best_confidence=max(observation.detection.confidence for observation in group.observations),
                first_seen_timestamp=min(timestamps),
                last_seen_timestamp=max(timestamps),
            )
        )
        selected_groups.append(group)

    total_observations = sum(group.size for group in groups) + unassigned_count
    return IdentityGallery(
        cards=cards,
        groups=selected_groups,
        total_observations=total_observations,
        unassigned_count=unassigned_count,
    )


def render_identity_gallery_montage(
    cards: list[PersonCard],
    columns: int = 4,
    tile_padding: int = 12,
    label_height: int = 56,
    background_color: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """Render identity cards into a simple image grid, one tile per person."""
    if columns <= 0:
        raise ValueError("columns must be greater than 0")

    tile_width, tile_height = DEFAULT_THUMBNAIL_SIZE
    cell_width = tile_width + tile_padding * 2
    cell_height = tile_height + label_height + tile_padding * 2

    if not cards:
        return np.full((cell_height, cell_width, 3), background_color, dtype=np.uint8)

    rows = ceil(len(cards) / columns)
    montage = np.full(
        (rows * cell_height, columns * cell_width, 3),
        background_color,
        dtype=np.uint8,
    )

    for index, card in enumerate(cards):
        row = index // columns
        column = index % columns
        top = row * cell_height
        left = column * cell_width

        tile = montage[top : top + cell_height, left : left + cell_width]
        thumbnail_top = tile_padding
        thumbnail_left = tile_padding
        tile[
            thumbnail_top : thumbnail_top + tile_height,
            thumbnail_left : thumbnail_left + tile_width,
        ] = card.representative_thumbnail

        cv2.rectangle(
            tile,
            (thumbnail_left, thumbnail_top),
            (thumbnail_left + tile_width - 1, thumbnail_top + tile_height - 1),
            (80, 80, 80),
            1,
        )

        title_text = f"Person #{index + 1}  ({card.detection_count} detections)"
        span_text = f"seen {card.first_seen_timestamp:.1f}s - {card.last_seen_timestamp:.1f}s"
        confidence_text = f"best conf={card.best_confidence:.2f}"

        text_x = tile_padding
        text_y = tile_padding + tile_height + 18
        cv2.putText(tile, title_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        cv2.putText(tile, span_text, (text_x, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
        cv2.putText(tile, confidence_text, (text_x, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return montage


def save_identity_gallery_montage(
    cards: list[PersonCard],
    output_path,
    columns: int = 4,
) -> np.ndarray:
    """Render the identity gallery montage and save it to disk."""
    montage = render_identity_gallery_montage(cards, columns=columns)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return montage


def format_person_card(card: PersonCard, index: int | None = None) -> str:
    """Format a selected person card for console output."""
    prefix = f"Selected person #{index + 1}" if index is not None else "Selected person"
    return (
        f"{prefix}:\n"
        f"  group_id = {card.group_id}\n"
        f"  detections = {card.detection_count}\n"
        f"  best_confidence = {card.best_confidence:.2f}\n"
        f"  first_seen = {card.first_seen_timestamp:.2f}s\n"
        f"  last_seen = {card.last_seen_timestamp:.2f}s"
    )