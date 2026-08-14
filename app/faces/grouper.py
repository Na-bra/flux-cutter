from dataclasses import dataclass, field

import numpy as np

from app.faces.detector import FaceDetection

# OpenCV Zoo's documented cosine-similarity threshold for SFace is 0.363
# (validated on standard face-verification benchmarks), but running this
# project's own test footage through the pipeline showed that value lets
# through at least one false merge (several different faces sharing an
# open-mouth expression clustered into one 7-detection group that fell
# apart again once the threshold was raised). 0.45 was the lowest value
# that kept that cluster split while still merging genuine same-shot
# repeats a few seconds apart. See the stage-0.2 accuracy notes for the
# full experiment.
DEFAULT_SIMILARITY_THRESHOLD = 0.45
# Required gap between the best and second-best group match before a
# detection is assigned rather than treated as a new/uncertain identity.
DEFAULT_MARGIN_THRESHOLD = 0.05
DEFAULT_MIN_CONFIDENCE = 0.7
# Shorter side of the face box, in pixels, below which an embedding is
# considered too unreliable to trust for identity matching.
DEFAULT_MIN_FACE_SIZE = 40


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings (their dot product)."""
    return float(np.dot(embedding_a, embedding_b))


@dataclass(frozen=True)
class FaceObservation:
    """A single embedded face detection, ready for identity grouping."""

    embedding: np.ndarray
    detection: FaceDetection
    face_crop: np.ndarray
    source_timestamp: float
    frame_index: int | None = None


@dataclass
class FaceIdentityGroup:
    """One identity's accumulated observations and running representation."""

    group_id: int
    observations: list[FaceObservation] = field(default_factory=list)
    representative_embedding: np.ndarray | None = None
    representative_observation: FaceObservation | None = None

    @property
    def size(self) -> int:
        return len(self.observations)


def _is_valid_embedding(embedding) -> bool:
    return (
        isinstance(embedding, np.ndarray)
        and embedding.ndim == 1
        and embedding.size > 0
        and bool(np.all(np.isfinite(embedding)))
    )


def _observation_quality(observation: FaceObservation) -> float:
    """A simple confidence-weighted-by-area score used to pick representative images."""
    box = observation.detection.box
    area = max(0, box.x_max - box.x_min) * max(0, box.y_max - box.y_min)
    return observation.detection.confidence * area


class IdentityGrouper:
    """Groups embedded face observations into per-identity clusters.

    This intentionally is not a clustering framework: each new observation
    is compared against every existing group's running centroid embedding
    and assigned to the closest one only if that match clears both an
    absolute similarity floor and a margin over the next-closest group.
    Otherwise it seeds a new group. A false merge is treated as worse than
    a missed match, so ties and near-ties fall through to a new group
    rather than being forced together.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
    ):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [-1.0, 1.0]")
        if margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")

        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size

        self.groups: list[FaceIdentityGroup] = []
        self.unassigned: list[FaceObservation] = []
        self._next_group_id = 1

    def _is_reliable(self, observation: FaceObservation) -> bool:
        if not _is_valid_embedding(observation.embedding):
            return False
        if observation.detection.confidence < self.min_confidence:
            return False

        box = observation.detection.box
        width = box.x_max - box.x_min
        height = box.y_max - box.y_min
        if min(width, height) < self.min_face_size:
            return False

        return True

    def _best_match(self, embedding: np.ndarray) -> tuple[int | None, float, float]:
        """Returns (best_group_index, best_similarity, second_best_similarity)."""
        if not self.groups:
            return None, -1.0, -1.0

        similarities = [
            cosine_similarity(embedding, group.representative_embedding) for group in self.groups
        ]
        ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        best_index = ranked_indices[0]
        best_similarity = similarities[best_index]
        second_best_similarity = similarities[ranked_indices[1]] if len(ranked_indices) > 1 else -1.0
        return best_index, best_similarity, second_best_similarity

    def _start_group(self, observation: FaceObservation) -> int:
        group = FaceIdentityGroup(
            group_id=self._next_group_id,
            observations=[observation],
            representative_embedding=observation.embedding.copy(),
            representative_observation=observation,
        )
        self.groups.append(group)
        self._next_group_id += 1
        return group.group_id

    def _update_group(self, group: FaceIdentityGroup, observation: FaceObservation) -> None:
        group.observations.append(observation)

        embeddings = np.stack([obs.embedding for obs in group.observations])
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        group.representative_embedding = centroid.astype(np.float32)

        if _observation_quality(observation) > _observation_quality(group.representative_observation):
            group.representative_observation = observation

    def add(self, observation: FaceObservation) -> int | None:
        """
        Assigns an observation to an identity group.

        Low-quality or malformed observations (below the confidence/size
        floor, or a non-finite embedding) are kept aside in `unassigned`
        rather than risking a bad match.

        Returns:
            The assigned group_id, or None if the observation was left
            unassigned.
        """
        if not self._is_reliable(observation):
            self.unassigned.append(observation)
            return None

        best_index, best_similarity, second_best_similarity = self._best_match(observation.embedding)

        is_confident_match = (
            best_index is not None
            and best_similarity >= self.similarity_threshold
            and (best_similarity - second_best_similarity) >= self.margin_threshold
        )

        if is_confident_match:
            group = self.groups[best_index]
            self._update_group(group, observation)
            return group.group_id

        return self._start_group(observation)
