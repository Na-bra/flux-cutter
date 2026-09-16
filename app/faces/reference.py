"""Choosing a person with a photograph instead of a gallery click.

The gallery flow asks a question the user can only answer after a scan:
"which of these forty faces did you mean?". When they already know who they
want and have a picture of them, the question is answerable up front --
embed the photo once, and the scan's own identity groups can be scored
against it without anybody looking at a montage.

That is the whole idea, and it is deliberately built on the pieces that
already exist rather than a second matching path: the same detector, the
same embedder, the same cosine similarity, and the same per-mode floor the
grouper uses to decide two faces are one person. A reference face is just
one more embedding in the space the scan is already working in.

Two things this module refuses to do quietly, because both produce a
confidently wrong reel:

- **Match across embedding spaces.** An ArcFace vector and a CCIP vector
  have a cosine similarity; it means nothing. A photo embedded in one mode
  cannot select an identity grouped in the other.
- **Pick between two plausible people.** When the best and second-best
  identities are within a margin of each other, that is reported as
  ambiguous rather than resolved by rounding.

A reference does not have to be a photograph. A person already named on a
card in one scan is a better one: their face is averaged over every time
the scan saw them, in the footage's own lighting and on its own camera,
rather than taken from one picture the user had to go and find
(`reference_from_groups`). Matching treats both identically.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.faces.detector import FaceDetection
from app.faces.grouper import FaceIdentityGroup, cosine_similarity
from app.modes import DEFAULT_MODE, get_mode

# What PIL will open and this project will accept as a reference. Video
# extensions are deliberately absent: a still is the whole point, and a
# caller who wants a frame of a video can save one.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# How far ahead of the runner-up the best match has to be. Grouping already
# ensures the identities are meant to be distinct people, so a reference
# that scores nearly the same against two of them is evidence the scan
# split one person in two -- or that the photo is not clearly either of
# them. Both are worth stopping for, because the alternative is minutes of
# encoding the wrong face.
DEFAULT_MATCH_MARGIN = 0.05


class ReferenceError(Exception):
    """Raised when a reference image cannot be turned into one face."""


@dataclass(frozen=True)
class ReferenceFace:
    """One face to find in a scan: from a photograph, or from a named card."""

    embedding: np.ndarray
    embedding_space: str
    # The face found in the photograph; None when this came from cards.
    detection: FaceDetection | None
    # The photograph; None when this came from cards.
    source: Path | None
    # How many faces the photo held. One is the ordinary case; more means
    # the largest was taken and the caller should probably say so.
    face_count: int
    # What to call this reference when reporting; the photo's file name
    # when not given.
    label: str | None = None

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return self.source.name if self.source is not None else "the reference"


@dataclass(frozen=True)
class ReferenceMatch:
    """Which identity a reference picked out, and how clearly."""

    index: int
    similarity: float
    # None when there was only one identity to choose from.
    runner_up_similarity: float | None

    @property
    def margin(self) -> float | None:
        if self.runner_up_similarity is None:
            return None
        return self.similarity - self.runner_up_similarity


def load_image(image_path: str | Path) -> np.ndarray:
    """Reads a still image as an RGB array the detector can take.

    Raises:
        ReferenceError: If the path is missing, is not a supported image
            format, or cannot be decoded.
    """
    path = Path(image_path)

    if not path.exists():
        raise ReferenceError(f"Reference image does not exist: {path}")
    if not path.is_file():
        raise ReferenceError(f"Reference path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ReferenceError(
            f"Unsupported reference image format: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        with Image.open(path) as image:
            # EXIF rotation matters here in a way it does not for decoded
            # video: a phone photo of a face stored sideways is a sideways
            # face to the detector, which then finds nothing at all.
            from PIL import ImageOps

            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise ReferenceError(f"Could not read reference image: {path} ({error})") from error


def load_reference_face(
    image_path: str | Path,
    mode: str = DEFAULT_MODE,
    confidence_threshold: float | None = None,
) -> ReferenceFace:
    """Detects and embeds the face in a photograph.

    The largest face wins when a photo holds several, which is the useful
    reading of "here is a picture of them": the subject of a photograph is
    usually the biggest face in it, and bystanders are usually smaller and
    further back. The count comes back either way so a caller can say what
    it did rather than choose silently.

    Args:
        image_path: The photograph.
        mode: Which pipeline's models to use. Must be the same mode the
            scan runs in, or the resulting vector is not comparable.
        confidence_threshold: Detector floor; the mode's own when omitted.

    Raises:
        ReferenceError: If the image cannot be read, holds no detectable
            face, or holds one that cannot be embedded.
    """
    spec = get_mode(mode)
    image = load_image(image_path)

    detector = spec.build_detector(
        confidence_threshold=(
            spec.detection.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        ),
        min_face_size=spec.detection.min_face_size,
    )
    try:
        detections = detector.detect(image)
    finally:
        detector.close()

    if not detections:
        raise ReferenceError(
            f"No face found in {Path(image_path).name}. A clear, front-facing "
            "photo of one person works best."
        )

    largest = max(
        detections,
        key=lambda d: (d.box.x_max - d.box.x_min) * (d.box.y_max - d.box.y_min),
    )

    embedder = spec.build_embedder()
    try:
        embedded = embedder.embed_batch(image, [largest])[0]
    finally:
        embedder.close()

    if embedded is None:
        raise ReferenceError(
            f"The face in {Path(image_path).name} could not be embedded. It may "
            "be too small, too blurred, or too far from front-facing."
        )

    return ReferenceFace(
        embedding=embedded.embedding,
        embedding_space=embedded.embedding_space,
        detection=largest,
        source=Path(image_path),
        face_count=len(detections),
    )


def reference_from_groups(
    groups: list[FaceIdentityGroup], label: str
) -> ReferenceFace:
    """A reference built from cards a person has already named.

    Each card's own average face is normalised and the results averaged,
    so a card seen for twenty minutes and one seen for twenty seconds
    count equally: the aim is what the person looks like across the
    episodes they were named in, not in whichever episode had most of them.

    Raises:
        ReferenceError: If none of the cards has a face to average, or they
            were embedded by different models and cannot be combined.
    """
    vectors = []
    spaces = set()
    for group in groups:
        if group.representative_embedding is None:
            continue
        vector = np.asarray(group.representative_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        vectors.append(vector / norm)
        spaces.update(
            obs.embedding_space
            for obs in group.observations
            if obs.embedding_space is not None
        )

    if not vectors:
        raise ReferenceError(f"The cards named {label!r} hold no faces to match on.")
    if len(spaces) > 1:
        raise ReferenceError(
            f"The cards named {label!r} come from scans in different modes "
            f"({', '.join(sorted(spaces))}), and faces from different modes "
            "cannot be compared. Scan every video in the same mode."
        )

    average = np.mean(vectors, axis=0)
    return ReferenceFace(
        embedding=(average / np.linalg.norm(average)).astype(np.float32),
        embedding_space=next(iter(spaces)) if spaces else "",
        detection=None,
        source=None,
        face_count=1,
        label=label,
    )


def score_groups(
    reference: ReferenceFace, groups: list[FaceIdentityGroup]
) -> list[float]:
    """Similarity between the reference and each identity, in group order.

    Scored against each group's centroid rather than its best single frame.
    The centroid is the mean over every observation, so it is the same
    steadier estimate that made track averaging worth doing -- matching a
    photo against one lucky frame would be matching against noise.

    Raises:
        ReferenceError: If the groups were embedded by a different model
            than the reference, which makes the comparison meaningless.
    """
    scores = []
    for group in groups:
        if group.representative_embedding is None:
            scores.append(-1.0)
            continue

        space = next(
            (
                obs.embedding_space
                for obs in group.observations
                if obs.embedding_space is not None
            ),
            None,
        )
        if (
            space is not None
            and reference.embedding_space
            and space != reference.embedding_space
        ):
            raise ReferenceError(
                f"The reference {reference.name} was embedded as {reference.embedding_space} "
                f"but this scan produced {space} vectors. Run both in the same "
                "mode."
            )

        scores.append(cosine_similarity(reference.embedding, group.representative_embedding))
    return scores


def match_reference(
    reference: ReferenceFace,
    groups: list[FaceIdentityGroup],
    minimum_similarity: float | None = None,
    margin: float = DEFAULT_MATCH_MARGIN,
    mode: str = DEFAULT_MODE,
) -> ReferenceMatch:
    """Picks the identity a reference photo refers to.

    Args:
        reference: The embedded photograph.
        groups: The scan's identities, in the order the caller numbers them.
        minimum_similarity: The floor a match must clear. Defaults to the
            mode's own grouping threshold -- the number that already means
            "these are the same person" in this embedding space, so a
            reference is held to the same standard as a track.
        margin: How far clear of the runner-up the winner must be.
        mode: Supplies the default floor.

    Raises:
        ReferenceError: If there is nothing to match, nothing clears the
            floor, or two identities are too close to separate.
    """
    if not groups:
        raise ReferenceError("This video produced no identities to match against.")

    if minimum_similarity is None:
        minimum_similarity = get_mode(mode).grouping.similarity_threshold

    scores = score_groups(reference, groups)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best = ranked[0]
    runner_up = scores[ranked[1]] if len(ranked) > 1 else None

    if scores[best] < minimum_similarity:
        raise ReferenceError(
            f"No identity in this video matches {reference.name}. The "
            f"closest scored {scores[best]:.2f}, under the {minimum_similarity:.2f} "
            "floor. They may not appear in this video, or not clearly enough "
            "to be grouped."
        )

    if runner_up is not None and scores[best] - runner_up < margin:
        raise ReferenceError(
            f"{reference.name} matches two people about equally well "
            f"(#{best + 1} at {scores[best]:.2f}, #{ranked[1] + 1} at "
            f"{runner_up:.2f}). Pick one with --select-index rather than "
            "guessing between them."
        )

    return ReferenceMatch(
        index=best, similarity=scores[best], runner_up_similarity=runner_up
    )
