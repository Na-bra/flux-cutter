from dataclasses import dataclass, field

import numpy as np

from app.faces.detector import FaceDetection

# Re-tuned when ArcFace (w600k_r50) replaced SFace as the embedder. The
# previous 0.45 was fitted to SFace's similarity distribution and does not
# transfer: ArcFace pushes different-person pairs much closer to zero
# (median pairwise similarity 0.077 vs SFace's 0.154 on this footage), so
# the whole operating range shifts down.
#
# Sweeping the same test footage showed a sharp cliff rather than a gentle
# curve. At 0.33 and below, centroid drift still chained 9 detections
# spanning 19s into a single group whose weakest internal pair scored only
# +0.070 -- clearly different people, the same failure mode the original
# SFace experiment found. At 0.34 that group breaks apart, and 0.35/0.36/
# 0.37 all produce identical grouping (a stable plateau).
#
# 0.35 is the lowest value on that plateau. It is preferred over the
# literal cliff-edge value of 0.34 because sitting exactly on a
# discontinuity is fragile against footage variation, and this grouper
# already treats a false merge as worse than a missed match. See the
# stage-0.2 accuracy notes for the full experiment.
DEFAULT_SIMILARITY_THRESHOLD = 0.35
# Required gap between a merge and the best competing alternative before
# the merge is allowed. Disabled by default (0.0) as of the first
# full-length test, and that default is the interesting part.
#
# The rule looked free on the 23s clip: grouping there was identical
# anywhere in 0.00-0.08, so 0.05 seemed like a safe margin bought for
# nothing. On a 22-minute episode with 26 recurring characters it turned
# out to be the single largest source of fragmentation. 54% of the small
# leftover groups already scored >= the similarity floor against a main
# character and were refused anyway: with that many similar-looking young
# actors on screen, a fragment is routinely near-tied between two of them,
# which is exactly the condition this rule vetoes.
#
# Measured on test_3.mp4 at 1.0s sampling (margin 0.05 -> 0.00):
#   identity groups        412  -> 221
#   groups of <= 2 dets    294  -> 138
#   share of detections in
#   main-character groups  64.5% -> 78.8%
# and it is not a precision-for-recall trade: the weakest 5% of members in
# the worst group moved from 0.565 to 0.475 similarity-to-centroid, still
# far above the ~0.08 typical of genuinely different people.
#
# The rule is kept, not deleted, because it guards a real failure mode on
# footage with few faces and heavy ambiguity -- raise it per-run via
# --margin-threshold if a specific video shows false merges.
DEFAULT_MARGIN_THRESHOLD = 0.0
# Centroid agreement required to fold two whole groups together in the
# consolidation pass that follows clustering. See IdentityGrouper._consolidate
# for why average linkage cannot catch these on its own.
#
# Re-derived once co-occurrence gave a way to label group pairs without
# guessing. 0.50 was fitted to the three same-actor pairs that happened to be
# visible among the 30 largest groups; measured across all 780 pairs it turned
# out to be far above anything that actually merges. At 0.50, and anywhere down
# to 0.41, consolidation fires zero times on test_3.mp4.
#
# The labelled picture (see DEFAULT_FORBID_COOCCURRING for where the labels
# come from):
#
#   pairs that provably co-occur, so are different people   max 0.334
#   pairs free to merge, confirmed by eye to be one actor       0.400, 0.405
#
# 0.375 sits in that gap. It recovers both confirmed missed merges -- one
# actor in face paint, and one girl split across two lighting setups -- while
# staying above every pair the footage proves is two people.
#
# Lower would not obviously be wrong, but it would be unevidenced: below 0.334
# there are labelled different-person pairs, and the co-occurrence rule only
# protects pairs that happen to share a frame.
DEFAULT_CONSOLIDATION_THRESHOLD = 0.375
DEFAULT_MIN_CONFIDENCE = 0.7
# Shorter side of the face box, in pixels, below which an embedding is
# considered too unreliable to trust for identity matching.
DEFAULT_MIN_FACE_SIZE = 40
# Median eye separation, as a fraction of face-box width, below which a whole
# GROUP is judged not to be a person and its observations are returned as
# unassigned instead of becoming a person card.
#
# YuNet reports "faces" confidently for backs of heads, extreme profiles and
# graphics (on test_3.mp4, the show's spinning logo), and nothing downstream
# asks whether a detection is really a face. Those degenerate detections then
# resemble *each other* -- their landmarks collapse the same way -- so they
# cluster into a large, stable phantom identity. On test_3.mp4 that phantom
# held 185 detections and spanned the whole episode.
#
# The test is applied per group rather than per detection on purpose. At the
# detection level the signal does not separate: real people are also filmed in
# profile, and a cutoff that removed most degenerate frames also discarded
# 8-14% of genuine ones. Those frames are not a problem individually because
# tracking attaches them to a track that has better frames. It is only a whole
# *cluster* of them, with no good frames anywhere, that is not a person.
#
# 0.15 is calibrated on test_3.mp4: real identity groups have a median eye span
# of 0.31-0.43, the phantom group 0.14. At this value the phantom is the only
# group of 5+ detections removed -- the costumed character (whose mask distorts
# the landmarks) and a legitimate profile-heavy cluster both survive.
DEFAULT_MIN_GROUP_EYE_SPAN = 0.15
# How much screen time an identity needs before it is worth reporting as a
# person, used by auto_min_detections() to derive a minimum detection count.
#
# A raw detection count cannot be the setting, because it means different
# things at different sampling rates: on the 23s clip the largest identity
# holds 5 detections at a 1.0s interval and 26 at 0.25s. Screen time
# (detections x interval) is the sampling-invariant quantity.
#
# Screen time alone is not enough either, because significance is relative to
# runtime: three seconds is an eighth of a 23s clip and a rounding error in a
# feature film. So the requirement is the larger of an absolute floor and a
# share of the runtime.
# Whether two identities seen in the SAME sampled frame may ever be merged.
#
# They may not: a person cannot stand in two places at once, so two
# simultaneous non-overlapping face boxes are two different people. That is
# ground truth, not a similarity estimate, and it is free -- the frame index
# is already on every observation.
#
# It matters because it is the only hard evidence in the whole pipeline.
# Everything else here is a threshold fitted to one video; this is a fact
# about the footage. Measured on test_3.mp4 at 1.0s, the baseline grouping
# put two same-frame faces into one identity 11 times, across 5 of its 40
# groups -- every one a false merge, and none of them reachable by tuning:
# the detections involved have median confidence 0.917, median box 74px and
# not one of them falls below the blur floor that was the suspected culprit.
#
# It also earns its keep in the other direction. The highest-scoring pair of
# groups that provably co-occur scores 0.334 centroid similarity, and the two
# confirmed missed merges score 0.400 and 0.405. Blocking the co-occurring
# pair outright is what makes it safe to lower the consolidation threshold
# into that gap instead of sitting at 0.50 where nothing merges at all.
#
# The assumption it makes is that one person is not visible twice in a single
# frame. Mirrors, photographs and screens can break that; when they do, the
# result is a refused merge, which is this project's cheap error.
DEFAULT_FORBID_COOCCURRING = True
# Similarity above which a shared frame stops being evidence of two people.
#
# The rule above assumes one person cannot be in two places at once. Screens,
# photographs and split-screen editing break that, and this footage does all
# three: at t=336s and t=368s the title sequence tiles several clips of the
# SAME actor side by side, and at t=490s YuNet finds two faces in the
# photographs stuck to a fridge. Those pairs are not two people, and blocking
# them costs real merges.
#
# They are separable because they do not look like two people. Across 4060
# same-frame pairs on test_3.mp4 the similarity distribution is
# median 0.024, p99 0.291 -- and only 6 pairs reach 0.50, four of which are
# the title sequence showing one actor twice. So a shared frame is read as
# evidence only while the two faces are as different as two people actually
# are; past that, the split screen is the better explanation.
#
# 0.50 misreads 2 pairs in 4060 (0.05%), both between tiny background extras
# rather than anyone who ends up on a person card. The alternative -- no
# ceiling -- breaks the main cast apart at the title sequence, which is
# measurably worse: it moved 28 of group 1's 516 detections into a duplicate
# identity.
DEFAULT_COOCCURRENCE_SIMILARITY_CEILING = 0.50
DEFAULT_MIN_APPEARANCE_SECONDS = 3.0
DEFAULT_MIN_APPEARANCE_SHARE = 0.005


def auto_min_detections(
    duration_seconds: float | None,
    sample_interval: float,
    floor_seconds: float = DEFAULT_MIN_APPEARANCE_SECONDS,
    share: float = DEFAULT_MIN_APPEARANCE_SHARE,
) -> int:
    """How many detections an identity needs before it is worth reporting.

    Returns the number of sampled detections corresponding to
    `max(floor_seconds, share * duration)` seconds of screen time, so the
    same setting means the same thing whatever the sampling interval.

    Worked examples at a 1.0s interval: a 23s clip requires 3 detections, a
    22-minute episode 7, a 90-minute film 27. At 0.25s those become 12, 27
    and 108 -- the same screen time, four times the samples.

    A video whose duration is unknown falls back to the floor alone, which
    is the conservative choice: it filters obvious noise without assuming a
    runtime that might be wrong.
    """
    if sample_interval <= 0:
        raise ValueError("sample_interval must be greater than 0")

    required_seconds = floor_seconds
    if duration_seconds is not None and duration_seconds > 0:
        required_seconds = max(floor_seconds, share * duration_seconds)

    return max(1, round(required_seconds / sample_interval))


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings (their dot product)."""
    return float(np.dot(embedding_a, embedding_b))


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalized mean of several embeddings.

    Used both for a group's running centroid and for a track's averaged
    identity vector: a single frame's ArcFace embedding on hard footage
    (profile, motion blur, harsh key light) is noisy enough that
    same-person pairs routinely fall below the similarity floor, while the
    mean over several frames is a much steadier estimate.
    """
    stacked = np.stack(embeddings)
    centroid = stacked.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


@dataclass(frozen=True)
class FaceObservation:
    """A single embedded face detection, ready for identity grouping."""

    embedding: np.ndarray
    detection: FaceDetection
    face_crop: np.ndarray
    source_timestamp: float
    frame_index: int | None = None
    # Focus of the aligned crop this embedding came from. Optional because
    # observations built by hand (tests, and the appearance-timeline code)
    # have no crop to measure; None means "unknown", never "bad".
    sharpness: float | None = None
    # Which model produced `embedding`. None means "not stated", which is
    # what hand-built observations carry and is treated as compatible with
    # anything -- the guard is against mixing two *named* spaces, not
    # against callers that never had one.
    embedding_space: str | None = None


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


def eye_span_ratio(detection: FaceDetection) -> float | None:
    """Eye separation as a fraction of the face box width, or None if unmeasurable.

    On a face turned toward the camera the two eyes sit roughly a third of the
    box apart. On a back of a head, a hard profile, or a graphic that merely
    tripped the detector, YuNet still emits five points but they collapse
    together, so this ratio falls away. It is the cheapest available check on
    whether a "face" is really a face -- the landmarks are already computed.
    """
    if detection.landmarks is None:
        return None

    width = detection.box.x_max - detection.box.x_min
    if width <= 0:
        return None

    left = np.asarray(detection.landmarks.left_eye, dtype=np.float64)
    right = np.asarray(detection.landmarks.right_eye, dtype=np.float64)
    return float(np.linalg.norm(left - right) / width)


def _cooccurrence_conflicts(
    units: list[list[FaceObservation]],
    similarity_ceiling: float = DEFAULT_COOCCURRENCE_SIMILARITY_CEILING,
) -> np.ndarray:
    """Which units can never be the same person, because they share a frame.

    Built from a frame -> units index rather than by comparing every pair of
    units: the work is then proportional to the number of faces that actually
    share a frame (a few thousand) instead of to the square of the unit count
    (nearly two million on a 22-minute episode).

    Observations with no frame index contribute nothing, so hand-built
    observations and any future caller that does not track frames simply get
    no constraint rather than a wrong one.
    """
    unit_count = len(units)
    conflict = np.zeros((unit_count, unit_count), dtype=bool)

    # frame -> [(unit index, the observation that unit has in this frame)]
    units_in_frame: dict[int, list[tuple[int, FaceObservation]]] = {}
    for index, unit in enumerate(units):
        for observation in unit:
            if observation.frame_index is None:
                continue
            units_in_frame.setdefault(observation.frame_index, []).append((index, observation))

    for sharing in units_in_frame.values():
        if len(sharing) < 2:
            continue
        for position, (first, first_observation) in enumerate(sharing):
            for second, second_observation in sharing[position + 1 :]:
                if first == second:
                    continue
                # See DEFAULT_COOCCURRENCE_SIMILARITY_CEILING: too alike to be
                # two people means the frame is showing one person twice.
                if (
                    cosine_similarity(first_observation.embedding, second_observation.embedding)
                    >= similarity_ceiling
                ):
                    continue
                conflict[first, second] = True
                conflict[second, first] = True

    return conflict


# Where each factor in _observation_quality stops earning more credit. Both
# saturate because past a point they stop describing a better picture of the
# person: a 400px face is not twice the portrait a 200px one is, and neither
# is a crop of a patterned shirt collar that happens to score 2000 on a
# sharpness measure that rewards any high-frequency detail at all.
REPRESENTATIVE_SIZE_SATURATION = 120
REPRESENTATIVE_SHARPNESS_SATURATION = 200.0


def _observation_quality(
    observation: FaceObservation, centroid: np.ndarray | None = None
) -> float:
    """How well one observation would stand for its identity in a gallery.

    Used only to choose the picture on a person card. It has no influence on
    grouping, so getting it wrong costs a bad thumbnail rather than a bad
    identity -- but a bad thumbnail is what the person picking a face to
    export actually sees.

    This was confidence x box area, which is dominated by area: face boxes on
    one video vary by two orders of magnitude in area and barely one in
    confidence, so in practice it selected "the biggest box" and nothing else.
    A big motion-blurred lunge at the camera outranked every clean mid-shot,
    which is how the accuracy notes ended up describing cards showing hair and
    necks for groups whose members are mostly clean faces.

    Four factors now, all in [0, 1] so none can run away with the score:

    - **confidence**, as before.
    - **size**, saturating: enough pixels to see, past which more do not help.
    - **sharpness**, saturating: the point of the change. An observation whose
      sharpness is unknown scores neutrally rather than badly, so hand-built
      observations do not all rank last.
    - **agreement with the identity's centroid**, which is what rules out the
      hair and neck crops directly. They are not merely blurry, they are
      unlike the rest of the group -- and unlike the person the card claims to
      show.
    """
    box = observation.detection.box
    short_side = min(max(0, box.x_max - box.x_min), max(0, box.y_max - box.y_min))
    size_term = min(short_side, REPRESENTATIVE_SIZE_SATURATION) / REPRESENTATIVE_SIZE_SATURATION

    if observation.sharpness is None:
        sharpness_term = 0.5
    else:
        sharpness_term = (
            min(observation.sharpness, REPRESENTATIVE_SHARPNESS_SATURATION)
            / REPRESENTATIVE_SHARPNESS_SATURATION
        )

    agreement = 1.0
    if centroid is not None and _is_valid_embedding(observation.embedding):
        agreement = max(0.0, float(np.dot(observation.embedding, centroid)))

    return observation.detection.confidence * size_term * sharpness_term * agreement


class MixedEmbeddingSpaces(ValueError):
    """Raised when observations from two different models reach one grouper.

    Cosine similarity between an ArcFace vector and a CCIP one is a
    perfectly well-formed float, which is the danger: nothing downstream
    would notice, and every threshold in this module would be meaningless.
    The two spaces are not even scaled alike -- two *different* anime
    characters sit at a median 0.567 where two different people sit at 0.03
    -- so a mixed run does not degrade gracefully, it collapses everyone
    into one identity.
    """


class IdentityGrouper:
    """Groups embedded face observations into per-identity clusters.

    Clustering is agglomerative with average linkage: every unit (a single
    observation, or a whole face track) starts as its own cluster, and the
    globally most-similar pair of clusters is merged repeatedly until the
    best remaining pair falls below the similarity floor.

    This replaced an incremental nearest-centroid pass that assigned each
    unit to the best group that happened to exist at the moment it arrived.
    That approach was order-dependent to a degree that swamped everything
    else: on the test footage, feeding the same 33 tracks in 12 different
    orders produced anywhere from 11 to 17 groups, with identical data and
    identical thresholds. A single early mistake also permanently polluted
    a centroid, which is the "centroid drift" the stage-0.2 accuracy notes
    describe. Average linkage removes both problems -- it looks at every
    pair before committing to any merge, so the result is a property of the
    data rather than of arrival order.

    A false merge is still treated as worse than a missed match: merges
    must clear an absolute similarity floor, and a merge whose runner-up is
    nearly as good is refused as ambiguous rather than forced.

    Because clustering needs every unit before it can run, `add`/`add_track`
    only buffer; grouping happens on first access to `groups` (or an
    explicit `finish()`).
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
        consolidation_threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
        min_group_eye_span: float = DEFAULT_MIN_GROUP_EYE_SPAN,
        min_detections: int = 1,
        forbid_cooccurring: bool = DEFAULT_FORBID_COOCCURRING,
        cooccurrence_similarity_ceiling: float = DEFAULT_COOCCURRENCE_SIMILARITY_CEILING,
    ):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [-1.0, 1.0]")
        if margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        if not -1.0 <= consolidation_threshold <= 1.0:
            raise ValueError("consolidation_threshold must be within [-1.0, 1.0]")

        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.consolidation_threshold = consolidation_threshold
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.min_group_eye_span = min_group_eye_span
        self.min_detections = max(1, int(min_detections))
        self.forbid_cooccurring = forbid_cooccurring
        self.cooccurrence_similarity_ceiling = cooccurrence_similarity_ceiling

        # The space every observation must agree on, learned from the first
        # one that names it rather than configured: the grouper has no
        # business knowing which models exist.
        self._embedding_space: str | None = None
        self._unreliable: list[FaceObservation] = []
        self._rejected: list[FaceObservation] = []
        self._units: list[list[FaceObservation]] = []
        self._groups: list[FaceIdentityGroup] | None = None

    @property
    def unassigned(self) -> list[FaceObservation]:
        """Observations that produced no identity.

        Two sources: detections rejected up front as unreliable, and whole
        groups the non-face filter rejected. The second only exists once
        clustering has run, so this resolves grouping first.
        """
        groups = self.groups  # noqa: F841 - ensures _rejected is populated
        return self._unreliable + self._rejected

    @property
    def groups(self) -> list[FaceIdentityGroup]:
        """The clustered identity groups, computing them on first access."""
        if self._groups is None:
            self._groups = self._cluster()
        return self._groups

    def _check_embedding_space(self, observation: FaceObservation) -> None:
        """Refuses observations from a model this grouper is not already using."""
        space = observation.embedding_space
        if space is None:
            return
        if self._embedding_space is None:
            self._embedding_space = space
            return
        if space != self._embedding_space:
            raise MixedEmbeddingSpaces(
                f"This grouper is comparing {self._embedding_space!r} embeddings "
                f"and was given a {space!r} one. Vectors from different models "
                "are not comparable; run each content mode as its own scan."
            )

    def accepts_detection(self, detection) -> bool:
        """Whether a detection could survive grouping, judged before embedding.

        Two of the three reliability tests -- confidence and face size --
        look only at the detection, so they can be answered before the
        embedder runs. Embedding is the pipeline's dominant cost, and on
        the test footage 12.3% of detections were embedded only to be
        discarded here immediately afterwards.

        Exposed rather than duplicated at the call site so the thresholds
        keep one home: a caller that skips a face this rejects gets the
        same grouping it would have got by embedding everything --
        verified as an identical partition, 0 disagreements over 2.44
        million co-membership pairs on test_3.mp4.

        The third test (that the embedding is finite and non-degenerate)
        necessarily still runs afterwards, in _is_reliable.
        """
        if detection.confidence < self.min_confidence:
            return False

        box = detection.box
        width = box.x_max - box.x_min
        height = box.y_max - box.y_min
        return min(width, height) >= self.min_face_size

    def _is_reliable(self, observation: FaceObservation) -> bool:
        if not _is_valid_embedding(observation.embedding):
            return False
        return self.accepts_detection(observation.detection)

    def _add_unit(self, observations: list[FaceObservation]) -> int:
        """Buffers one indivisible unit and invalidates any cached grouping."""
        self._units.append(observations)
        self._groups = None
        return len(self._units) - 1

    def add(self, observation: FaceObservation) -> int | None:
        """
        Buffers a single observation as its own unit.

        Low-quality or malformed observations (below the confidence/size
        floor, or a non-finite embedding) are kept aside in `unassigned`
        rather than risking a bad match.

        Returns:
            The unit index, or None if the observation was left unassigned.
            Note this is a unit index, not a group id: groups do not exist
            until clustering runs.
        """
        self._check_embedding_space(observation)
        if not self._is_reliable(observation):
            self._unreliable.append(observation)
            return None

        return self._add_unit([observation])

    def add_track(self, track) -> int | None:
        """
        Buffers a whole face track as one indivisible unit.

        A track's frames are already known to be the same person by spatial
        continuity, so they are never split apart by clustering, and the
        track is compared to other units by its averaged embedding rather
        than any single noisy frame. Unreliable observations are dropped
        first so a few bad frames cannot drag that average around.

        Returns:
            The unit index, or None if nothing in the track was reliable
            enough to group.
        """
        for observation in track.observations:
            self._check_embedding_space(observation)
        reliable = [obs for obs in track.observations if self._is_reliable(obs)]
        if not reliable:
            self._unreliable.extend(track.observations)
            return None

        return self._add_unit(reliable)

    def finish(self) -> list[FaceIdentityGroup]:
        """Forces clustering to run and returns the resulting groups."""
        return self.groups

    def _cluster(self) -> list[FaceIdentityGroup]:
        """Agglomerative average-linkage merging over the buffered units.

        Average linkage between two clusters is the mean cosine similarity
        over every cross-cluster observation pair. Because the embeddings
        are L2-normalized, that mean is (sum_a . sum_b) / (n_a * n_b),
        which lets the whole initial matrix be built with one matrix
        product instead of an explicit pairwise loop.

        Merges then update the matrix by the Lance-Williams rule for
        average linkage, an observation-count-weighted mean of the two
        merged rows. That is exact -- it gives the same numbers a full
        recomputation would -- while keeping the whole run O(n^2) in the
        number of units rather than recomputing every pair after each
        merge.
        """
        if not self._units:
            return []

        unit_count = len(self._units)
        sizes = np.array([len(unit) for unit in self._units], dtype=np.float64)
        sums = np.stack(
            [np.stack([obs.embedding for obs in unit]).astype(np.float64).sum(axis=0) for unit in self._units]
        )

        # Mean pairwise similarity between every pair of units.
        similarity = (sums @ sums.T) / np.outer(sizes, sizes)
        np.fill_diagonal(similarity, -np.inf)

        # Units seen in the same frame are different people, so their
        # similarity is irrelevant -- struck out rather than merely
        # discouraged, because no amount of resemblance makes two faces
        # standing side by side one person.
        conflict = (
            _cooccurrence_conflicts(self._units, self.cooccurrence_similarity_ceiling)
            if self.forbid_cooccurring
            else np.zeros_like(similarity, dtype=bool)
        )
        similarity[conflict] = -np.inf

        members: list[list[int] | None] = [[i] for i in range(unit_count)]
        # Pairs refused as ambiguous, so a blocked pair is not retried forever.
        blocked: set[tuple[int, int]] = set()

        while True:
            candidate = self._best_mergeable_pair(similarity, blocked)
            if candidate is None:
                break

            first, second, best_similarity = candidate
            if not self._merge_is_unambiguous(
                similarity, first, second, best_similarity, self.similarity_threshold
            ):
                blocked.add((min(first, second), max(first, second)))
                continue

            # Lance-Williams average-linkage update, weighted by observation count.
            merged_size = sizes[first] + sizes[second]
            similarity[first, :] = (
                sizes[first] * similarity[first, :] + sizes[second] * similarity[second, :]
            ) / merged_size
            similarity[:, first] = similarity[first, :]
            similarity[first, first] = -np.inf

            similarity[second, :] = -np.inf
            similarity[second, first] = -np.inf
            similarity[:, second] = -np.inf

            # The merged cluster inherits both halves' conflicts: anyone who
            # could not be either of them cannot be the pair of them.
            conflict[first] |= conflict[second]
            conflict[:, first] = conflict[first]
            similarity[first, conflict[first]] = -np.inf
            similarity[conflict[first], first] = -np.inf

            members[first] = members[first] + members[second]
            members[second] = None
            sizes[first] = merged_size

            # A merged cluster is a different cluster: give its pairs a fresh chance.
            blocked = {pair for pair in blocked if first not in pair and second not in pair}

        self._rejected = []
        groups = self._consolidate(self._build_groups(members))
        groups = self._reject_non_face_groups(groups)
        groups = self._reject_brief_groups(groups)
        return self._renumber(groups)

    def _best_mergeable_pair(
        self, similarity: np.ndarray, blocked: set[tuple[int, int]]
    ) -> tuple[int, int, float] | None:
        """The most similar pair of live clusters that clears the floor."""
        workspace = similarity.copy()
        for first, second in blocked:
            workspace[first, second] = -np.inf
            workspace[second, first] = -np.inf

        best_flat = int(np.argmax(workspace))
        first, second = np.unravel_index(best_flat, workspace.shape)
        best_similarity = float(workspace[first, second])

        if not np.isfinite(best_similarity) or best_similarity < self.similarity_threshold:
            return None
        return int(first), int(second), best_similarity

    def _merge_is_unambiguous(
        self,
        similarity: np.ndarray,
        first: int,
        second: int,
        best_similarity: float,
        floor: float,
    ) -> bool:
        """Whether this merge is free of a genuinely competing alternative.

        This carries the margin rule over from the previous nearest-centroid
        implementation, but it cannot be carried over literally. There,
        "the runner-up is nearly as similar" meant a unit matched two rival
        identities about equally well. Under average linkage it usually
        means the opposite: when three clips of one person are all mutually
        similar, every pair scores high, so a naive runner-up test blocks
        the very merges it should allow.

        So a near-tie only counts as ambiguity when the competitor is a
        *different* identity -- close to one endpoint, yet too far from the
        other to be merged with it. That is the case where picking the
        better-by-a-hair option is a coin flip, and a coin flip is how a
        false merge gets in.
        """
        if self.margin_threshold <= 0.0:
            return True

        for endpoint, partner in ((first, second), (second, first)):
            for other in range(similarity.shape[0]):
                if other in (first, second):
                    continue

                competitor = float(similarity[endpoint, other])
                if not np.isfinite(competitor):
                    continue
                if best_similarity - competitor >= self.margin_threshold:
                    continue

                # Near-tie. Ambiguous only if the competitor is a different
                # identity rather than another piece of the same one.
                rival = float(similarity[partner, other])
                if not np.isfinite(rival) or rival < floor:
                    return False

        return True

    def _build_groups(self, members: list[list[int] | None]) -> list[FaceIdentityGroup]:
        """Materializes surviving clusters into FaceIdentityGroups, largest first."""
        groups: list[FaceIdentityGroup] = []
        for member_units in members:
            if member_units is None:
                continue

            observations = [obs for unit_index in sorted(member_units) for obs in self._units[unit_index]]
            group = FaceIdentityGroup(group_id=0, observations=observations)
            self._recompute_group(group)
            groups.append(group)

        groups.sort(key=lambda group: (-group.size, group.observations[0].source_timestamp))
        for position, group in enumerate(groups, start=1):
            group.group_id = position
        return groups

    def _consolidate(self, groups: list[FaceIdentityGroup]) -> list[FaceIdentityGroup]:
        """Folds together whole groups whose centroids agree, after clustering.

        Average linkage asks whether a candidate resembles *every* frame of a
        cluster, which is the right question while clusters are small and the
        wrong one once they are large and varied. One actor across a 22-minute
        episode is not a blob in embedding space: frontal frames, profiles and
        -- on this footage literally -- the same character wearing a superhero
        mask occupy separate regions. The mean over all cross-pairs is dragged
        below the similarity floor by that spread even when both halves plainly
        belong to one person.

        Measured on test_3.mp4, the three big-group pairs that were visibly the
        same actor scored 0.549, 0.550 and 0.626 centroid similarity but only
        0.245, 0.275 and 0.315 average linkage -- all three below the 0.35 floor,
        so clustering left them as separate people. Those are the "duplicates"
        that leak into the montage.

        Comparing prototype to prototype instead recovers them. It is run as a
        second phase rather than as the linkage rule itself because centroid
        linkage applied from the start is much looser while clusters are still
        one or two frames wide, where a single noisy embedding *is* the
        prototype. Clustering conservatively first and consolidating afterwards
        gets the benefit only where there is enough evidence to support it.

        Merging is greedy on the globally best pair and repeats until nothing
        clears the threshold, so like the clustering phase the result does not
        depend on group order.
        """
        if self.consolidation_threshold > 1.0 or len(groups) < 2:
            return groups

        sums = np.stack(
            [
                np.stack([obs.embedding for obs in group.observations]).astype(np.float64).sum(axis=0)
                for group in groups
            ]
        )
        members: list[list[int] | None] = [[i] for i in range(len(groups))]
        alive = np.ones(len(groups), dtype=bool)

        def centroid_row(index: int) -> np.ndarray:
            """Centroid similarity of one group against every live group."""
            centre = sums[index] / np.linalg.norm(sums[index])
            live = np.where(alive)[0]
            row = np.full(len(groups), -np.inf)
            norms = np.linalg.norm(sums[live], axis=1, keepdims=True)
            row[live] = ((sums[live] / norms) @ centre)
            row[index] = -np.inf
            return row

        similarity = np.stack([centroid_row(i) for i in range(len(groups))])

        # The same hard constraint as clustering. It matters more here: this
        # pass compares whole identities by centroid, which is exactly the
        # comparison most likely to find two different actors similar once
        # each has enough frames to average out lighting and pose.
        conflict = (
            _cooccurrence_conflicts(
                [group.observations for group in groups], self.cooccurrence_similarity_ceiling
            )
            if self.forbid_cooccurring
            else np.zeros(similarity.shape, dtype=bool)
        )
        similarity[conflict] = -np.inf

        # Pairs refused as ambiguous, so a blocked pair is not retried forever.
        blocked: set[tuple[int, int]] = set()

        while True:
            workspace = similarity.copy()
            for blocked_first, blocked_second in blocked:
                workspace[blocked_first, blocked_second] = -np.inf
                workspace[blocked_second, blocked_first] = -np.inf

            first, second = np.unravel_index(int(np.argmax(workspace)), workspace.shape)
            best = float(workspace[first, second])
            if not np.isfinite(best) or best < self.consolidation_threshold:
                break

            # The margin rule applies here too. Without it, consolidation would
            # quietly re-merge the very pairs clustering refused as a coin flip
            # between two identities, which is the opposite of what it is for.
            if not self._merge_is_unambiguous(
                similarity, int(first), int(second), best, self.consolidation_threshold
            ):
                blocked.add((min(first, second), max(first, second)))
                continue

            sums[first] = sums[first] + sums[second]
            members[first] = members[first] + members[second]
            members[second] = None
            alive[second] = False
            similarity[second, :] = -np.inf
            similarity[:, second] = -np.inf

            conflict[first] |= conflict[second]
            conflict[:, first] = conflict[first]

            # centroid_row recomputes from the embeddings and so knows nothing
            # about the constraint; re-applying it here is what stops a merged
            # group quietly becoming mergeable with someone it stood next to.
            row = centroid_row(int(first))
            row[conflict[first]] = -np.inf
            similarity[first, :] = row
            similarity[:, first] = row
            # A merged group is a different group: give its pairs a fresh chance.
            blocked = {pair for pair in blocked if first not in pair and second not in pair}

        merged: list[FaceIdentityGroup] = []
        for member_groups in members:
            if member_groups is None:
                continue
            observations = [obs for i in member_groups for obs in groups[i].observations]
            group = FaceIdentityGroup(group_id=0, observations=observations)
            self._recompute_group(group)
            merged.append(group)

        merged.sort(key=lambda g: (-g.size, g.observations[0].source_timestamp))
        for position, group in enumerate(merged, start=1):
            group.group_id = position
        return merged

    def _reject_non_face_groups(
        self, groups: list[FaceIdentityGroup]
    ) -> list[FaceIdentityGroup]:
        """Drops clusters whose landmark geometry says they are not a person.

        See DEFAULT_MIN_GROUP_EYE_SPAN for why this is judged per group rather
        than per detection. Rejected observations are reported as unassigned
        rather than discarded: they were detected, they just could not be
        attributed to anyone, and silently losing them would misreport how much
        of the video the pipeline actually accounted for.
        """
        if self.min_group_eye_span <= 0.0:
            return groups

        kept: list[FaceIdentityGroup] = []
        for group in groups:
            spans = [
                span
                for span in (eye_span_ratio(obs.detection) for obs in group.observations)
                if span is not None
            ]
            if spans and float(np.median(spans)) < self.min_group_eye_span:
                self._rejected.extend(group.observations)
                continue
            kept.append(group)

        return kept

    def _reject_brief_groups(
        self, groups: list[FaceIdentityGroup]
    ) -> list[FaceIdentityGroup]:
        """Drops identities with too little screen time to be worth selecting.

        The count itself is supplied by the caller rather than derived here,
        because it depends on the video's duration and sampling interval and
        the grouper deliberately knows about neither. See auto_min_detections().

        As with the non-face filter, the observations are reported as
        unassigned rather than discarded: a face that appeared for one sampled
        frame was still a face, it just is not an identity anyone would pick
        out of a gallery.
        """
        if self.min_detections <= 1:
            return groups

        kept: list[FaceIdentityGroup] = []
        for group in groups:
            if group.size < self.min_detections:
                self._rejected.extend(group.observations)
                continue
            kept.append(group)

        return kept

    @staticmethod
    def _renumber(groups: list[FaceIdentityGroup]) -> list[FaceIdentityGroup]:
        """Assigns contiguous 1-based ids after filtering has removed groups."""
        for position, group in enumerate(groups, start=1):
            group.group_id = position
        return groups

    def _recompute_group(self, group: FaceIdentityGroup) -> None:
        """Refreshes a group's centroid and representative from its members.

        The centroid is the mean over every member observation, so a group
        built from tracks is weighted by how long each track ran rather
        than treating a one-frame track and a thirty-frame one as equals.
        """
        group.representative_embedding = mean_embedding(
            [obs.embedding for obs in group.observations]
        )
        centroid = group.representative_embedding
        group.representative_observation = max(
            group.observations, key=lambda obs: _observation_quality(obs, centroid)
        )
