from dataclasses import dataclass, field

import numpy as np

from app.faces.detector import DEFAULT_IOU_THRESHOLD, intersection_over_union
from app.faces.grouper import (
    FaceObservation,
    _is_valid_embedding,
    cosine_similarity,
    mean_embedding,
)

# The overlap a live track's last detection and a candidate in the next
# sampled frame need before they are linked is DEFAULT_IOU_THRESHOLD, which
# lives beside intersection_over_union itself: the grouper's profile rescue
# asks the same question of the same number.
#
# How many consecutive sampled frames a track may go unmatched before it is
# closed. One tolerates a single dropped detection (a blink, a brief
# occlusion, a frame the detector missed) without splitting the track, which
# mirrors the same one-missed-sample allowance the timeline stage already
# makes when chaining appearances.
DEFAULT_MAX_FRAME_GAP = 1
# Appearance veto, NOT a match criterion. IoU alone cannot tell a continuing
# face from a hard shot cut that happens to place a different person in the
# same part of the screen, and this project treats a false merge as worse
# than a missed match. A candidate this dissimilar to the track's PREVIOUS
# frame is treated as a different person and breaks the track even when the
# boxes overlap.
#
# The comparison is deliberately against the immediately preceding frame
# rather than the track's running average. A running average washes the
# signal out: once a track has run for a couple of seconds its mean is
# dominated by history, so a cut-through still scores comfortably against
# it and slips past, while adjacent frames within a shot stay strongly
# similar no matter how long the track has run.
#
# Measured on the real test footage at 0.25s sampling, adjacent-frame
# similarity across 53 links has a median of 0.58 and runs up to 0.94.
# Links spanning a verified hard cut scored 0.177, 0.104 and -0.155, while
# the hardest genuine same-shot links (sharp head turns, motion blur) sat
# at 0.296-0.330. That leaves a usable gap, and 0.25 sits inside it:
# strict enough to break every verified cut, loose enough to keep the
# difficult real links intact. Sweeping the value confirmed it -- 0.25
# produced the cleanest grouping (14 identities), while 0.35 started
# splitting genuine links and 0.45 shattered tracking entirely (23).
#
# When it does misfire, it errs the cheap way: the grouping stage can
# re-merge two split tracks of one person, whereas a track that merges two
# people contaminates a group centroid irreversibly.
DEFAULT_CONTRADICTION_FLOOR = 0.25


def _comparable_similarity(
    observations: list[FaceObservation],
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair's similarity, and which pairs could be compared at all.

    An embedding that is missing, non-finite or zero-length cannot be
    compared with anything. Dividing by its length used to fill its whole
    row and column with NaN, and NaN loses every comparison below, so one
    unusable face made every cut look like the worst one and the track came
    apart a frame at a time. Such a face now carries no evidence either
    way: the faces around it are judged on each other.

    Returns:
        The similarity matrix, zero wherever a pair cannot be compared, and
        a boolean matrix of the pairs that can.
    """
    usable = np.array([_is_valid_embedding(o.embedding) for o in observations])
    width = max((o.embedding.size for o, ok in zip(observations, usable) if ok), default=0)
    vectors = np.zeros((len(observations), width), dtype=np.float64)
    for row, observation in enumerate(observations):
        if not usable[row] or observation.embedding.size != width:
            usable[row] = False
            continue
        length = np.linalg.norm(observation.embedding)
        if not length:
            usable[row] = False
            continue
        vectors[row] = observation.embedding / length
    return vectors @ vectors.T, np.outer(usable, usable)


def _worst_cut(
    similarity: np.ndarray, comparable: np.ndarray, start: int, end: int, threshold: float
) -> int | None:
    """Where one stretch of a track is least alike across, if that is below
    `threshold`.

    Returns:
        The index the tail begins at, or None to leave the stretch whole.
    """
    count = end - start
    if count < 2:
        return None
    block = similarity[start:end, start:end]
    # Sums of block[:k, k:] and of how many of its pairs count, for every
    # k, from 2-D cumulative sums -- the whole sweep in two passes rather
    # than one per candidate cut.
    totals = block.cumsum(axis=0).cumsum(axis=1)
    tallies = comparable[start:end, start:end].cumsum(axis=0).cumsum(axis=1)
    cuts = np.arange(1, count)
    sums = totals[cuts - 1, count - 1] - totals[cuts - 1, cuts - 1]
    pairs = tallies[cuts - 1, count - 1] - tallies[cuts - 1, cuts - 1]
    # A cut with nothing comparable across it is not evidence of anything.
    across = np.where(pairs > 0, sums / np.maximum(pairs, 1), np.inf)
    best = int(np.argmin(across))
    if across[best] >= threshold:
        return None
    return start + best + 1


def split_track(track: "FaceTrack", threshold: float) -> list[list[FaceObservation]]:
    """A track's observations, cut where it turns into someone else.

    The floor above compares each face with the one before, so a track can
    pass it at every step and still end on a different person: a dissolve
    from one face to another, or two people standing in the same place in
    turn. What gives it away is that its two ends are unlike each other.
    The cut goes where the faces before and after it are least alike on
    average, and is made only if that average is below `threshold`; each
    piece is then examined the same way.

    The pieces are found from one similarity matrix rather than by
    recursing with a fresh one per piece. The numbers are the same -- a
    piece's block of the whole matrix is the matrix that piece would build
    -- but a long track is measured once instead of once per cut, and a
    track that cuts at every frame no longer recurses as deep as it is
    long.
    """
    observations = track.observations
    if len(observations) < 2:
        return [observations]

    similarity, comparable = _comparable_similarity(observations)
    pieces: list[tuple[int, int]] = []
    stretches = [(0, len(observations))]
    while stretches:
        start, end = stretches.pop()
        cut = _worst_cut(similarity, comparable, start, end, threshold)
        if cut is None:
            pieces.append((start, end))
        else:
            stretches.append((start, cut))
            stretches.append((cut, end))

    pieces.sort()
    return [observations[start:end] for start, end in pieces]


@dataclass
class FaceTrack:
    """One face followed across consecutive sampled frames within a shot."""

    track_id: int
    observations: list[FaceObservation] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.observations)

    @property
    def start_time(self) -> float:
        return min(observation.source_timestamp for observation in self.observations)

    @property
    def end_time(self) -> float:
        return max(observation.source_timestamp for observation in self.observations)

    @property
    def embedding(self) -> np.ndarray:
        """The track's averaged identity embedding."""
        return mean_embedding([observation.embedding for observation in self.observations])

    @property
    def representative_observation(self) -> FaceObservation:
        """The highest-confidence observation, used for thumbnails."""
        return max(self.observations, key=lambda obs: obs.detection.confidence)


@dataclass
class _LiveTrack:
    """A track still open for extension, plus the bookkeeping to extend it."""

    track: FaceTrack
    last_frame_index: int
    last_observation: FaceObservation


class FaceTracker:
    """Links per-frame face detections into tracks by spatial continuity.

    Two faces detected at nearly the same place in consecutive sampled
    frames are the same person by continuity — no embedding comparison
    required to establish it. That is a far stronger signal than comparing
    two independent single-frame embeddings, and it is what lets the
    grouping stage work with averaged, low-noise identity vectors instead
    of raw per-frame ones.

    This only holds while sampling is dense enough that a face cannot move
    most of its own width between samples. At coarse intervals (~1s) shots
    cut and faces jump, so tracks degenerate to length 1 and the stage
    becomes a no-op rather than becoming wrong.
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
        contradiction_floor: float = DEFAULT_CONTRADICTION_FLOOR,
        split_threshold: float | None = None,
    ):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be within [0.0, 1.0]")
        if max_frame_gap < 0:
            raise ValueError("max_frame_gap must be non-negative")
        if not -1.0 <= contradiction_floor <= 1.0:
            raise ValueError("contradiction_floor must be within [-1.0, 1.0]")

        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self.contradiction_floor = contradiction_floor
        self.split_threshold = split_threshold

        self._live: list[_LiveTrack] = []
        self._closed: list[FaceTrack] = []
        self._next_track_id = 1

    def _close_stale(self, frame_index: int) -> None:
        still_live = []
        for live in self._live:
            # Count frames actually missed, not the raw index difference: a
            # track last seen at frame 4 and offered frame 5 has missed
            # nothing, while frame 6 means one missed sample.
            missed_frames = frame_index - live.last_frame_index - 1
            if missed_frames > self.max_frame_gap:
                self._closed.append(live.track)
            else:
                still_live.append(live)
        self._live = still_live

    def _start_track(self, observation: FaceObservation, frame_index: int) -> None:
        track = FaceTrack(track_id=self._next_track_id, observations=[observation])
        self._next_track_id += 1
        self._live.append(
            _LiveTrack(
                track=track,
                last_frame_index=frame_index,
                last_observation=observation,
            )
        )

    def add_frame(self, frame_index: int, observations: list[FaceObservation]) -> None:
        """Feeds one sampled frame's observations to the tracker.

        Frames must be supplied in chronological order, with `frame_index`
        counting sampled frames (not source frames), since the gap
        allowance is expressed in sampled frames.
        """
        self._close_stale(frame_index)

        if not observations:
            return

        # Score every live-track/observation pairing, then commit them
        # greedily from the strongest overlap down so each track takes at
        # most one detection and each detection joins at most one track.
        # Greedy-by-best-IoU is enough here because faces that overlap each
        # other strongly enough to be confusable are rare in this footage;
        # a full assignment solve would be more machinery than the signal
        # justifies at this stage.
        pairings = []
        for live_index, live in enumerate(self._live):
            for observation_index, observation in enumerate(observations):
                overlap = intersection_over_union(
                    live.last_observation.detection.box, observation.detection.box
                )
                if overlap < self.iou_threshold:
                    continue
                similarity = cosine_similarity(
                    live.last_observation.embedding, observation.embedding
                )
                if similarity < self.contradiction_floor:
                    # Overlapping boxes but an unrelated face: a shot cut
                    # landed someone new in the same part of the frame.
                    continue
                pairings.append((overlap, live_index, observation_index))

        pairings.sort(reverse=True)

        claimed_tracks: set[int] = set()
        claimed_observations: set[int] = set()
        for _, live_index, observation_index in pairings:
            if live_index in claimed_tracks or observation_index in claimed_observations:
                continue
            claimed_tracks.add(live_index)
            claimed_observations.add(observation_index)

            live = self._live[live_index]
            observation = observations[observation_index]
            live.track.observations.append(observation)
            live.last_frame_index = frame_index
            live.last_observation = observation

        for observation_index, observation in enumerate(observations):
            if observation_index not in claimed_observations:
                self._start_track(observation, frame_index)

    def finish(self) -> list[FaceTrack]:
        """Closes all open tracks and returns every track in start-time order."""
        self._closed.extend(live.track for live in self._live)
        self._live = []
        tracks = self._closed
        if self.split_threshold is not None:
            tracks = []
            for track in self._closed:
                for piece in split_track(track, self.split_threshold):
                    tracks.append(FaceTrack(track_id=self._next_track_id, observations=piece))
                    self._next_track_id += 1
            self._closed = tracks
        return sorted(tracks, key=lambda track: (track.start_time, track.track_id))
