from dataclasses import dataclass, field

import numpy as np

from app.faces.detector import intersection_over_union
from app.faces.grouper import FaceObservation, cosine_similarity, mean_embedding

# Minimum box overlap between a live track's last detection and a candidate
# detection in the next sampled frame before the two are linked. 0.3 is the
# conventional tracking-by-detection value and held up on the real test
# footage: faces move far enough between samples that anything stricter
# breaks tracks mid-shot, while anything looser starts linking across
# neighbouring faces in two-shots.
DEFAULT_IOU_THRESHOLD = 0.3
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
        return sorted(self._closed, key=lambda track: (track.start_time, track.track_id))
