"""Where one camera shot ends and the next begins.

A shot is a continuous stretch of footage between two cuts. Nothing here
knows about faces: shots are a property of the footage, found from the
same sampled frames a scan reads, so a video can be divided into shots
without detecting anyone in it. The appearance timeline does not use them
yet; this is the foundation it will stand on.

**Why samples, not every frame.** The cutter already finds cuts frame by
frame (`_cuts` in app/video/cutter.py), but only across the few frames at
a segment's edge. Doing that for a whole episode means decoding and
thumbnailing every one of 32,508 frames -- 84s for the 22-minute test
episode. A scan already holds a frame every half second with its real
timestamp (app/video/frames.py), so shots are found from those, and later
from the scan's own pass rather than a second one. The price is precision:
a cut is known to lie between two samples, not on a frame. Each shot keeps
both ends of that window (`earliest_start`, `start_time`).

**What counts as a cut.** Each sample is shrunk to a 64x36 colour
thumbnail, the size the cutter uses, and compared with the one before in
two ways:

- the distance between their hue-saturation histograms (Bhattacharyya),
  which a camera move or a person walking across the frame barely changes,
  since the same colours stay in the picture;
- the mean difference of their greyscale pictures with each frame's own
  brightness and contrast taken out, which a shift of colour balance
  within a shot barely changes.

Half a second apart, frames within one shot differ far more than
neighbouring frames do, so the cutter's own thresholds do not carry over.
These were measured instead, against two full-rate references on the
22-minute live-action episode -- the cutter's detector run over every frame
and ffmpeg's `scdet`, which agree on 97% of ffmpeg's cuts:

    rule                                      recall   precision
    pixel difference >= 32                    97.4%      72.0%
    histogram distance >= 0.28                97.1%      80.4%
    histogram, 1.3x its neighbours, pixels    96.7%      88.3%
    ... or histogram >= 0.5 regardless        98.5%      86.5%
    ... and shots under 0.75s merged          96.2%      92.3%

A 23-second fan edit then showed what that missed. It is dark and cut
every half second to second and a half, and the rule found 7 of its 16 cuts
(labelled by eye from its samples). Two things were in the way:

- raw pixel differences: at night everything is near black, so a real cut
  moved them by 20-24, just under the limit. Taking each frame's brightness
  and contrast out first fixed that without changing the episode at all;
- the median of the neighbours as "typical": in rapid cutting half the
  neighbouring gaps are cuts, so nothing stood out. The 35th percentile --
  the quieter neighbours -- found 13 of 16 before merging, still with no
  false cut there, for three points of precision on the episode.

The module as it stands, reading through `extract_frames`:

    episode (22 min, live action)   recall 96.2%  precision 90.7%  560 shots
    fan edit (23 s, dark, fast)     12 of 16 cuts, none false
    animation (7 min)               precision 91.2% against the references

The unconditional 0.5 exists for rapid-fire cutting too -- a title montage,
two cuts half a second apart -- where every neighbour is a cut and nothing
stands out. Most of what the references call false positives are real
transitions they cannot see, because they only look for a jump between
consecutive frames: wipes, fades from black, animated title cards. The
genuine errors are effect flashes and a light going off within a shot.

On animation the references themselves disagree (half of each one's cuts
are not in the other; both fire on flicker, fire and impact flashes), so
the shared rule was judged by eye there: of twelve cuts it found at random,
about ten were cuts or real transitions and about two were effects inside
one shot. That did not justify a separate animation detector.

**How often to sample.** Recall falls with the interval, because a shot
shorter than the gap between two samples cannot be seen at all: 98% at
0.25s, 96% at 0.5s, 83% at 1s and 64% at 2s on the test episode, whose
median shot lasts 2.1s. `SHOT_SAMPLE_INTERVAL` is therefore half a second
whatever interval a scan uses.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.video.timeline import format_timestamp

# The sampling a shot pass reads at, whatever interval a scan uses. See
# "How often to sample" above.
SHOT_SAMPLE_INTERVAL = 0.5

# The size frames are compared at: the cutter's own thumbnail.
THUMB_SIZE = (64, 36)
# Hue and saturation bins. Brightness is left out, so a dimmer moment of
# the same shot does not look like another shot.
HISTOGRAM_BINS = (32, 16)

# A cut's change must be this many times the typical change around it...
NEIGHBOUR_RATIO = 1.3
# ...measured over this many gaps either side...
NEIGHBOURS = 4
# ...as this percentile of them: the quieter ones, so that in rapid cutting,
# where half the neighbouring gaps are cuts too, "typical" is still the
# footage within a shot rather than the cuts either side.
TYPICAL_PERCENTILE = 35
# A change this large is a cut whatever surrounds it: rapid cutting leaves
# nothing for a cut to stand out from.
CERTAIN_CHANGE = 0.5
# And the picture must have changed at least this much, so a shift of
# colour balance within a shot is not a cut. Measured on greyscale
# thumbnails with each frame's brightness and contrast evened out (mean 0,
# spread 1), as a mean per pixel: a dark scene's cut moves raw pixel values
# very little, because everything is near black, and read raw a night-time
# cut fell short of the old limit (24 of 255) at 20-24.
MIN_PICTURE_CHANGE = 0.5
# Contrast is never evened out beyond this spread (0-255), so a nearly
# flat frame -- black, a fade -- is not blown up into noise.
MIN_SPREAD = 8.0


@dataclass(frozen=True)
class ShotSettings:
    """The few things worth changing about how shots are found.

    Attributes:
        enabled: Off, a video is one shot from start to end -- no
            boundaries, so anything built on shots behaves as it would
            without them.
        threshold: How far apart two samples' colour histograms must be
            (Bhattacharyya distance, 0-1) to be a cut when the change also
            stands out from its neighbours. Lower finds more cuts and more
            false ones.
        min_shot_seconds: A shot shorter than this is merged away. At
            half-second sampling that removes one-sample "shots", which are
            mostly a flash going off and on again: where the footage either
            side matches, both cuts go and it stays one shot. It also merges
            a genuinely brief cutaway that returns to the same shot, which
            cost 0.7 points of recall on the test episode.
    """

    enabled: bool = True
    threshold: float = 0.30
    min_shot_seconds: float = 0.75


@dataclass(frozen=True)
class Shot:
    """One continuous stretch of footage between two cuts.

    Shots tile the video: each starts where the one before it ends, the
    first at 0 and the last at the video's end. `start_time` is the first
    sample that shows this shot; the cut itself happened after
    `earliest_start`, the last sample of the shot before -- sampling cannot
    say more precisely than that.
    """

    index: int
    start_time: float
    end_time: float
    earliest_start: float = 0.0
    # How different this shot's first sample was from the last one before
    # it (histogram distance, 0-1). 0 for the first shot.
    change: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)


def _evened(thumb: np.ndarray) -> np.ndarray:
    """Greyscale, with the frame's own brightness and contrast taken out."""
    grey = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return (grey - grey.mean()) / max(float(grey.std()), MIN_SPREAD)


def _histogram(thumb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, list(HISTOGRAM_BINS), [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


@dataclass
class ShotDetector:
    """Finds shots from sampled frames fed to it in order.

    Incremental, so a scan can feed it the frames it is already reading
    rather than decoding the video a second time. It keeps each sample's
    colour histogram (2 KB) and the previous greyscale thumbnail, never the frames
    themselves, so its memory does not grow with the picture size. The decision waits for
    `finish`, because whether a change stands out depends on the gaps after
    it as well as before.
    """

    settings: ShotSettings = field(default_factory=ShotSettings)
    _times: list[float] = field(default_factory=list)
    _histogram_changes: list[float] = field(default_factory=list)
    _pixel_changes: list[float] = field(default_factory=list)
    _histograms: list[np.ndarray] = field(default_factory=list)
    _previous: tuple[np.ndarray, np.ndarray] | None = None

    def add(self, timestamp: float, frame: np.ndarray) -> None:
        """Takes the next sample: its source timestamp and its RGB pixels.

        Raises:
            ValueError: If the timestamp goes backwards. Shots are built from
                the order of the samples, so an out-of-order one would put a
                boundary in the wrong place without anything noticing.
        """
        if self._times and timestamp < self._times[-1]:
            raise ValueError(
                f"samples must be in time order: {timestamp} came after {self._times[-1]}"
            )
        self._times.append(float(timestamp))
        if not self.settings.enabled:
            return
        thumb = _thumbnail(frame)
        histogram = _histogram(thumb)
        picture = _evened(thumb)
        self._histograms.append(histogram)
        if self._previous is not None:
            previous_picture, previous_histogram = self._previous
            self._histogram_changes.append(
                float(cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA))
            )
            self._pixel_changes.append(float(np.abs(picture - previous_picture).mean()))
        self._previous = (picture, histogram)

    def finish(self, video_duration: float | None = None) -> list[Shot]:
        """The shots, in order, covering the video from 0 to its end.

        Args:
            video_duration: Where the last shot ends. Without it, the last
                shot ends at the last sample.
        """
        if not self._times:
            if video_duration and video_duration > 0:
                return [Shot(index=0, start_time=0.0, end_time=float(video_duration))]
            return []
        end = max(float(video_duration), self._times[-1]) if video_duration else self._times[-1]
        cuts = self._cuts() if self.settings.enabled else []
        cuts = self._without_short_shots(cuts, end)

        starts = [0.0] + [self._times[gap + 1] for gap in cuts]
        ends = starts[1:] + [end]
        return [
            Shot(
                index=position,
                start_time=start,
                end_time=stop,
                earliest_start=self._times[cuts[position - 1]] if position else 0.0,
                change=self._histogram_changes[cuts[position - 1]] if position else 0.0,
            )
            for position, (start, stop) in enumerate(zip(starts, ends))
        ]

    def _cuts(self) -> list[int]:
        """Gaps (between sample i and i+1) where a new shot starts."""
        changes = self._histogram_changes
        cuts = []
        for gap, change in enumerate(changes):
            if self._pixel_changes[gap] < MIN_PICTURE_CHANGE:
                continue
            if change >= CERTAIN_CHANGE:
                cuts.append(gap)
                continue
            if change < self.settings.threshold:
                continue
            around = changes[max(0, gap - NEIGHBOURS) : gap] + changes[gap + 1 : gap + 1 + NEIGHBOURS]
            typical = float(np.percentile(around, TYPICAL_PERCENTILE)) if around else 0.0
            if change >= NEIGHBOUR_RATIO * max(typical, 1e-3):
                cuts.append(gap)
        return cuts

    def _without_short_shots(self, cuts: list[int], end: float) -> list[int]:
        """Drops cuts until no shot is shorter than the minimum.

        A short shot with the same footage either side of it is a flash --
        a jump away and straight back -- and both its cuts go, so the
        footage around it stays one shot. Otherwise it is a brief shot or a
        stray cut, and the weaker of its two cuts goes, joining it to the
        neighbour it is more like. The video's own start and end are not
        cuts and cannot be dropped.
        """
        minimum = self.settings.min_shot_seconds
        cuts = list(cuts)
        while minimum > 0 and cuts:
            bounds = [0.0] + [self._times[gap + 1] for gap in cuts] + [end]
            short = next(
                (k for k in range(len(bounds) - 1) if bounds[k + 1] - bounds[k] < minimum - 1e-9),
                None,
            )
            if short is None:
                break
            around = [j for j in (short - 1, short) if 0 <= j < len(cuts)]
            if len(around) == 2 and self._same_footage(cuts[short - 1], cuts[short] + 1):
                del cuts[short - 1 : short + 1]
                continue
            weakest = min(around, key=lambda j: self._histogram_changes[cuts[j]])
            del cuts[weakest]
        return cuts

    def _same_footage(self, before: int, after: int) -> bool:
        """Whether two samples look like one shot, by the rule a cut is found by."""
        distance = float(
            cv2.compareHist(self._histograms[before], self._histograms[after], cv2.HISTCMP_BHATTACHARYYA)
        )
        return distance < self.settings.threshold


def detect_shots(
    frames: Iterable[tuple[float, np.ndarray]],
    video_duration: float | None = None,
    settings: ShotSettings | None = None,
) -> list[Shot]:
    """Shots from (timestamp, RGB frame) samples, as `extract_frames` yields them."""
    detector = ShotDetector(settings or ShotSettings())
    for timestamp, frame in frames:
        detector.add(timestamp, frame)
    return detector.finish(video_duration)


def shots_of_video(
    video_path: str | Path,
    sample_interval: float = SHOT_SAMPLE_INTERVAL,
    settings: ShotSettings | None = None,
) -> list[Shot]:
    """Reads a video through the scan's own frame pipeline and finds its shots.

    Timestamps are the decoded frames' own (see app/video/frames.py), never
    a sample count times an interval, so a video that starts late or runs at
    a variable rate still gets its boundaries at the right moments.
    """
    from app.video.frames import extract_frames
    from app.video.loader import get_video_info, load_video

    with load_video(video_path) as container:
        duration = get_video_info(container)["duration"]
        if duration is None and container.duration:
            # Matroska and WebM keep a duration for the file, not the stream.
            duration = container.duration / 1_000_000
        return detect_shots(
            extract_frames(container, sample_interval=sample_interval),
            video_duration=duration,
            settings=settings,
        )


def describe(shots: list[Shot]) -> list[str]:
    """One line per shot, for reading: `Shot 1: 00:00.00 → 00:12.40 (12.40s)`."""
    return [
        f"Shot {shot.index + 1}: {format_timestamp(shot.start_time)} → "
        f"{format_timestamp(shot.end_time)} ({shot.duration:.2f}s)"
        for shot in shots
    ]
