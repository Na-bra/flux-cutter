"""Footage one video repeats from another: recaps, intros, reused shots.

A season reel is the same person cut out of every episode, and a lot of
episodes open with the last one's closing minutes, or share a title
sequence. Cut naively, a character in the recap is in the reel twice, and
one in the opening credits is in it once per episode.

**Why frames and not faces.** The scans already hold every face they saw,
so the first attempt looked for repeats there, and could not tell them
apart. On a two-minute recap re-encoded from an episode, each face matched
its original at a median of only 0.82 -- the scan samples every 0.5s from
a different starting point, so it catches each face a fraction of a second
later -- while new footage of the same people reached 0.86. Adding the
time offset between matches and the face's position on screen still
flagged thirteen stretches of an episode that repeats nothing.

A repeat is the same *pixels*, whoever is in them, so this fingerprints the
frames themselves: a 63-bit perceptual hash (the signs of a frame's lowest
DCT frequencies, as pHash does) every 0.5s. On the same footage:

    closest frame, bits apart   median   within 8   within 12
    recap vs its episode           4        72%        84%
    new episode vs the other      18         0%         1%

**Finding a repeat.** Frames within `MAX_BITS` of each other vote for the
time offset between the two videos; the best-supported offset marks the
frames that agree with it, and runs of those, joined across short gaps,
are the repeated stretches. A repeat is one continuous block lifted from
elsewhere, so it has one offset throughout, and new footage has none -- on
the two halves of the test episode not one stretch was found in either
direction, while the recap was found as one stretch at +30.239s (made at
+30.23s) covering 0-119.2s of its 120.

Frames with almost no detail -- black, a flat title card -- look alike in
every video, and are left out of the vote.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from app.video.frames import skip_nonreference_frames
from app.video.timeline import AppearanceInterval

# Seconds between fingerprinted frames: the scans' own default interval.
FINGERPRINT_INTERVAL = 0.5
# Bumped whenever the fingerprint itself changes, so an old cache is not
# compared with a new one.
FINGERPRINT_FORMAT = 1

# How many of 63 bits two frames may differ by and still be one frame.
# 10 found the whole recap and nothing in new footage; 8 and 12 found no
# false stretches either, so this is not on a knife edge.
MAX_BITS = 10
# How far two matches' offsets may differ and still agree. Samples fall up
# to half an interval apart in the two videos.
OFFSET_WINDOW = 0.35
# Agreeing frames further apart than this start a new stretch. A repeat is
# one block, and its fast-moving moments hash less alike than the rest;
# at 2s the recap came back in six pieces, at 5s in one.
JOIN_GAP = 5.0
# A stretch needs this many agreeing frames (3s), and at least this share
# of all the frames it spans, to count.
MIN_FRAMES = 6
MIN_DENSITY = 0.5
# A frame whose 32x32 thumbnail varies less than this carries no detail
# worth matching: black, a fade, a flat card.
MIN_DETAIL = 6.0
# Shorter than this, what a repeat leaves of a segment is dropped too.
MIN_LEFTOVER = 0.5


@dataclass(frozen=True)
class Fingerprints:
    """A video's frames, as hashes, every FINGERPRINT_INTERVAL seconds."""

    times: np.ndarray  # float64 seconds
    hashes: np.ndarray  # uint64
    usable: np.ndarray  # bool: enough detail to match on


@dataclass(frozen=True)
class Repeat:
    """A stretch of one video that repeats another: t here is t + offset there."""

    start: float
    end: float
    offset: float
    frames: int


def _hash(gray32: np.ndarray) -> tuple[int, bool]:
    coefficients = cv2.dct(gray32.astype(np.float32))[:8, :8].flatten()[1:]
    bits = coefficients > np.median(coefficients)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value, float(gray32.std()) >= MIN_DETAIL


def fingerprint(path: Path, interval: float = FINGERPRINT_INTERVAL) -> Fingerprints:
    """Hashes a frame every `interval` seconds. About 13s for 11 minutes of 720p."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        skip_nonreference_frames(stream)
        times, hashes, usable = [], [], []
        due = 0.0
        for frame in container.decode(stream):
            if frame.time is None or frame.time < due:
                continue
            gray = frame.reformat(width=32, height=32, format="gray").to_ndarray()
            value, detailed = _hash(gray)
            times.append(frame.time)
            hashes.append(value)
            usable.append(detailed)
            due = frame.time + interval - 1e-6
    finally:
        container.close()
    return Fingerprints(
        np.asarray(times, dtype=np.float64),
        np.asarray(hashes, dtype=np.uint64),
        np.asarray(usable, dtype=bool),
    )


def _cache_name(path: Path) -> str | None:
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
    except OSError:
        return None
    identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}|{FINGERPRINT_FORMAT}"
    return hashlib.sha256(identity.encode()).hexdigest()


def cached_fingerprint(path: Path, directory: Path) -> Fingerprints | None:
    """A kept fingerprint for this file as it is now, or None."""
    name = _cache_name(path)
    if name is None:
        return None
    try:
        with np.load(directory / f"{name}.npz") as data:
            return Fingerprints(data["times"], data["hashes"], data["usable"])
    except (OSError, KeyError, ValueError):
        return None


def fingerprint_kept(path: Path, directory: Path) -> Fingerprints:
    """Fingerprints a video once, and keeps the result beside the scans.

    Keyed by the file's identity and nothing else: unlike a scan, a
    fingerprint does not depend on any setting.
    """
    kept = cached_fingerprint(path, directory)
    if kept is not None:
        return kept
    made = fingerprint(path)
    name = _cache_name(path)
    if name is not None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f"{name}.tmp.npz"
            np.savez(temporary, times=made.times, hashes=made.hashes, usable=made.usable)
            temporary.replace(directory / f"{name}.npz")
        except OSError:
            pass
    return made


def find_repeats(later: Fingerprints, earlier: Fingerprints) -> list[Repeat]:
    """The stretches of `later` that repeat footage from `earlier`."""
    b = np.flatnonzero(later.usable)
    a = np.flatnonzero(earlier.usable)
    if not len(b) or not len(a):
        return []

    distance = np.bitwise_count(
        np.bitwise_xor(later.hashes[b][:, None], earlier.hashes[a][None, :])
    )
    rows, columns = np.nonzero(distance <= MAX_BITS)
    if not len(rows):
        return []
    when = later.times[b][rows]
    offsets = earlier.times[a][columns] - when

    found: list[Repeat] = []
    while len(offsets):
        best = _best_offset(when, offsets)
        agreeing = np.unique(when[np.abs(offsets - best) <= OFFSET_WINDOW])
        stretches = np.split(agreeing, np.flatnonzero(np.diff(agreeing) > JOIN_GAP) + 1)

        kept = []
        for stretch in stretches:
            spanned = np.count_nonzero(
                (later.times >= stretch[0]) & (later.times <= stretch[-1])
            )
            if len(stretch) >= MIN_FRAMES and len(stretch) / max(1, spanned) >= MIN_DENSITY:
                kept.append(Repeat(float(stretch[0]), float(stretch[-1]), float(best), len(stretch)))
        if not kept:
            break
        found.extend(kept)

        # Everything inside a found stretch is explained; a static shot also
        # matches its own neighbours a frame or two off, and those would
        # otherwise come back as smaller repeats inside this one.
        inside = np.zeros(len(when), dtype=bool)
        for repeat in kept:
            inside |= (when >= repeat.start) & (when <= repeat.end)
        when, offsets = when[~inside], offsets[~inside]

    return sorted(found, key=lambda r: r.start)


def _best_offset(when: np.ndarray, offsets: np.ndarray) -> float:
    """The offset the most distinct frames agree on."""
    order = np.argsort(offsets)
    offsets, when = offsets[order], when[order]
    best, support = float(offsets[0]), 0
    for centre in np.arange(offsets[0], offsets[-1] + 0.05, 0.05):
        lo, hi = np.searchsorted(offsets, [centre - OFFSET_WINDOW, centre + OFFSET_WINDOW], side="left")
        count = len(np.unique(when[lo:hi]))
        if count > support:
            best, support = float(centre), count
    # Refined to the median of the matches it gathered: the grid only says
    # which window holds the most, and on the recap its centre was 0.15s
    # off the true offset -- which would move every cut edge by as much.
    near = np.abs(offsets - best) <= OFFSET_WINDOW
    return float(np.median(offsets[near]))


def _subtract(segments, cuts) -> list[AppearanceInterval]:
    """segments minus cuts, dropping leftovers too short to be worth a cut."""
    result = []
    for segment in segments:
        pieces = [(segment.start_time, segment.end_time)]
        for lo, hi in cuts:
            next_pieces = []
            for start, end in pieces:
                if hi <= start or lo >= end:
                    next_pieces.append((start, end))
                    continue
                if lo > start:
                    next_pieces.append((start, lo))
                if hi < end:
                    next_pieces.append((hi, end))
            pieces = next_pieces
        result.extend(
            AppearanceInterval(start, end) for start, end in pieces if end - start >= MIN_LEFTOVER
        )
    return result


def without_repeats(
    plans: list[tuple[int, list[AppearanceInterval]]],
    repeats: dict[tuple[int, int], list[Repeat]],
) -> tuple[list[tuple[int, list[AppearanceInterval]]], dict[int, float]]:
    """Leaves out of each video what the reel has already shown from an earlier one.

    Args:
        plans: (video, segments) in reel order.
        repeats: (later video, earlier video) -> what later repeats of earlier.

    Only footage the reel really does show earlier is left out. A recap of
    a scene the reel skipped -- the person was not in the earlier cut of
    it, or that episode is not in the reel -- is the only time it appears,
    so it stays.

    Returns:
        The plans, and the seconds left out of each video.
    """
    kept: dict[int, list[AppearanceInterval]] = {}
    result = []
    removed: dict[int, float] = {}
    for video, segments in plans:
        cuts = []
        for earlier, earlier_segments in kept.items():
            for repeat in repeats.get((video, earlier), []):
                for shown in earlier_segments:
                    # The earlier segment, in this video's time.
                    lo = max(repeat.start, shown.start_time - repeat.offset)
                    hi = min(repeat.end + FINGERPRINT_INTERVAL, shown.end_time - repeat.offset)
                    if hi > lo:
                        cuts.append((lo, hi))
        trimmed = _subtract(segments, cuts) if cuts else list(segments)
        before = sum(s.end_time - s.start_time for s in segments)
        after = sum(s.end_time - s.start_time for s in trimmed)
        if before - after > 1e-6:
            removed[video] = before - after
        kept[video] = trimmed
        result.append((video, trimmed))
    return result, removed
