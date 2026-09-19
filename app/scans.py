"""Keeping a scan, so the same video is not scanned twice.

A scan is the expensive thing this app does: on a 22-minute episode the
detect-and-embed pass runs for minutes, and until now every one of them
was thrown away the moment the window closed or the command returned. The
documented workflow made that worse rather than better -- `group` to see
the montage, then `export --select-index 0` to cut a reel, which scanned
the same footage a second time to reach the same answer.

What is stored is the *identities*, not the footage: every observation's
embedding, its box, landmarks and timestamp, and one representative crop
per person for the gallery to draw. The other crops are dropped, because
nothing downstream reads them -- only the representative one becomes a
thumbnail (app/ui/gallery.py).

Freshness is a question about the key, never about the contents. The key
covers the video's identity (path, size, modification time) and every
setting that can change what the scan produces, so a cache hit means the
same scan would have produced the same answer. There is no staleness
check to get wrong, and no way to ask for one thing and be handed
another.

The app's own version is part of that key. Grouping thresholds in this
project have been retuned more than once against real footage, and a
cached scan from before such a change is not merely old, it is wrong. A
release therefore invalidates every entry, which costs a rescan and buys
never silently serving an answer the current code would not give.
"""

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import app
from app.faces.detector import BoundingBox, FaceDetection, FaceLandmarks
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.models import cache_dir

# Bumped when the stored layout changes in a way an older reader would
# misread. Entries written by another version are ignored, not repaired.
# Unchanged by the move to SCAN_FORMAT below: the file's layout is the
# same, only the name it is filed under changed -- which is what lets an
# entry written by 1.9.x still be read and re-keyed rather than rebuilt.
CACHE_VERSION = 1

# Bumped when the pipeline would give a *different answer* -- a change to
# detection, embedding, tracking or grouping, or to any default those use.
# It is not the app's version, and that distinction is the whole point of
# it: 1.9.1, 1.9.2 and 1.9.3 changed only how audio is cut, yet keying on
# the app version made each of them discard every kept scan, and with it
# every name, merge, split and discard a person had made. Nothing about
# cutting audio can change who is in a video.
#
# Raise this deliberately when the identity pipeline changes. Leaving it
# alone through a release that alters grouping is the one mistake it
# cannot catch for you, so the release that does that says so in its
# commit message.
SCAN_FORMAT = 1

# App versions whose entries were keyed on the version itself, newest
# first. A scan kept by one of these is found and re-keyed rather than
# rebuilt -- see `load`. Nothing is added here again: releases after 1.9.3
# key on SCAN_FORMAT, which does not move unless the answer would.
LEGACY_APP_VERSIONS = ("1.9.3", "1.9.2", "1.9.1", "1.9.0", "1.8.0", "1.7.0")

# Scans are large and worth keeping, but not without limit. Oldest first,
# by modification time, until the total is back under this.
MAX_CACHE_BYTES = 2 * 1024**3


def scan_cache_dir() -> Path:
    """Where kept scans live.

    FLUXCUTTER_SCAN_DIR overrides it, which is what the tests use. It sits
    beside the model cache and so follows FLUXCUTTER_MODEL_DIR when that is
    set -- one override moves the whole per-user directory, which is the
    behaviour a shared machine or a read-only volume wants.
    """
    override = os.environ.get("FLUXCUTTER_SCAN_DIR")
    if override:
        return Path(override).expanduser()
    return cache_dir().parent / "scans"


def cache_key(
    video_path: Path,
    *,
    sample_interval: float,
    confidence_threshold: float,
    padding_ratio: float,
    similarity_threshold: float,
    margin_threshold: float,
    consolidation_threshold: float,
    min_confidence: float,
    min_face_size: int,
    min_group_eye_span: float,
    mode: str,
    forbid_cooccurring: bool,
    cooccurrence_similarity_ceiling: float,
    min_detections: int | None,
    skip_nonreference: bool = True,
    _stamp: str | None = None,
) -> str:
    """Identifies one scan of one video under one set of settings.

    Every argument is required and named, deliberately. A caller that
    forgets one gets a TypeError rather than a key that quietly differs
    from the other caller's, which is what would stop the window and the
    command line sharing a cache entry for the same work.

    The video is identified by size and modification time rather than by
    hashing its contents: hashing an 815 MB file to avoid re-reading it is
    a poor trade, and an edit that preserved both would have to be
    deliberate.

    `_stamp` replaces the scan-format marker and exists only so `load` can
    rebuild the key an older release would have written. No caller outside
    this module passes it.
    """
    path = Path(video_path).resolve()
    try:
        stat = path.stat()
        identity = [str(path), str(stat.st_size), str(stat.st_mtime_ns)]
    except OSError:
        # A video that cannot be stat'd cannot be identified, so it gets a
        # key nothing will ever match rather than one that collides.
        identity = [str(path), "unstattable", str(time.time_ns())]

    parts = identity + [
        f"v{CACHE_VERSION}",
        # What the answer depends on, not what the app happens to be
        # called this week.
        _stamp if _stamp is not None else f"scan{SCAN_FORMAT}",
        f"{sample_interval!r}",
        f"{confidence_threshold!r}",
        f"{padding_ratio!r}",
        f"{similarity_threshold!r}",
        f"{margin_threshold!r}",
        f"{consolidation_threshold!r}",
        f"{min_confidence!r}",
        f"{min_face_size!r}",
        f"{min_group_eye_span!r}",
        mode,
        f"{forbid_cooccurring!r}",
        f"{cooccurrence_similarity_ceiling!r}",
        f"{min_detections!r}",
        f"{skip_nonreference!r}",
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def find(video_path: Path, **settings) -> tuple[str, "CachedScan | None"]:
    """The key to file this scan under, and whatever is already kept for it.

    Callers ask for both at once because the second question depends on
    more than the first. A scan kept by an older release was filed under a
    key built from that release's version, so a miss on the current key is
    not proof there is nothing: the older names are tried, and anything
    found is re-filed under the current key and the stale copy removed.

    That migration runs once per video. It moves a file; it does not
    rescan, and the names and corrections inside come across with it.
    """
    key = cache_key(video_path, **settings)

    kept = load(key)
    if kept is not None:
        return key, kept

    for version in LEGACY_APP_VERSIONS:
        stale = cache_key(video_path, _stamp=version, **settings)
        older = load(stale)
        if older is None:
            continue
        try:
            save(key, older)
            (scan_cache_dir() / f"{stale}.npz").unlink(missing_ok=True)
        except OSError:
            # Re-filing is a convenience; the scan itself is already in
            # hand, and failing to tidy up must not lose it.
            pass
        return key, older

    return key, None


@dataclass
class CachedScan:
    """One scan's identities and the statistics that described the run.

    This is what both front ends need and neither can cheaply recompute:
    the groups themselves, plus the counts the reports print. It is not a
    PipelineResult -- the grouper that produced it is gone, and rebuilding
    one to hold groups it did not cluster would be a lie about where they
    came from.
    """

    groups: list[FaceIdentityGroup] = field(default_factory=list)
    unassigned_count: int = 0
    total_detections: int = 0
    track_count: int = 0
    frame_count: int = 0
    last_timestamp: float = 0.0
    embedding_time: float = 0.0
    grouping_time: float = 0.0
    video_duration: float | None = None
    # The resolved screen-time cutoff this scan's groups were filtered
    # by, kept because it is reported and cannot be recomputed without
    # the duration and interval that produced it.
    min_detections: int = 0
    # When the scan ran, and how long it took -- reported when a scan is
    # reused so the saving is visible rather than merely felt.
    created_at: float = 0.0
    scan_seconds: float = 0.0
    # True once a person has merged, split or discarded a card. The
    # key still identifies the scan that produced this, but the groups
    # are no longer only what the clustering said -- and that is the
    # point: a correction the user made must outlive the window.
    edited: bool = False


# ------------------------------------------------------------- writing it


def _landmarks_to_list(landmarks: FaceLandmarks | None):
    if landmarks is None:
        return None
    return [float(value) for value in landmarks.as_tuple()]


def _landmarks_from_list(values):
    if values is None:
        return None
    points = [(float(values[i]), float(values[i + 1])) for i in range(0, 10, 2)]
    return FaceLandmarks(*points)


def save(key: str, scan: CachedScan) -> Path:
    """Stores a scan under its key, replacing any entry already there.

    Written to a temporary file and moved into place, so an interrupted
    write leaves the previous entry intact rather than a half-file that
    the next run would have to detect and discard.
    """
    directory = scan_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{key}.npz"

    embeddings = []
    arrays = {}
    manifest_groups = []

    for position, group in enumerate(scan.groups):
        observations = []
        for observation in group.observations:
            embeddings.append(np.asarray(observation.embedding, dtype=np.float32))
            observations.append(
                {
                    # Everything is coerced to a plain Python number on
                    # the way in. The detector and the sharpness measure
                    # both hand back numpy scalars, which json refuses --
                    # and refuses at the very end of a scan that has
                    # already cost minutes.
                    "box": [
                        int(observation.detection.box.x_min),
                        int(observation.detection.box.y_min),
                        int(observation.detection.box.x_max),
                        int(observation.detection.box.y_max),
                    ],
                    "confidence": float(observation.detection.confidence),
                    "landmarks": _landmarks_to_list(observation.detection.landmarks),
                    "timestamp": float(observation.source_timestamp),
                    "frame_index": (
                        None
                        if observation.frame_index is None
                        else int(observation.frame_index)
                    ),
                    "sharpness": (
                        None
                        if observation.sharpness is None
                        else float(observation.sharpness)
                    ),
                    "space": observation.embedding_space,
                }
            )

        representative = group.representative_observation
        representative_index = None
        if representative is not None:
            # Found by identity, never by ==. FaceObservation is a frozen
            # dataclass holding numpy arrays, so its generated __eq__
            # compares embeddings elementwise and returns an array;
            # list.index() would then raise "truth value is ambiguous" on
            # the first non-matching member it tried.
            representative_index = next(
                (
                    index
                    for index, observation in enumerate(group.observations)
                    if observation is representative
                ),
                None,
            )
            if representative_index is not None:
                arrays[f"crop_{position}"] = np.asarray(
                    representative.face_crop, dtype=np.uint8
                )

        centroid = group.representative_embedding
        if centroid is not None:
            arrays[f"centroid_{position}"] = np.asarray(centroid, dtype=np.float32)

        manifest_groups.append(
            {
                "group_id": group.group_id,
                "representative": representative_index,
                # Without these a reopened scan cannot be split: track
                # boundaries are not recoverable from the observations
                # once they have been concatenated.
                "unit_sizes": [int(size) for size in group.unit_sizes],
                "name": group.name,
                "observations": observations,
            }
        )

    manifest = {
        "version": CACHE_VERSION,
        "app_version": app.__version__,
        "created_at": scan.created_at or time.time(),
        "scan_seconds": scan.scan_seconds,
        "unassigned_count": int(scan.unassigned_count),
        "total_detections": int(scan.total_detections),
        "track_count": int(scan.track_count),
        "frame_count": int(scan.frame_count),
        "last_timestamp": float(scan.last_timestamp),
        "embedding_time": float(scan.embedding_time),
        "grouping_time": float(scan.grouping_time),
        "video_duration": (
            None if scan.video_duration is None else float(scan.video_duration)
        ),
        "min_detections": int(scan.min_detections),
        "edited": bool(scan.edited),
        "groups": manifest_groups,
    }

    arrays["manifest"] = np.array(json.dumps(manifest))
    arrays["embeddings"] = (
        np.stack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)
    )

    temporary = destination.with_suffix(".npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)

    # Every name reaches the people library the moment it reaches a scan,
    # so naming, renaming and clearing need nothing of their own there.
    from app.faces import library

    library.remember(key, scan.groups)

    prune()
    return destination


# ------------------------------------------------------------- reading it


def load(key: str) -> CachedScan | None:
    """Returns the scan stored under this key, or None if there is none.

    A missing entry and an unreadable one are the same answer: rescan.
    Nothing here raises, because every caller's fallback is to do the work
    -- turning a corrupt cache file into a failed scan would make this
    feature a liability rather than a saving. A file that cannot be read
    is deleted on the way past so it stops being tried.
    """
    path = scan_cache_dir() / f"{key}.npz"
    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as stored:
            manifest = json.loads(str(stored["manifest"]))
            if manifest.get("version") != CACHE_VERSION:
                return None

            embeddings = stored["embeddings"]
            groups = []
            cursor = 0
            for position, entry in enumerate(manifest["groups"]):
                observations = []
                for record in entry["observations"]:
                    box = BoundingBox(*(int(value) for value in record["box"]))
                    observations.append(
                        FaceObservation(
                            embedding=embeddings[cursor],
                            detection=FaceDetection(
                                box=box,
                                confidence=record["confidence"],
                                landmarks=_landmarks_from_list(record["landmarks"]),
                            ),
                            # Only the representative's crop is kept; see
                            # the module docstring. Filled in below.
                            face_crop=None,
                            source_timestamp=record["timestamp"],
                            frame_index=record["frame_index"],
                            sharpness=record["sharpness"],
                            embedding_space=record["space"],
                        )
                    )
                    cursor += 1

                group = FaceIdentityGroup(
                    group_id=entry["group_id"],
                    observations=observations,
                    unit_sizes=list(entry.get("unit_sizes", [])),
                    name=entry.get("name"),
                )
                centroid_key = f"centroid_{position}"
                if centroid_key in stored:
                    group.representative_embedding = stored[centroid_key]

                representative_index = entry["representative"]
                crop_key = f"crop_{position}"
                if representative_index is not None and crop_key in stored:
                    # Replaced rather than mutated: the observation is a
                    # frozen dataclass, and the copy has to go back into
                    # the list too so the group's representative is the
                    # same object the group holds.
                    representative = replace(
                        observations[representative_index], face_crop=stored[crop_key]
                    )
                    observations[representative_index] = representative
                    group.observations = observations
                    group.representative_observation = representative

                groups.append(group)
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None

    return CachedScan(
        groups=groups,
        unassigned_count=manifest["unassigned_count"],
        total_detections=manifest["total_detections"],
        track_count=manifest["track_count"],
        frame_count=manifest["frame_count"],
        last_timestamp=manifest["last_timestamp"],
        embedding_time=manifest["embedding_time"],
        grouping_time=manifest["grouping_time"],
        video_duration=manifest["video_duration"],
        min_detections=manifest.get("min_detections", 0),
        edited=manifest.get("edited", False),
        created_at=manifest["created_at"],
        scan_seconds=manifest["scan_seconds"],
    )


# ------------------------------------------------------------- managing it


@dataclass(frozen=True)
class CacheEntry:
    """One stored scan, as the `scans` command reports it."""

    path: Path
    size_bytes: int
    modified: float


def entries() -> list[CacheEntry]:
    """Every stored scan, newest first."""
    directory = scan_cache_dir()
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob("*.npz"):
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append(
            CacheEntry(path=path, size_bytes=stat.st_size, modified=stat.st_mtime)
        )
    return sorted(found, key=lambda entry: entry.modified, reverse=True)


def total_bytes() -> int:
    return sum(entry.size_bytes for entry in entries())


def prune(max_bytes: int = MAX_CACHE_BYTES) -> int:
    """Deletes the oldest scans until the cache fits. Returns how many went."""
    kept = entries()
    total = sum(entry.size_bytes for entry in kept)
    removed = 0
    for entry in reversed(kept):
        if total <= max_bytes:
            break
        try:
            entry.path.unlink()
        except OSError:
            continue
        total -= entry.size_bytes
        removed += 1
    return removed


def clear() -> int:
    """Deletes every stored scan. Returns how many went."""
    directory = scan_cache_dir()
    if not directory.is_dir():
        return 0
    count = len(list(directory.glob("*.npz")))
    shutil.rmtree(directory, ignore_errors=True)
    return count
