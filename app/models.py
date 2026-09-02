"""Finding the ONNX models, and fetching them the first time they are needed.

The two models are 166 MB and 224 KB. Shipping them inside the repository
was fine while this was only ever run from a checkout; it stops being fine
as soon as anything is distributed, where they are half the download and
most of it is a file that never changes.

So they are fetched on first use instead, and this module is the single
place that knows where they live and how to get them.

Three things this is careful about, each for a reason that has already bitten
this project or would obviously bite a user:

- **Checksums are verified, not trusted.** A model was once downloaded
  through a text-mode round trip and arrived at 70 MB instead of 38 MB with
  16 million replacement characters in it. It failed as five confusing test
  errors rather than as a download error, because nothing checked. A partial
  or tampered file must be rejected here, where the message can say so.

- **The write is atomic.** The download goes to a .part file and is renamed
  only after its hash matches, so an interrupted download cannot be mistaken
  for a finished one on the next run. This is what makes an aborted fetch
  safe to simply retry.

- **The cache is outside the app.** A frozen .app in /Applications may sit on
  a read-only volume and should not be written to in any case, so downloads
  go to the user's own data directory.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class ModelDownloadError(Exception):
    """Raised when a model cannot be fetched or fails verification."""


@dataclass(frozen=True)
class ModelSpec:
    """One model: what it is called, where it comes from, and what it must be."""

    filename: str
    url: str
    sha256: str
    size_bytes: int
    description: str

    @property
    def megabytes(self) -> float:
        return self.size_bytes / 1_000_000

    @property
    def size_label(self) -> str:
        """A size a person can read. YuNet rounds to 0 MB, which reads as a bug."""
        if self.size_bytes < 1_000_000:
            return f"{self.size_bytes / 1_000:.0f} KB"
        return f"{self.megabytes:.0f} MB"


# The ArcFace weights ship inside InsightFace's `buffalo_l` bundle, which is
# 288,621,354 bytes to extract a 174,383,860 byte file. Fetching the bundle
# would mean a user downloads 275 MB to keep 166 MB -- worse than shipping it
# and not worth doing. The file is mirrored standalone on Hugging Face, and
# `content-length` there matches our own copy exactly.
#
# A mirror is a supply-chain question, which is the other reason the hash
# below is pinned: it is this project's copy of the model that is authorised,
# not whatever that URL happens to serve. If the mirror ever changes what it
# returns, verification fails and nothing is loaded.
MODELS = {
    "detector": ModelSpec(
        filename="face_detection_yunet_2026may.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2026may.onnx"
        ),
        sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        size_bytes=229_738,
        description="YuNet face detector",
    ),
    "embedder": ModelSpec(
        filename="face_recognition_arcface_w600k_r50.onnx",
        url=(
            "https://huggingface.co/public-data/insightface/resolve/main/"
            "models/buffalo_l/w600k_r50.onnx"
        ),
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        size_bytes=174_383_860,
        description="ArcFace w600k_r50 recognition model",
    ),
    # --- animation mode -------------------------------------------------
    #
    # Neither of these is the model the brief named, and the reasons are
    # recorded in Instructions.md 17. In short: the anime ViT weights the
    # reference project uses are gone -- both its Google Drive ids return a
    # hard 404 -- and its Faster R-CNN detector needs PyTorch, which is 529 MB
    # installed against onnxruntime's 80 MB. These two are ONNX, live, and
    # measured on the actual test footage.
    "anime_detector": ModelSpec(
        filename="anime_face_detection_v1.1_s.onnx",
        url=(
            "https://huggingface.co/deepghs/anime_face_detection/resolve/main/"
            "face_detect_v1.1_s/model.onnx"
        ),
        sha256="5ac333ce11805828f25d7abfaba543d4efbac4c1c68b82d0ec3f2890271b8df5",
        size_bytes=44_583_229,
        description="Anime face detector (YOLO, deepghs, MIT)",
    ),
    "anime_embedder": ModelSpec(
        filename="anime_character_ccip_caformer_24.onnx",
        url=(
            "https://huggingface.co/deepghs/ccip_onnx/resolve/main/"
            "ccip-caformer-24-randaug-pruned/model_feat.onnx"
        ),
        sha256="4ea118d16496274f4f6e08d3afc768cc592389e8f7f32f8732ce2215c228ac5f",
        size_bytes=150_248_245,
        description="CCIP anime character embedding (deepghs, OpenRAIL-M)",
    ),
}

_REPOSITORY_MODELS = Path(__file__).resolve().parents[1] / "assets" / "models"
_DOWNLOAD_CHUNK = 1024 * 256


def cache_dir() -> Path:
    """Where downloaded models are kept, per platform convention.

    FLUXCUTTER_MODEL_DIR overrides it, which is what a test or a shared
    machine wants.
    """
    override = os.environ.get("FLUXCUTTER_MODEL_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base / "FluxCutter" / "models"


def _search_paths(spec: ModelSpec) -> list[Path]:
    """Where a model might already be, best first.

    The repository copy wins so a checkout with models already in it keeps
    working untouched, and so a developer is never made to re-download what
    they have.
    """
    return [_REPOSITORY_MODELS / spec.filename, cache_dir() / spec.filename]


def find_model(spec: ModelSpec) -> Path | None:
    """Returns an existing copy of the model, or None if it must be fetched."""
    for candidate in _search_paths(spec):
        if candidate.is_file():
            return candidate
    return None


def file_sha256(path: Path) -> str:
    """Hashes a file in chunks; these are too big to read into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(spec: ModelSpec, on_progress=None) -> Path:
    """Fetches one model into the cache and verifies it.

    Args:
        spec: Which model to fetch.
        on_progress: Called as (fraction, bytes_done, bytes_total). Fraction
            is 0.0 while the server declines to say how long the body is.

    Returns:
        The path to the verified file.

    Raises:
        ModelDownloadError: If the download fails, or if what arrived does
            not hash to what was expected.
    """
    destination = cache_dir() / spec.filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Downloading beside the destination keeps the rename on one filesystem,
    # which is what makes it atomic.
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f"{spec.filename}.", suffix=".part"
    )
    os.close(handle)
    temporary = Path(temporary_name)

    try:
        request = urllib.request.Request(
            spec.url, headers={"User-Agent": "FluxCutter"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            declared = response.headers.get("Content-Length")
            total = int(declared) if declared else spec.size_bytes
            downloaded = 0

            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if on_progress is not None:
                        fraction = min(1.0, downloaded / total) if total else 0.0
                        on_progress(fraction, downloaded, total)

    except (urllib.error.URLError, OSError, TimeoutError) as error:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Could not download the {spec.description} from {spec.url}: {error}"
        ) from error
    except BaseException:
        # Includes the cancellation the UI raises through on_progress. A
        # half-written file must never survive to look like a whole one.
        temporary.unlink(missing_ok=True)
        raise

    # Checked before the hash so a truncated transfer says so, rather than
    # being reported as the wrong file. This is not hypothetical: fetching
    # this exact model over this exact URL, curl returned a 135,783,125 byte
    # body for a 174,383,860 byte file and exited 0. A server closing the
    # connection early looks identical to a finished download unless someone
    # counts.
    actual_size = temporary.stat().st_size
    if actual_size != total:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"The download of the {spec.description} stopped early "
            f"({human_size(actual_size)} of {human_size(total)}). "
            "Nothing was kept; running the same command again will retry it."
        )

    actual = file_sha256(temporary)
    if actual != spec.sha256:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"The {spec.description} downloaded from {spec.url} is not the "
            f"expected file (sha256 {actual}, expected {spec.sha256}). "
            "It was discarded rather than used."
        )

    # Only now does it get the real name, so an interrupted run leaves
    # nothing that a later run would mistake for a finished download.
    temporary.replace(destination)
    return destination


def ensure_model(spec: ModelSpec, on_progress=None) -> Path:
    """Returns the model's path, downloading it first if it is not here yet."""
    existing = find_model(spec)
    if existing is not None:
        return existing
    return download_model(spec, on_progress=on_progress)


def human_size(num_bytes: int) -> str:
    """Bytes at whatever scale reads sensibly. 230 KB, not 0 MB."""
    if num_bytes < 1_000_000:
        return f"{num_bytes / 1_000:.0f} KB"
    return f"{num_bytes / 1_000_000:.0f} MB"


_PROGRESS_WIDTH = 72


def _print_progress(spec: ModelSpec):
    """A terminal progress line for the CLI, rewritten in place."""
    state = {"last": -1}

    def report(fraction: float, done: int, total: int) -> None:
        percent = int(fraction * 100)
        if percent == state["last"]:
            return
        state["last"] = percent
        bar = "#" * (percent // 4)
        line = (
            f"  [{bar:<25}] {percent:3d}%  "
            f"{human_size(done)} of {human_size(total)}"
        )
        print(f"\r{line:<{_PROGRESS_WIDTH}}", end="", flush=True)

    return report


def ensure_model_cli(spec: ModelSpec) -> Path:
    """ensure_model, with an explanation and a progress bar on stdout.

    A first run that pauses for a 166 MB download with no output looks
    exactly like a hang, so it says what it is doing before it starts.
    """
    existing = find_model(spec)
    if existing is not None:
        return existing

    print(
        f"Downloading the {spec.description} ({spec.size_label}) -- "
        f"first run only.\n  to: {cache_dir()}"
    )
    path = download_model(spec, on_progress=_print_progress(spec))
    print(f"\r{'  done.':<{_PROGRESS_WIDTH}}")
    return path


def clear_cache() -> int:
    """Deletes downloaded models. Returns how many files were removed."""
    directory = cache_dir()
    if not directory.is_dir():
        return 0

    removed = 0
    for spec in MODELS.values():
        target = directory / spec.filename
        if target.is_file():
            target.unlink()
            removed += 1
    shutil.rmtree(directory, ignore_errors=True)
    return removed
