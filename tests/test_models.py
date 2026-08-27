"""Tests for model discovery and first-use downloading.

Nothing here reaches the network. The one thing that must be exercised
against a real server -- that the pinned URLs still serve the pinned
hashes -- is a check of the outside world rather than of this code, and is
done by `python -m app models fetch` rather than by the suite.

What is tested is the part that goes wrong quietly: that a file which is
not what was expected is rejected and removed, and that an interrupted
download leaves nothing behind that a later run would trust.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from app import models


def spec_for(tmp_path: Path, payload: bytes, name: str = "fake-model.onnx"):
    """A ModelSpec pointing at a local file, served through a file:// URL."""
    source = tmp_path / "source" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return models.ModelSpec(
        filename=name,
        url=source.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        description="test model",
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the cache at a temporary directory with no repository copy."""
    directory = tmp_path / "cache"
    monkeypatch.setenv("FLUXCUTTER_MODEL_DIR", str(directory))
    monkeypatch.setattr(models, "_REPOSITORY_MODELS", tmp_path / "no-repo-models")
    return directory


def test_a_model_downloads_and_verifies(cache, tmp_path):
    spec = spec_for(tmp_path, b"pretend onnx bytes")

    path = models.ensure_model(spec)

    assert path == cache / spec.filename
    assert path.read_bytes() == b"pretend onnx bytes"


def test_progress_is_reported_as_a_fraction(cache, tmp_path):
    spec = spec_for(tmp_path, b"x" * 4096)
    seen = []

    models.ensure_model(spec, on_progress=lambda f, d, t: seen.append((f, d, t)))

    assert seen
    assert seen[-1][0] == 1.0
    assert seen[-1][1] == seen[-1][2] == 4096


def test_a_second_call_does_not_download_again(cache, tmp_path):
    spec = spec_for(tmp_path, b"once")
    first = models.ensure_model(spec)

    # Removing the source makes any re-download impossible, so a success
    # here proves the cached copy was used.
    Path(spec.url.removeprefix("file://")).unlink()

    assert models.ensure_model(spec) == first


def test_a_file_that_is_not_what_was_expected_is_rejected(cache, tmp_path):
    """The reason checksums are here at all.

    A model once arrived corrupted through a text-mode round trip -- 70 MB
    instead of 38 MB -- and surfaced as five confusing test failures rather
    than as a download problem.
    """
    spec = replace(spec_for(tmp_path, b"the wrong bytes"), sha256="0" * 64)

    with pytest.raises(models.ModelDownloadError) as caught:
        models.download_model(spec)

    assert "not the expected file" in str(caught.value)


def test_a_rejected_download_leaves_nothing_behind(cache, tmp_path):
    """A bad file must not sit in the cache looking usable."""
    spec = replace(spec_for(tmp_path, b"the wrong bytes"), sha256="0" * 64)

    with pytest.raises(models.ModelDownloadError):
        models.download_model(spec)

    assert list(cache.iterdir()) == []


def test_an_interrupted_download_leaves_nothing_behind(cache, tmp_path):
    """Cancelling mid-download must not produce a half file the next run trusts.

    The UI cancels by raising from the progress callback, so that is how
    this interrupts it.
    """
    spec = spec_for(tmp_path, b"y" * 100_000)

    def stop(_fraction, _done, _total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        models.download_model(spec, on_progress=stop)

    assert list(cache.iterdir()) == []


def test_a_download_failure_explains_which_model_and_where_from(cache, tmp_path):
    spec = replace(
        spec_for(tmp_path, b"never read"),
        url=(tmp_path / "does-not-exist.onnx").as_uri(),
    )

    with pytest.raises(models.ModelDownloadError) as caught:
        models.download_model(spec)

    assert "test model" in str(caught.value)


def test_a_repository_copy_is_preferred_over_downloading(tmp_path, monkeypatch):
    """A checkout with models already in it must keep working untouched."""
    repository = tmp_path / "assets" / "models"
    repository.mkdir(parents=True)
    monkeypatch.setenv("FLUXCUTTER_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(models, "_REPOSITORY_MODELS", repository)

    spec = spec_for(tmp_path, b"whatever")
    (repository / spec.filename).write_bytes(b"the checkout's own copy")

    assert models.find_model(spec) == repository / spec.filename
    assert models.ensure_model(spec).read_bytes() == b"the checkout's own copy"


def test_find_model_returns_none_when_it_must_be_fetched(cache, tmp_path):
    assert models.find_model(spec_for(tmp_path, b"absent")) is None


def test_the_cache_directory_is_outside_the_app(monkeypatch):
    """A frozen .app may be on a read-only volume; never write next to it."""
    monkeypatch.delenv("FLUXCUTTER_MODEL_DIR", raising=False)

    directory = models.cache_dir()

    assert directory.is_absolute()
    assert "FluxCutter" in directory.parts
    assert Path(models.__file__).parent not in directory.parents


def test_sizes_read_sensibly_at_both_scales():
    """YuNet is 0.23 MB, which rounded to '0 MB' and read as a bug."""
    assert models.MODELS["detector"].size_label == "230 KB"
    assert models.MODELS["embedder"].size_label == "174 MB"


def test_every_shipped_spec_pins_a_real_looking_hash():
    """A blank or short hash would make verification silently meaningless."""
    for spec in models.MODELS.values():
        assert len(spec.sha256) == 64
        assert set(spec.sha256) <= set("0123456789abcdef")
        assert spec.url.startswith("https://")


def test_a_truncated_download_says_so_rather_than_blaming_the_file(
    cache, tmp_path, monkeypatch
):
    """A server closing early looks like a finished download unless counted.

    Fetching the real ArcFace model over its real URL, curl returned a
    135,783,125 byte body for a 174,383,860 byte file and exited 0. The
    hash would have caught it either way, but "not the expected file"
    points the user at the wrong problem: the fix is to retry, not to
    distrust the source.
    """
    payload = b"z" * 50_000
    spec = spec_for(tmp_path, payload)

    real_urlopen = models.urllib.request.urlopen

    class TruncatingResponse:
        """Serves half the body, then reports the stream as finished."""

        def __init__(self, inner):
            self._inner = inner
            self.headers = inner.headers
            self._budget = len(payload) // 2

        def read(self, size):
            if self._budget <= 0:
                return b""
            chunk = self._inner.read(min(size, self._budget))
            self._budget -= len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._inner.close()
            return False

    monkeypatch.setattr(
        models.urllib.request,
        "urlopen",
        lambda request, timeout=None: TruncatingResponse(real_urlopen(request)),
    )

    with pytest.raises(models.ModelDownloadError) as caught:
        models.download_model(spec)

    assert "stopped early" in str(caught.value)
    assert "retry" in str(caught.value)
    assert list(cache.iterdir()) == []
