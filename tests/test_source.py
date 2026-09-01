"""Tests for keeping a video readable after its file moves.

Most of these need no real footage: VideoSource validates and holds a
descriptor without decoding anything, so a file with the right extension is
enough to prove the file-system half. The two that actually decode are
marked, and skip on a clone with no assets/.
"""

import os
import shutil
from pathlib import Path

import pytest

from app.video.loader import VideoLoadError
from app.video.source import SourceMismatch, SourceMissing, VideoSource

VIDEO = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test.mp4"

needs_footage = pytest.mark.skipif(
    not VIDEO.is_file(), reason=f"no sample footage at {VIDEO}"
)


@pytest.fixture
def fake_video(tmp_path):
    """A file that passes validation without being decodable.

    Enough for everything except opening it, and it keeps these tests off
    the 47 MB clip.
    """
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video, but it has the right name" * 10)
    return path


# ------------------------------------------------------------ construction


def test_an_unsupported_extension_is_refused(tmp_path):
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"x")
    with pytest.raises(VideoLoadError, match="Unsupported"):
        VideoSource(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(VideoLoadError, match="does not exist"):
        VideoSource(tmp_path / "nope.mp4")


def test_the_size_is_recorded_at_construction(fake_video):
    with VideoSource(fake_video) as source:
        assert source.size == fake_video.stat().st_size


# --------------------------------------------------------- the held handle


def test_a_held_source_survives_a_rename(fake_video):
    """The point of the descriptor: the name can change under it."""
    with VideoSource(fake_video, keep_open=True) as source:
        os.rename(fake_video, fake_video.parent / "renamed.mp4")
        assert source.is_available()


def test_a_held_source_survives_deletion(fake_video):
    """A descriptor refers to the inode, so even unlinking is survivable."""
    with VideoSource(fake_video, keep_open=True) as source:
        os.unlink(fake_video)
        assert not fake_video.exists()
        assert source.is_available()


def test_without_a_handle_a_moved_file_is_lost(fake_video):
    """What Windows gets, and what makes relocate necessary there."""
    with VideoSource(fake_video, keep_open=False) as source:
        os.rename(fake_video, fake_video.parent / "renamed.mp4")
        assert not source.is_available()


def test_a_pathless_source_reports_what_went_wrong(fake_video):
    source = VideoSource(fake_video, keep_open=False)
    os.unlink(fake_video)
    with pytest.raises(SourceMissing, match="no longer at"):
        source.open()


def test_closing_twice_is_harmless(fake_video):
    source = VideoSource(fake_video, keep_open=True)
    source.close()
    source.close()


def test_closing_falls_back_to_the_path(fake_video):
    """Releasing the descriptor should not orphan a file that is still there."""
    source = VideoSource(fake_video, keep_open=True)
    source.close()
    assert source.is_available()


# ------------------------------------------------------------- relocation


def test_relocating_follows_the_file(fake_video, tmp_path):
    moved = tmp_path / "elsewhere" / "clip.mp4"
    moved.parent.mkdir()
    with VideoSource(fake_video, keep_open=False) as source:
        shutil.move(fake_video, moved)
        assert not source.is_available()

        source.relocate(moved)
        assert source.is_available()
        assert source.path == moved


def test_relocating_to_a_different_video_is_refused(fake_video, tmp_path):
    """The guard against picking the wrong file out of a folder of clips."""
    other = tmp_path / "other.mp4"
    other.write_bytes(b"a different length entirely")

    with VideoSource(fake_video, keep_open=False) as source:
        with pytest.raises(SourceMismatch, match="bytes"):
            source.relocate(other)
        # And the source is left pointing where it was, not half-moved.
        assert source.path == fake_video


def test_relocating_to_a_non_video_is_refused(fake_video, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"x")
    with VideoSource(fake_video, keep_open=False) as source:
        with pytest.raises(VideoLoadError, match="Unsupported"):
            source.relocate(notes)


def test_relocating_closes_the_old_handle(fake_video, tmp_path):
    """Otherwise every relocation leaks a descriptor."""
    copy = tmp_path / "copy.mp4"
    shutil.copy(fake_video, copy)
    with VideoSource(fake_video, keep_open=True) as source:
        old = source._handle
        source.relocate(copy)
        assert old.closed
        assert not source._handle.closed


# ------------------------------------------------------------ real decoding


@needs_footage
def test_a_moved_video_still_decodes(tmp_path):
    """The whole feature, end to end: unlink the file, then read it anyway."""
    copy = tmp_path / "clip.mp4"
    shutil.copy(VIDEO, copy)

    with VideoSource(copy, keep_open=True) as source:
        os.unlink(copy)
        with source.open() as container:
            stream = container.streams.video[0]
            container.seek(int(2 / stream.time_base), stream=stream)
            decoded = sum(1 for _, _ in zip(container.decode(stream), range(10)))

    assert decoded == 10


@needs_footage
def test_two_opens_do_not_disturb_each_other(tmp_path):
    """Each open duplicates the descriptor, so readers have their own offset."""
    copy = tmp_path / "clip.mp4"
    shutil.copy(VIDEO, copy)

    with VideoSource(copy, keep_open=True) as source:
        first = source.open()
        first.seek(int(5 / first.streams.video[0].time_base),
                   stream=first.streams.video[0])
        next(first.decode(first.streams.video[0]))

        # A second reader, opened while the first is mid-stream, should see
        # the file from the beginning rather than from wherever the first
        # left the offset.
        with source.open() as second:
            stream = second.streams.video[0]
            frame = next(second.decode(stream))
            assert frame.time is not None and frame.time < 1.0

        first.close()
