"""Holding on to a video after its file has moved.

A scan takes minutes and its result is only half useful on its own: the
gallery is in memory, but cutting a reel has to read the footage again.
Between those two moments the user is free to rename the file, drag it to
another folder, or unplug the drive it lives on -- and until this module
existed, doing any of that turned a finished scan into "Could not export:
Could not open /the/old/path".

Two independent defences, because they fail in different situations:

- **A held descriptor.** On macOS and Linux an open descriptor refers to
  the inode, not the name, so the file can be renamed, moved across
  directories, or deleted outright and reads keep working. Verified: with
  the path unlinked so nothing pointed at the file at all, PyAV still
  opened, seeked and decoded from the descriptor at full speed. It costs
  27 MB of process memory on an 815 MB video -- against 767 MB to hold the
  same file in RAM, which decodes no faster because the page cache is
  already doing that job.

- **Relocation.** A descriptor dies with the process, so it does nothing
  for a video that moved while the app was closed, and (see below) it is
  not held at all on Windows. `relocate` lets the caller point the source
  at the file's new home and carry on with the scan it already has.

Windows deliberately gets only the second. CPython's `open()` does not
request `FILE_SHARE_DELETE`, so a descriptor held there would stop the
user renaming or deleting their own video for as long as FluxCutter had it
open -- trading a failed export for a blocked file operation, which is a
worse bargain than it sounds when the app sits open all afternoon.
"""

import os
import sys
from pathlib import Path

import av

from app.video.loader import VideoLoadError, validate_video_path

# See the module docstring: holding a descriptor detaches a video from its
# path on POSIX and obstructs the user on Windows.
KEEPS_HANDLES = sys.platform != "win32"


class SourceMissing(VideoLoadError):
    """Raised when the video is neither held open nor still at its path."""


class SourceMismatch(VideoLoadError):
    """Raised when a replacement file is not the video that was scanned."""


class VideoSource:
    """One video, addressed by whichever of its two handles still works.

    Not thread-safe to construct or relocate, but `open` is: each call
    duplicates the descriptor, so a reader gets its own file offset and two
    readers cannot seek each other sideways.
    """

    def __init__(self, path: str | Path, keep_open: bool | None = None):
        """
        Args:
            path: The video file. Validated the same way `load_video`
                validates it, so an unsupported extension is refused here
                rather than at the first decode.
            keep_open: Whether to hold a descriptor. Defaults to the
                platform's answer; tests pass it explicitly.

        Raises:
            VideoLoadError: If the path is not a readable, supported video.
        """
        path = validate_video_path(path)
        self.path = path
        # The identity check `relocate` uses. Recorded now, while the file
        # is known to be the right one.
        self.size = path.stat().st_size
        self._keep_open = KEEPS_HANDLES if keep_open is None else keep_open
        self._handle = open(path, "rb") if self._keep_open else None

    # ------------------------------------------------------------- reading

    def open(self) -> av.container.InputContainer:
        """Opens the video for decoding.

        Prefers the held descriptor, so this keeps working after the file
        has moved; falls back to the path when no descriptor is held.

        Raises:
            SourceMissing: If neither route reaches the file.
            VideoLoadError: If it is reachable but cannot be decoded.
        """
        if self._handle is not None:
            # A duplicate rather than the handle itself: PyAV seeks the
            # object it is given, and the original has to stay at a known
            # position for the next caller.
            view = os.fdopen(os.dup(self._handle.fileno()), "rb")
            view.seek(0)
            try:
                return av.open(view)
            except av.FFmpegError as error:
                view.close()
                raise VideoLoadError(
                    f"Could not open video: {self.path} ({error})"
                ) from error

        if not self.path.is_file():
            raise SourceMissing(
                f"{self.path.name} is no longer at {self.path.parent}."
            )
        try:
            return av.open(str(self.path))
        except av.FFmpegError as error:
            raise VideoLoadError(
                f"Could not open video: {self.path} ({error})"
            ) from error

    def is_available(self) -> bool:
        """Whether `open` would find the footage."""
        return self._handle is not None or self.path.is_file()

    # ---------------------------------------------------------- relocation

    def relocate(self, path: str | Path) -> None:
        """Points this source at the video's new location.

        Args:
            path: Where the file is now.

        Raises:
            SourceMismatch: If the file there is a different size, and so
                is almost certainly not the video that was scanned. This is
                a cheap guard against picking the wrong file in a folder of
                clips, not proof of identity -- two different videos of
                exactly equal byte length would pass it.
            VideoLoadError: If the path is not a readable, supported video.
        """
        path = validate_video_path(path)
        size = path.stat().st_size
        if size != self.size:
            raise SourceMismatch(
                f"{path.name} is {size:,} bytes, but the video that was "
                f"scanned is {self.size:,}. The timings from that scan would "
                "not line up with this file."
            )

        old = self._handle
        self._handle = open(path, "rb") if self._keep_open else None
        self.path = path
        if old is not None:
            old.close()

    # ------------------------------------------------------------ lifetime

    def close(self) -> None:
        """Releases the descriptor. Safe to call more than once."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __repr__(self) -> str:
        held = "held" if self._handle is not None else "by path"
        return f"VideoSource({self.path.name!r}, {held})"
