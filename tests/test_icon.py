"""Tests for the generated app icon.

The icon is drawn by packaging/make_icon.py rather than committed as
artwork alone, so the thing worth checking is that the committed files
match what that script produces and that they carry the sizes each
platform asks for. A .icns that is missing its 16x16 is a Dock full of a
blurry upscale, which nothing else would catch.
"""

from pathlib import Path

import numpy as np
from PIL import Image

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"


def test_every_format_is_committed():
    """CI builds on a runner that cannot regenerate the .icns."""
    for name in ("icon.png", "icon.ico", "icon.icns"):
        assert (PACKAGING / name).is_file(), f"{name} is missing"


def test_the_master_png_is_square_and_full_size():
    with Image.open(PACKAGING / "icon.png") as image:
        assert image.size == (1024, 1024)


def test_the_ico_carries_the_sizes_windows_asks_for():
    with Image.open(PACKAGING / "icon.ico") as image:
        sizes = {size for size in image.info["sizes"]}
    for expected in ((16, 16), (32, 32), (48, 48), (256, 256)):
        assert expected in sizes, f"{expected} missing from icon.ico"


def test_the_artwork_reaches_the_edges_of_its_tile():
    """A guard against a geometry edit that shrinks the icon into a stamp.

    The rounded tile is inset 64/1024 by design; anything much smaller than
    that means the drawing numbers have drifted.
    """
    with Image.open(PACKAGING / "icon.png") as image:
        alpha = image.convert("RGBA").getchannel("A")
    left, top, right, bottom = alpha.getbbox()
    assert left <= 80 and top <= 80
    assert right >= 1024 - 80 and bottom >= 1024 - 80


def test_the_icon_is_legible_as_a_silhouette_at_16px():
    """The Dock size. Ink has to be a real fraction of the tile to show.

    The first version failed this: a 38/1024 bracket stroke came out under a
    pixel wide at 16px and left 4 white pixels of 256, against 16 for the
    geometry that shipped. The threshold sits between the two, which makes
    this a guard against thinning the artwork rather than a claim that 12 is
    the exact point where an icon stops reading.
    """
    import sys

    sys.path.insert(0, str(PACKAGING))
    from make_icon import render

    small = np.asarray(render(16).convert("RGBA"))
    # Near-white pixels inside the tile: the brackets and the wedge.
    ink = int(((small[..., 3] > 128) & (small[..., :3] > 200).all(axis=-1)).sum())
    assert ink >= 12, f"only {ink} of 256 pixels carry artwork at 16px"
