#!/usr/bin/env python3
"""Draws the FluxCutter app icon and writes it in both platforms' formats.

    python packaging/make_icon.py

Generated rather than committed as a lone .png nobody can edit: the shape
is a dozen numbers, and having them here means the icon can be re-cut at
any size or recoloured without hunting for the original artwork.

The design has to survive 16x16 in a Dock and a taskbar, which rules out
anything with interior detail. What is left is a silhouette: focus
brackets around a play triangle, split by the cut. Brackets say "find
someone", the triangle says "video", the split says "cut" -- and at 16px
the brackets and the triangle are still four corners and a wedge.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SIZE = 1024
# Everything is drawn at 4x and shrunk, which is cheaper than working out
# antialiasing by hand and gives cleaner diagonals than PIL's own.
SCALE = 4

# A violet-to-blue diagonal. Dark enough that white artwork holds up on a
# light desktop, saturated enough not to read as a system utility.
TOP_LEFT = (124, 58, 237)
BOTTOM_RIGHT = (29, 78, 216)
INK = (255, 255, 255)


def gradient(size: int) -> Image.Image:
    """A diagonal two-stop gradient, as the icon's ground."""
    ramp = np.add.outer(np.linspace(0, 1, size), np.linspace(0, 1, size)) / 2
    ramp = ramp[:, :, None]
    start = np.array(TOP_LEFT, dtype=float)
    end = np.array(BOTTOM_RIGHT, dtype=float)
    pixels = start + (end - start) * ramp
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB")


def draw_artwork(size: int) -> Image.Image:
    """The icon at one size, on a transparent ground."""
    s = size / 1024  # every number below is in 1024-space

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # macOS wants the artwork inset rather than full-bleed; Windows does not
    # mind, and the same inset keeps one file working for both.
    margin = int(64 * s)
    radius = int(200 * s)
    tile = Image.new("L", (size, size), 0)
    ImageDraw.Draw(tile).rounded_rectangle(
        [margin, margin, size - margin - 1, size - margin - 1],
        radius=radius,
        fill=255,
    )
    canvas.paste(gradient(size), (0, 0), tile)

    draw = ImageDraw.Draw(canvas)

    # --- focus brackets: four corners, no sides, which is what makes them
    # read as a viewfinder rather than as a frame.
    # Chunky on purpose. At 38/1024 the brackets were under a pixel wide
    # in a 16px Dock tile and dissolved into the gradient; 64 survives it.
    stroke = int(64 * s)
    inset = int(238 * s)
    arm = int(132 * s)
    left, top = inset, inset
    right, bottom = size - inset, size - inset
    for x, y, dx, dy in (
        (left, top, 1, 1),
        (right, top, -1, 1),
        (left, bottom, 1, -1),
        (right, bottom, -1, -1),
    ):
        draw.line([(x, y), (x + dx * arm, y)], fill=INK, width=stroke, joint="curve")
        draw.line([(x, y), (x, y + dy * arm)], fill=INK, width=stroke, joint="curve")
        # Square corners on their own leave a notch at the join.
        half = stroke // 2
        draw.ellipse([x - half, y - half, x + half, y + half], fill=INK)

    # --- the play triangle, drawn into its own mask so the cut can be taken
    # out of it cleanly.
    wedge = Image.new("L", (size, size), 0)
    wedge_draw = ImageDraw.Draw(wedge)
    # A triangle's optical centre sits a third of the way from its base,
    # so the geometric centre has to sit left of the icon's to look centred.
    wedge_draw.polygon(
        [(int(424 * s), int(366 * s)),
         (int(424 * s), int(658 * s)),
         (int(672 * s), int(512 * s))],
        fill=255,
    )

    # --- the cut: a diagonal gap through the wedge, offset from centre so it
    # is clearly a slice and not a symmetry line.
    gap = int(34 * s)
    wedge_draw.line(
        [(int(360 * s), int(672 * s)), (int(730 * s), int(330 * s))],
        fill=0,
        width=gap,
    )

    canvas.paste(Image.new("RGBA", (size, size), INK + (255,)), (0, 0), wedge)
    return canvas


def render(size: int) -> Image.Image:
    return draw_artwork(size * SCALE).resize((size, size), Image.LANCZOS)


def write_icns(master: Image.Image, out: Path) -> bool:
    """Builds a .icns through iconutil, which is macOS-only."""
    if sys.platform != "darwin":
        print("  .icns skipped: iconutil is macOS-only")
        return False

    iconset = out.with_suffix(".iconset")
    iconset.mkdir(exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        render(size).save(iconset / f"icon_{size}x{size}.png")
        render(size * 2).save(iconset / f"icon_{size}x{size}@2x.png")

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True
    )
    for leftover in iconset.iterdir():
        leftover.unlink()
    iconset.rmdir()
    return True


def main() -> None:
    master = render(SIZE)
    master.save(ROOT / "icon.png")
    print(f"  wrote {ROOT / 'icon.png'}")

    # Every size Windows will ask for, so it never has to scale one itself.
    master.save(
        ROOT / "icon.ico",
        sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)],
    )
    print(f"  wrote {ROOT / 'icon.ico'}")

    if write_icns(master, ROOT / "icon.icns"):
        print(f"  wrote {ROOT / 'icon.icns'}")


if __name__ == "__main__":
    main()
