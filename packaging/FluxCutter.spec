# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for FluxCutter, macOS and Windows.

Run from the repository root:

    pyinstaller packaging/FluxCutter.spec --noconfirm

The models are deliberately not bundled. They are 174 MB, they are not in
the repository (assets/ is ignored), and the app downloads and verifies them
on first use anyway (app/models.py). Bundling them would roughly triple the
download for something the app can fetch once, in the background, with a
progress bar.
"""

import re
import sys
from pathlib import Path



OPTIONAL_EXCLUDES = ["matplotlib", "scipy", "sklearn", "pytest"]
if sys.platform != "darwin":
    OPTIONAL_EXCLUDES.append("onnxruntime")


def project_version(root: Path) -> str:
    """Reads __version__ out of app/__init__.py without importing it.

    Importing would pull in the whole dependency tree at spec-parse time,
    which is both slow and a way to fail the build for a reason that has
    nothing to do with packaging.
    """
    text = (root / "app" / "__init__.py").read_text()
    match = re.search(r'^__version__ = "([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"

IS_MACOS = sys.platform == "darwin"
ROOT = Path(SPECPATH).resolve().parent

# Both formats are committed, because .icns can only be produced on macOS
# (iconutil) and this spec has to build on Windows too. Regenerate them with
# `python packaging/make_icon.py`. A clone that has deleted them still
# builds, just with PyInstaller's own logo.
VERSION = project_version(ROOT)

ICNS = ROOT / "packaging" / "icon.icns"
ICO = ROOT / "packaging" / "icon.ico"
PNG = ROOT / "packaging" / "icon.png"
EXE_ICON = str(ICNS if IS_MACOS else ICO) if (ICNS.exists() and ICO.exists()) else None

analysis = Analysis(
    [str(ROOT / "packaging" / "fluxcutter.py")],
    pathex=[str(ROOT)],
    binaries=[],
    # window.html is the desktop window. It is read at runtime rather than
    # imported, so the dependency analysis cannot see it, and a bundle
    # without it raises FileNotFoundError before anything is drawn.
    # The .png rides along for the Windows title bar and taskbar icon.
    datas=[(str(ROOT / "app" / "ui" / "window.html"), ".")]
    + ([(str(PNG), ".")] if PNG.exists() else []),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    # onnxruntime costs ~80 MB in the bundle, so it ships only where it earns
    # that. On macOS it does: live-action embedding runs through its CoreML
    # provider, which is 4-5x faster than cv2.dnn on Apple silicon, so every
    # scan benefits and animation mode comes along for free. Elsewhere live
    # action does not touch it and only animation mode would, so it stays out
    # and the frozen app reports Animation as needing an install -- which is
    # what app/modes.availability already says.
    excludes=OPTIONAL_EXCLUDES,
    noarchive=False,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="FluxCutter",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=EXE_ICON,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="FluxCutter",
)

if IS_MACOS:
    app = BUNDLE(
        collection,
        name="FluxCutter.app",
        icon=str(ICNS) if ICNS.exists() else None,
        bundle_identifier="com.fluxcutter.app",
        info_plist={
            "NSHighResolutionCapable": True,
            # Without this the window opens at 1x on a Retina display.
            "CFBundleShortVersionString": VERSION,
        },
    )
