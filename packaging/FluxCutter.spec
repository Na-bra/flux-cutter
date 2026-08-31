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

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

IS_MACOS = sys.platform == "darwin"
ROOT = Path(SPECPATH).resolve().parent

analysis = Analysis(
    [str(ROOT / "packaging" / "fluxcutter.py")],
    pathex=[str(ROOT)],
    binaries=[],
    # CustomTkinter loads its themes from .json files at runtime, which are
    # data rather than imports and so are invisible to the dependency
    # analysis. Without this the frozen app raises FileNotFoundError on the
    # theme before it ever draws a window.
    datas=collect_data_files("customtkinter"),
    hiddenimports=["PIL._tkinter_finder"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "sklearn", "pytest"],
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
        icon=None,
        bundle_identifier="com.fluxcutter.app",
        info_plist={
            "NSHighResolutionCapable": True,
            # Without this the window opens at 1x on a Retina display.
            "CFBundleShortVersionString": "1.0.0",
        },
    )
