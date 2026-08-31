"""Entry point for the packaged desktop app.

PyInstaller freezes a script, not a module, so this exists to be that
script. It deliberately does nothing but call launch(): anything else here
would be code that only runs in the frozen build and is therefore never
exercised by the tests.
"""

import multiprocessing
import sys
from pathlib import Path

# A frozen app has no notion of the project root, and PyInstaller runs this
# file directly rather than importing it as part of the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ui.app import launch


if __name__ == "__main__":
    # Without this a frozen app that ever spawns a process re-runs the whole
    # bundle instead of the child, which shows up as the window opening
    # twice. Harmless to call when nothing spawns anything.
    multiprocessing.freeze_support()
    launch()
