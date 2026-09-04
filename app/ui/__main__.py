"""Opens the desktop window: `python -m app.ui [video]`.

`python -m app ui [video]` does the same thing through the main CLI.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ui.web import launch


if __name__ == "__main__":
    launch(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
