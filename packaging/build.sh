#!/usr/bin/env bash
#
# Build FluxCutter locally. Works on macOS and on Linux/Windows under a bash
# shell (Git Bash, WSL), producing whatever the machine it runs on can make.
#
#   ./packaging/build.sh            build, then zip it
#   ./packaging/build.sh --no-zip   build only
#   ./packaging/build.sh --run      build, then open it
#
# There is no cross-compilation: PyInstaller freezes the interpreter it is
# running on, so this makes a .app on a Mac and a folder on Windows, and
# neither machine can make the other's.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    elif [ -x ".venv/Scripts/python.exe" ]; then
        PYTHON=".venv/Scripts/python.exe"
    else
        PYTHON="python3"
    fi
fi

DO_ZIP=1
DO_RUN=0
for arg in "$@"; do
    case "$arg" in
        --no-zip) DO_ZIP=0 ;;
        --run)    DO_RUN=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "==> Python: $($PYTHON --version) at $PYTHON"

if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
    echo "==> Installing PyInstaller"
    "$PYTHON" -m pip install --quiet pyinstaller
fi

# Tk is a system package, not a pip one, and its absence fails at import
# rather than at build -- clearer to say so here.
if ! "$PYTHON" -c "import tkinter" 2>/dev/null; then
    echo "Error: this Python has no tkinter." >&2
    echo "  macOS:  brew install python-tk@3.12" >&2
    echo "  Ubuntu: sudo apt install python3-tk" >&2
    exit 1
fi

echo "==> Building"
rm -rf build dist
"$PYTHON" -m PyInstaller packaging/FluxCutter.spec --noconfirm --log-level WARN

case "$(uname -s)" in
    Darwin) BUNDLE="dist/FluxCutter.app"; ZIP="dist/FluxCutter-macos.zip" ;;
    *)      BUNDLE="dist/FluxCutter";     ZIP="dist/FluxCutter-$(uname -s).zip" ;;
esac

if [ ! -e "$BUNDLE" ]; then
    echo "Error: expected $BUNDLE, which is not there." >&2
    exit 1
fi

echo "==> Built $BUNDLE ($(du -sh "$BUNDLE" | cut -f1))"

if [ "$DO_ZIP" = 1 ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        # ditto rather than zip: a .app is a directory, and zip loses the
        # executable bit and the bundle's symlinks.
        ditto -c -k --sequesterRsrc --keepParent "$BUNDLE" "$ZIP"
    else
        (cd dist && zip -qr "$(basename "$ZIP")" "$(basename "$BUNDLE")")
    fi
    echo "==> Zipped $ZIP ($(du -sh "$ZIP" | cut -f1))"
    if [ "$(uname -s)" = "Darwin" ]; then
        echo
        echo "    Anyone you send this to must clear the quarantine flag,"
        echo "    or macOS will call it damaged:"
        echo
        echo "        xattr -dr com.apple.quarantine FluxCutter.app"
    fi
fi

if [ "$DO_RUN" = 1 ]; then
    echo "==> Opening"
    if [ "$(uname -s)" = "Darwin" ]; then
        # A locally built app is not quarantined, but clear it anyway so
        # this also works on one that came from a download.
        xattr -dr com.apple.quarantine "$BUNDLE" 2>/dev/null || true
        open "$BUNDLE"
    else
        "$BUNDLE/FluxCutter" &
    fi
fi
