"""The desktop window, drawn by the system's own web view.

Replaces the CustomTkinter window. The pipeline is untouched: everything
here calls app/ui/worker.py, which imports no toolkit of any kind and did
not change when the window did.

Why a web view rather than a toolkit. The screen this app needs is a grid
of face thumbnails that reflows to the window, and a reflowing grid is the
one thing Tk makes hardest and CSS makes free. It also drops a system
dependency: Tk ships separately from Python, so `brew install
python-tk@3.12` was a documented setup step and a Homebrew Python without
it could not open the window at all. Every machine this runs on already
has a web view -- WebKit on macOS, WebView2 on Windows.

The bridge is deliberately thin. Python owns every decision; the page
renders what it is told and reports clicks back. Nothing in `Bridge` does
pipeline work itself, so the rules about what a scan means stay in one
place rather than being re-implemented in JavaScript.
"""

import base64
import io
import json
import sys
import threading
from pathlib import Path

import webview

from app.modes import DEFAULT_MODE, MODES, availability, mode_ids
from app.ui.macos import set_application_name
from app.ui.worker import (
    Cancelled,
    ExportSettings,
    Person,
    ScanResult,
    ScanSettings,
    available_encoders,
    default_encoder,
    export,
    plan_export,
    quality_for,
    scan,
)
from app.video.loader import VideoLoadError
from app.video.source import SourceMismatch

DEFAULT_OUTPUT_DIR = Path.home() / "Movies"
DEFAULT_FILENAME = "reel.mp4"
QUALITY_LEVELS = ["Standard", "High", "Maximum"]
SAMPLE_INTERVALS = [0.25, 0.5, 1.0, 2.0]

WINDOW_TITLE = "FluxCutter"
WINDOW_SIZE = (1120, 760)
MINIMUM_SIZE = (900, 620)


def _page() -> str:
    """The window's HTML, from beside this module or from the bundle.

    PyInstaller unpacks data files to a temporary directory and points
    sys._MEIPASS at it, so the frozen app cannot look next to __file__.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    candidates = [Path(__file__).with_name("window.html")]
    if bundled:
        candidates.insert(0, Path(bundled) / "window.html")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("window.html is missing from the installation")


def _data_uri(image) -> str:
    """A PIL thumbnail as something an <img> can show.

    Inlined rather than written to a temporary file and served: the crops
    are a few kilobytes each, there are rarely more than a few dozen, and a
    file on disk would need cleaning up after a window that may be killed.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def output_path(folder: str, filename: str) -> Path:
    """Where the next export will be written.

    The folder and the name are separate fields because they change on
    different rhythms: a folder is chosen once for a session's worth of
    reels, while the name follows whichever face is selected. A missing
    .mp4 is added rather than refused -- the extension is not a decision
    anyone wants to be corrected about.
    """
    directory = Path(folder.strip() or DEFAULT_OUTPUT_DIR).expanduser()
    name = filename.strip() or DEFAULT_FILENAME
    if not name.lower().endswith(".mp4"):
        name = f"{name}.mp4"
    return directory / name


def _clock(seconds: float) -> str:
    """Seconds as m:ss, which is how long a reel is talked about."""
    minutes, remainder = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes}:{remainder:02d}"


class Bridge:
    """What the page may ask Python to do.

    Every method here is reachable from JavaScript, so each one is written
    as if the page were untrusted: arguments are validated, and anything
    that runs long goes to a worker thread rather than blocking the window.
    """

    def __init__(self, video_path: Path | None = None):
        self.window = None
        self._initial_video = str(video_path) if video_path else ""
        self._mode = DEFAULT_MODE
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._scan_result: ScanResult | None = None
        self._selected: Person | None = None
        self._suggested_filename = ""

    # ------------------------------------------------------------- helpers

    def _emit(self, function: str, payload=None) -> None:
        """Calls a function on the page. Safe from a worker thread."""
        if self.window is None:
            return
        argument = "" if payload is None else json.dumps(payload)
        try:
            self.window.evaluate_js(f"{function}({argument})")
        except Exception:
            # The window closed underneath a worker that was still
            # reporting. Losing the update is correct; raising is not.
            pass

    def _status(self, text: str) -> None:
        self._emit("onStatus", {"text": text})

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _start(self, target, *args) -> None:
        self._cancel.clear()
        self._worker = threading.Thread(target=target, args=args, daemon=True)
        self._worker.start()

    # -------------------------------------------------------- what to draw

    def initial_state(self) -> dict:
        """Everything the page needs to render itself once, at startup."""
        return {
            "video": self._initial_video,
            "modes": [
                {"id": mode, "label": MODES[mode].display_name} for mode in mode_ids()
            ],
            "mode": self._mode,
            "intervals": SAMPLE_INTERVALS,
            "interval": ScanSettings().sample_interval,
            "encoders": available_encoders(),
            "encoder": default_encoder(),
            "qualities": QUALITY_LEVELS,
            "quality": "High",
            "folder": str(DEFAULT_OUTPUT_DIR),
            "filename": DEFAULT_FILENAME,
            "status": self._availability_text(self._mode),
        }

    def _availability_text(self, mode: str) -> str:
        state = availability(mode)
        name = MODES[mode].display_name
        if state.usable:
            return f"{name} mode. Ready to scan."
        return f"{name} mode. {state.describe()}."

    # ------------------------------------------------------------- choices

    def choose_video(self) -> dict:
        chosen = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Video files (*.mp4;*.mov)", "All files (*.*)"),
        )
        if not chosen:
            return {"path": None}
        return {"path": str(chosen[0])}

    def choose_folder(self) -> dict:
        chosen = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not chosen:
            return {"path": None}
        return {"path": str(chosen[0])}

    def set_mode(self, mode: str) -> dict:
        """Records the content type and says what it still needs.

        Choosing a mode whose weights are missing is allowed -- the scan
        fetches them with a progress bar. What is not allowed is finding
        out only after a long scan, so the requirement is stated now.
        """
        if mode not in MODES:
            return {"status": self._availability_text(self._mode)}
        self._mode = mode
        return {"status": self._availability_text(mode)}

    # ---------------------------------------------------------------- scan

    def start_scan(self, video: str, mode: str, interval: float) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}

        path = Path(video.strip()) if video else None
        if path is None or not path.is_file():
            return {"started": False, "reason": "Choose a video first."}

        if mode in MODES:
            self._mode = mode
        settings = ScanSettings(sample_interval=float(interval), mode=self._mode)
        self._start(self._scan_worker, path, settings)
        return {"started": True}

    def _scan_worker(self, video_path: Path, settings: ScanSettings) -> None:
        """Runs on a worker thread. Only reports; never touches the DOM."""

        def report(fraction: float, timestamp: float) -> None:
            self._emit("onScanProgress", {"fraction": fraction, "timestamp": timestamp})

        def downloading(description: str, fraction: float, done: int, total: int) -> None:
            self._emit(
                "onDownload",
                {"description": description, "fraction": fraction, "done": done, "total": total},
            )

        try:
            result = scan(
                video_path,
                settings=settings,
                on_progress=report,
                cancel=self._cancel,
                on_download=downloading,
            )
        except Cancelled:
            self._emit("onScanCancelled")
            return
        except Exception as error:
            self._emit("onFailed", {"title": "Scan failed", "detail": str(error)})
            return

        if self._scan_result is not None:
            self._scan_result.close()
        self._scan_result = result
        self._selected = None
        self._suggested_filename = ""
        self._emit("onScanned", self._scan_payload(result))

    def _scan_payload(self, result: ScanResult) -> dict:
        return {
            "people": [
                {
                    "index": person.index,
                    "thumbnail": _data_uri(person.thumbnail),
                    "detections": person.detection_count,
                    "firstSeen": person.first_seen,
                    "lastSeen": person.last_seen,
                }
                for person in result.people
            ],
            "frames": result.frame_count,
            "detections": result.detection_count,
            "unassigned": result.unassigned_count,
            "elapsed": result.elapsed_seconds,
            "videoName": result.video_path.name,
        }

    # ----------------------------------------------------------- selection

    def select_person(self, index: int, current_filename: str = "") -> dict:
        """What the reel for this person would be, before committing to it."""
        # The gallery stays visible during an export but must not accept a
        # new selection: the running job already holds its own person, so
        # letting the click through would describe someone it is not
        # cutting.
        if self._busy() or self._scan_result is None:
            return {"accepted": False}

        person = next(
            (p for p in self._scan_result.people if p.index == int(index)), None
        )
        if person is None:
            return {"accepted": False}

        self._selected = person
        _, segments = plan_export(
            person,
            video_duration=self._scan_result.video_duration,
            sample_interval=self._scan_result.sample_interval,
        )
        reel_seconds = sum(s.end_time - s.start_time for s in segments)

        return {
            "accepted": True,
            "index": person.index,
            "cuts": len(segments),
            "reel": _clock(reel_seconds),
            "onScreen": _clock(person.detection_count * self._scan_result.sample_interval),
            "detections": person.detection_count,
            "filename": self._suggest_filename(person, current_filename),
            "summary": (
                f"Person #{person.index + 1} selected - "
                f"{len(segments)} cuts, about {_clock(reel_seconds)} of footage."
            ),
        }

    def _suggest_filename(self, person: Person, current: str) -> str | None:
        """Names the file after the video and the person, if that is free.

        Returns None when the box holds something this method did not put
        there, so a name typed by hand survives clicking through the whole
        gallery. `current` therefore has to come from the page: the rule is
        about what is in the box, not about what was suggested last.

        The video comes from the scan rather than the path box, which can
        have been edited since -- export always cuts the scanned footage,
        so reading the box named the file after footage it does not hold.
        """
        assert self._scan_result is not None
        stem = self._scan_result.video_path.stem or "reel"
        suggestion = f"{stem}-person-{person.index + 1}.mp4"

        hand_typed = current.strip() not in ("", DEFAULT_FILENAME, self._suggested_filename)
        self._suggested_filename = suggestion
        return None if hand_typed else suggestion

    # -------------------------------------------------------------- export

    def start_export(self, folder: str, filename: str, encoder: str, quality: str) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}
        if self._scan_result is None or self._selected is None:
            return {"started": False, "reason": "Choose a person first."}
        if not self._ensure_source_available():
            return {"started": False, "reason": None}

        settings = ExportSettings(
            encoder=encoder,
            quality=quality_for(encoder, quality),
        )
        self._start(self._export_worker, output_path(folder, filename), settings)
        return {"started": True}

    def _ensure_source_available(self) -> bool:
        """Makes sure the scanned footage can still be read, asking if not.

        Checked before the encode starts rather than caught when it fails,
        so a video that has gone missing costs a dialog rather than a
        progress bar that runs partway and then stops.

        On macOS and Linux this almost never fires: the scan holds a
        descriptor, which survives the file being renamed or moved. It is
        the path people lose -- a file moved while the app was closed, or
        a Windows scan, which holds no descriptor on purpose.
        """
        assert self._scan_result is not None
        source = self._scan_result.source
        if source is None or source.is_available():
            return True

        wants_to_locate = self.window.create_confirmation_dialog(
            "Video moved",
            f"FluxCutter cannot find {source.path.name} where it was scanned.\n\n"
            "Locate it to export without scanning again?",
        )
        if not wants_to_locate:
            return False

        chosen = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Video files (*.mp4;*.mov)", "All files (*.*)"),
        )
        if not chosen:
            return False

        try:
            source.relocate(chosen[0])
        except SourceMismatch as error:
            self._emit("onFailed", {"title": "Not the same video", "detail": str(error)})
            return False
        except VideoLoadError as error:
            self._emit("onFailed", {"title": "Cannot use that file", "detail": str(error)})
            return False

        self._emit("onRelocated", {"path": str(source.path)})
        return True

    def _export_worker(self, output_path: Path, settings: ExportSettings) -> None:
        def report(fraction: float, done: int, total: int) -> None:
            self._emit(
                "onExportProgress", {"fraction": fraction, "done": done, "total": total}
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export(
                self._scan_result,
                self._selected,
                output_path,
                settings=settings,
                on_progress=report,
                cancel=self._cancel,
            )
        except Cancelled:
            self._emit("onExportCancelled")
            return
        except Exception as error:
            self._emit("onFailed", {"title": "Export failed", "detail": str(error)})
            return

        self._emit("onExported", {"path": str(output_path)})

    # --------------------------------------------------------------- other

    def cancel(self) -> dict:
        """Stops a scan at the next frame, or an export after this cut."""
        self._cancel.set()
        return {"cancelling": True}

    def shutdown(self) -> dict:
        self._cancel.set()
        if self._scan_result is not None:
            self._scan_result.close()
        return {"ok": True}


def launch(video_path: Path | None = None) -> None:
    """Opens the FluxCutter window."""
    # Before the window exists: macOS reads the bundle name once, and an
    # unbundled Python process is called "Python" in the menu bar and the
    # Dock until told otherwise (app/ui/macos.py).
    set_application_name(WINDOW_TITLE)

    bridge = Bridge(video_path)
    window = webview.create_window(
        WINDOW_TITLE,
        html=_page(),
        js_api=bridge,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=MINIMUM_SIZE,
    )
    bridge.window = window
    window.events.closed += bridge.shutdown
    webview.start()
