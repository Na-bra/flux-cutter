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
    apply_edit,
    preview_frames,
    track_previews,
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
from app.faces.cast import Answers
from app.faces.edits import EditError
from app.ui.folder import (
    CastPerson,
    FolderScan,
    cast_of,
    cast_preview_frames,
    export_cast,
    name_cast_person,
    plan_cast_export,
    repeated_seconds,
    scan_folder,
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


def _filename_part(name: str | None) -> str:
    """A person's name as it can appear in a filename.

    Spaces become hyphens and the case is lowered, so a name typed as
    "Jamie Lee" is a file called `episode-jamie-lee.mp4`. Characters a
    path cannot carry never reach here -- they are refused when the name
    is set (app/faces/edits.py) rather than quietly dropped now.
    """
    if not name:
        return ""
    return "-".join(name.split()).lower()


def _edit_note(operation: str, indexes: list[int], name: str = "") -> str:
    """What the window says after an edit, in the user's own terms."""
    named = ", ".join(f"#{index + 1}" for index in indexes)
    if operation == "merge":
        return f"Merged {named} into one person."
    if operation == "split":
        return f"Split #{indexes[0] + 1} apart."
    if operation == "rename":
        return (
            f"#{indexes[0] + 1} is now {name}."
            if name.strip()
            else f"Cleared the name on #{indexes[0] + 1}."
        )
    return f"Discarded {named}."


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
        # A list, because a reel can be of more than one person. Empty
        # means nothing is chosen; order is the order they were picked.
        self._selected: list[Person] = []
        # Bumped on every selection change. A filmstrip is built off
        # the main thread and can land after the user has clicked
        # again, so a stale one is dropped rather than drawn over the
        # selection it does not belong to.
        self._preview_token = 0
        # The settings the current scan ran under. Edits are written
        # back to the cache entry those settings key, so they have to
        # be the same ones, not a fresh default that happens to look
        # similar.
        self._scan_settings: ScanSettings | None = None
        self._suggested_filename = ""
        # A folder is its own view with its own state, so opening one does
        # not throw away a single video already scanned, or the reverse.
        self._folder: FolderScan | None = None
        self._answers = Answers()
        self._cast: list[CastPerson] = []
        self._questions: list = []
        self._cast_selected: CastPerson | None = None

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
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=("Video files (*.mp4;*.mov)", "All files (*.*)"),
        )
        if not chosen:
            return {"path": None}
        return {"path": str(chosen[0])}

    def choose_folder(self) -> dict:
        chosen = self.window.create_file_dialog(webview.FileDialog.FOLDER)
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
        settings = ScanSettings.for_mode(self._mode, sample_interval=float(interval))
        self._scan_settings = settings
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
        self._selected = []
        # The last suggestion is deliberately kept. Forgetting it made the
        # file name already in the box look typed by hand, so it survived
        # and a reel of this video was saved under the previous one's name.
        self._emit("onScanned", self._scan_payload(result))

    def _scan_payload(self, result: ScanResult) -> dict:
        return {
            "people": [
                {
                    "index": person.index,
                    "name": person.name,
                    "label": person.label,
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
            # An instant gallery looks like a scan that did not really
            # look, so the window says when it reused one and what that
            # saved rather than leaving the speed unexplained.
            "reused": result.reused,
            "originalSeconds": result.original_seconds,
        }

    # ----------------------------------------------------------- selection

    def select_person(self, index: int, current_filename: str = "") -> dict:
        """Adds or removes one person, and describes the reel that results.

        Clicking is a toggle rather than a replacement, so a second card
        joins the first instead of displacing it: "every scene either lead
        is in" is one reel, and it is a thing people ask for. Clicking the
        only selected card clears the selection.
        """
        # The gallery stays visible during an export but must not accept a
        # new selection: the running job already holds its own people, so
        # letting the click through would describe a reel it is not
        # cutting.
        if self._busy() or self._scan_result is None:
            return {"accepted": False}

        person = next(
            (p for p in self._scan_result.people if p.index == int(index)), None
        )
        if person is None:
            return {"accepted": False}

        if any(chosen.index == person.index for chosen in self._selected):
            self._selected = [
                chosen for chosen in self._selected if chosen.index != person.index
            ]
        else:
            self._selected = sorted(
                self._selected + [person], key=lambda chosen: chosen.index
            )

        self._preview_token += 1
        if not self._selected:
            return {
                "accepted": True,
                "indexes": [],
                "summary": "Choose a person to export.",
            }

        _, segments = plan_export(
            self._selected,
            video_duration=self._scan_result.video_duration,
            sample_interval=self._scan_result.sample_interval,
        )
        reel_seconds = sum(s.end_time - s.start_time for s in segments)
        detections = sum(chosen.detection_count for chosen in self._selected)

        self._start_preview()

        return {
            "accepted": True,
            "token": self._preview_token,
            "indexes": [chosen.index for chosen in self._selected],
            "cuts": len(segments),
            "reel": _clock(reel_seconds),
            # Screen time is summed over the people chosen, so two of them
            # in one shot count twice -- which is why it can exceed the
            # reel length rather than matching it.
            "onScreen": _clock(detections * self._scan_result.sample_interval),
            "detections": detections,
            "name": self._selection_name(),
            "filename": self._suggest_filename(self._selected, current_filename),
            "summary": (
                f"{self._selection_name()} selected - "
                f"{len(segments)} cuts, about {_clock(reel_seconds)} of footage."
            ),
        }

    def _start_preview(self) -> None:
        """Builds the filmstrip for the current selection, off the main thread.

        Not a job in the `_busy` sense: it must not block a scan or an
        export, and cancelling one has no meaning -- it is six seeks. A
        stale result is dropped on arrival instead.
        """
        token = self._preview_token
        chosen = list(self._selected)
        result = self._scan_result
        if result is None or not chosen:
            return

        def build() -> None:
            try:
                frames = preview_frames(result, chosen)
                if token != self._preview_token:
                    return
                self._emit(
                    "onPreview",
                    {
                        "token": token,
                        "frames": [
                            {"at": _clock(timestamp), "image": _data_uri(image)}
                            for timestamp, image in frames
                        ],
                    },
                )
            except Exception:
                # Nothing above this to catch it: an exception escaping a
                # daemon thread prints a traceback the user cannot act on
                # and leaves the strip showing "Reading the reel...".
                # A preview is a convenience, so it simply does not appear.
                self._emit("onPreview", {"token": token, "frames": []})

        threading.Thread(target=build, daemon=True).start()

    def _selection_name(self) -> str:
        """How the chosen people are named in the window's own text.

        With nobody named this stays the compact "People #1 and #2" rather
        than repeating the word for each: numbers are what the cards show,
        and a name is only worth spelling out where there is one.
        """
        if not any(chosen.name for chosen in self._selected):
            numbers = [f"#{chosen.index + 1}" for chosen in self._selected]
            if len(numbers) == 1:
                return f"Person {numbers[0]}"
            if len(numbers) == 2:
                return f"People {numbers[0]} and {numbers[1]}"
            return f"People {', '.join(numbers[:-1])} and {numbers[-1]}"

        labels = [chosen.label for chosen in self._selected]
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return f"{', '.join(labels[:-1])} and {labels[-1]}"

    def _suggest_filename(self, people: list[Person], current: str) -> str | None:
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
        # A named person names the file: `episode-jamie.mp4` says what is
        # in it in a way `episode-person-2.mp4` never did, and the number
        # it replaces is the one that changes every time the gallery is
        # corrected. With nobody named the old compact form is kept --
        # `person-1+3` rather than `person-1+person-3`.
        if any(chosen.name for chosen in people):
            parts = [
                _filename_part(chosen.name) or f"person-{chosen.index + 1}"
                for chosen in people
            ]
            suggestion = f"{stem}-{'+'.join(parts)}.mp4"
        else:
            numbers = "+".join(str(chosen.index + 1) for chosen in people)
            suggestion = f"{stem}-person-{numbers}.mp4"

        hand_typed = current.strip() not in ("", DEFAULT_FILENAME, self._suggested_filename)
        self._suggested_filename = suggestion
        return None if hand_typed else suggestion

    # --------------------------------------------------------------- edits

    def _settings_for_scan(self) -> ScanSettings:
        """The settings the loaded scan was produced under."""
        if self._scan_settings is not None:
            return self._scan_settings
        interval = (
            self._scan_result.sample_interval
            if self._scan_result is not None
            else ScanSettings().sample_interval
        )
        return ScanSettings.for_mode(self._mode, sample_interval=interval)

    def edit_people(self, operation: str, tracks=None, name: str = "") -> dict:
        """Merges, splits or discards the chosen cards.

        Grouping gets most of a video right and still splits one actor
        across two cards, or keeps a logo as a person. This is the recourse
        that used to mean re-tuning thresholds and rescanning.
        """
        if self._busy() or self._scan_result is None:
            return {"applied": False, "reason": "Not while a job is running."}
        if not self._selected:
            return {"applied": False, "reason": "Choose a person first."}

        indexes = [chosen.index for chosen in self._selected]
        try:
            updated = apply_edit(
                self._scan_result,
                self._settings_for_scan(),
                operation,
                indexes,
                [int(track) for track in (tracks or [])],
                name,
            )
        except EditError as error:
            return {"applied": False, "reason": str(error)}

        # The footage handle is shared with the result being replaced, so
        # the old one is dropped rather than closed -- closing it would
        # take the descriptor out from under the export that follows.
        self._scan_result = updated
        if operation == "rename":
            # Membership did not change and the gallery kept its order, so
            # the cards stay selected: being deselected by naming somebody
            # would mean re-picking them to export.
            chosen = set(indexes) | {p.index for p in self._selected}
            self._selected = [p for p in updated.people if p.index in chosen]
        else:
            self._selected = []
        self._preview_token += 1
        return {"applied": True, **self._scan_payload(updated), "note": _edit_note(operation, indexes, name)}

    def tracks_of(self, index: int) -> dict:
        """The tracks in one card, as pictures, for choosing a split point."""
        if self._busy() or self._scan_result is None:
            return {"tracks": []}

        person = next(
            (p for p in self._scan_result.people if p.index == int(index)), None
        )
        if person is None:
            return {"tracks": []}

        return {
            "index": person.index,
            "tracks": [
                {
                    "track": track_index,
                    "at": _clock(timestamp),
                    "image": _data_uri(image),
                }
                for track_index, timestamp, image in track_previews(
                    self._scan_result, person
                )
            ],
        }

    # -------------------------------------------------------------- export

    def start_export(self, folder: str, filename: str, encoder: str, quality: str) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}
        if self._scan_result is None or not self._selected:
            return {"started": False, "reason": "Choose a person first."}
        if not self._ensure_source_available():
            return {"started": False, "reason": None}

        settings = ExportSettings(
            video_encoder=encoder,
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
            webview.FileDialog.OPEN,
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

    # -------------------------------------------------------------- folder

    def start_folder_scan(self, folder: str, mode: str, interval: float) -> dict:
        """Scans every video in a folder, reusing any already scanned."""
        if self._busy():
            return {"started": False, "reason": "already running"}

        path = Path(folder.strip()) if folder else None
        if path is None or not path.is_dir():
            return {"started": False, "reason": "Choose a folder of videos first."}

        if mode in MODES:
            self._mode = mode
        settings = ScanSettings.for_mode(self._mode, sample_interval=float(interval))
        self._start(self._folder_worker, path, settings)
        return {"started": True}

    def _folder_worker(self, folder: Path, settings: ScanSettings) -> None:
        def video(index: int, total: int, path: Path) -> None:
            self._status(f"Scanning {path.name} ({index + 1} of {total})…")

        def report(fraction: float, timestamp: float) -> None:
            self._emit("onScanProgress", {"fraction": fraction, "timestamp": timestamp})

        def downloading(description: str, fraction: float, done: int, total: int) -> None:
            self._emit(
                "onDownload",
                {"description": description, "fraction": fraction, "done": done, "total": total},
            )

        try:
            scanned = scan_folder(
                [folder],
                settings,
                on_video=video,
                on_progress=report,
                cancel=self._cancel,
                on_download=downloading,
                on_status=self._status,
            )
        except Cancelled:
            self._emit("onScanCancelled")
            return
        except Exception as error:
            self._emit("onFailed", {"title": "Scan failed", "detail": str(error)})
            return

        if self._folder is not None:
            self._folder.close()
        self._folder = scanned
        self._answers = Answers()
        self._cast_selected = None
        self._rebuild_cast()
        self._emit("onFolderScanned", {**self._cast_payload(), "folderName": folder.name})

    def _rebuild_cast(self) -> None:
        assert self._folder is not None
        self._cast, self._questions = cast_of(self._folder, self._answers)
        if self._cast_selected is not None:
            # Keep the same person selected by what they are made of, since
            # their position in the cast moves as answers change it.
            wanted = {(v, c.index) for v, c in self._cast_selected.appearances}
            self._cast_selected = next(
                (p for p in self._cast if wanted & {(v, c.index) for v, c in p.appearances}),
                None,
            )

    def _cast_payload(self) -> dict:
        assert self._folder is not None
        videos = self._folder.videos
        interval = self._folder.settings.sample_interval

        def card(video: int, person: Person) -> dict:
            return {
                "video": videos[video].video_path.name,
                "thumbnail": _data_uri(person.thumbnail),
                "label": person.label,
            }

        people = []
        for member in self._cast:
            # The face on the card is the one seen most; the strip beneath
            # it is one face per video, so a card that has mixed two people
            # up shows it at a glance.
            best_video, best = max(member.appearances, key=lambda a: a[1].detection_count)
            seen = {}
            for video, person in member.appearances:
                if video not in seen or person.detection_count > seen[video].detection_count:
                    seen[video] = person
            people.append(
                {
                    "index": member.index,
                    "name": member.name,
                    "label": member.label,
                    "thumbnail": _data_uri(best.thumbnail),
                    "faces": [card(v, seen[v]) for v in sorted(seen)],
                    "videos": len(seen),
                    "onScreen": _clock(member.detection_count * interval),
                    "detections": member.detection_count,
                }
            )

        return {
            "people": people,
            "questions": [
                {
                    "id": position,
                    "first": card(q.first.video, self._person_at(q.first)),
                    "second": card(q.second.video, self._person_at(q.second)),
                }
                for position, q in enumerate(self._questions)
            ],
            "videoCount": len(videos),
            "videos": [result.video_path.name for result in videos],
            "skipped": [
                {"video": path.name, "reason": reason} for path, reason in self._folder.skipped
            ],
            "selected": self._cast_selected.index if self._cast_selected else None,
        }

    def _person_at(self, ref) -> Person:
        result = self._folder.videos[ref.video]
        return next(p for p in result.people if p.index == ref.person)

    def answer_question(self, question: int, same: bool) -> dict:
        """Records yes or no to "same person?", and redraws the cast."""
        if self._busy() or self._folder is None:
            return {"applied": False, "reason": "Not while a job is running."}
        try:
            asked = self._questions[int(question)]
        except (IndexError, ValueError, TypeError):
            return {"applied": False, "reason": "That question has already been answered."}

        self._answers.record(asked.first, asked.second, same=bool(same))
        self._rebuild_cast()
        return {
            "applied": True,
            **self._cast_payload(),
            "note": "Joined them into one person." if same else "Kept them apart.",
        }

    def select_cast_person(self, index: int, current_filename: str = "") -> dict:
        """Chooses one person from the folder's cast, and describes their reel."""
        if self._busy() or self._folder is None:
            return {"accepted": False}
        chosen = next((p for p in self._cast if p.index == int(index)), None)
        if chosen is None:
            return {"accepted": False}

        self._preview_token += 1
        if self._cast_selected is not None and self._cast_selected.index == chosen.index:
            self._cast_selected = None
            return {"accepted": True, "index": None, "summary": "Choose a person to export."}

        self._cast_selected = chosen
        plans = plan_cast_export(self._folder, chosen)
        cuts = sum(len(segments) for _, segments in plans)
        reel = sum(s.end_time - s.start_time for _, segments in plans for s in segments)
        repeated = repeated_seconds(self._folder, chosen)
        self._start_cast_preview()

        return {
            "accepted": True,
            "index": chosen.index,
            "token": self._preview_token,
            "name": chosen.label,
            "cuts": cuts,
            "reel": _clock(reel),
            "onScreen": _clock(chosen.detection_count * self._folder.settings.sample_interval),
            "detections": chosen.detection_count,
            "videos": len(plans),
            "filename": self._suggest_cast_filename(chosen, current_filename),
            "repeated": _clock(repeated) if repeated >= 0.5 else None,
            "summary": (
                f"{chosen.label} selected - {cuts} cuts from {len(plans)} "
                f"video{'s' if len(plans) != 1 else ''}, about {_clock(reel)} of footage."
            ),
        }

    def _suggest_cast_filename(self, person: CastPerson, current: str) -> str | None:
        """`<folder>-<name>.mp4`, unless the box holds something typed by hand."""
        assert self._folder is not None
        stem = (
            self._folder.videos[0].video_path.parent.name if self._folder.videos else "reel"
        ) or "reel"
        part = _filename_part(person.name) or f"person-{person.index + 1}"
        suggestion = f"{stem}-{part}.mp4"
        hand_typed = current.strip() not in ("", DEFAULT_FILENAME, self._suggested_filename)
        self._suggested_filename = suggestion
        return None if hand_typed else suggestion

    def _start_cast_preview(self) -> None:
        token = self._preview_token
        folder, person = self._folder, self._cast_selected
        if folder is None or person is None:
            return

        def build() -> None:
            try:
                frames = cast_preview_frames(folder, person)
                if token != self._preview_token:
                    return
                # "V2 6:43" rather than the file name, which covered the frame
                # it labelled; the name is kept for the tooltip.
                number = {
                    result.video_path.name: position + 1
                    for position, result in enumerate(folder.videos)
                }
                self._emit(
                    "onPreview",
                    {
                        "token": token,
                        "frames": [
                            {
                                "at": f"V{number.get(name, '?')} {_clock(timestamp)}",
                                "video": name,
                                "image": _data_uri(image),
                            }
                            for name, timestamp, image in frames
                        ],
                    },
                )
            except Exception:
                self._emit("onPreview", {"token": token, "frames": []})

        threading.Thread(target=build, daemon=True).start()

    def name_cast_person(self, name: str = "") -> dict:
        """Names the chosen person in every video they are in."""
        if self._busy() or self._folder is None:
            return {"applied": False, "reason": "Not while a job is running."}
        if self._cast_selected is None:
            return {"applied": False, "reason": "Choose a person first."}
        try:
            self._folder = name_cast_person(self._folder, self._cast_selected, name)
        except EditError as error:
            return {"applied": False, "reason": str(error)}

        self._rebuild_cast()
        videos = len(self._cast_selected.videos) if self._cast_selected else 0
        note = (
            f"Named them {name.strip()} in {videos} video{'s' if videos != 1 else ''}."
            if name.strip()
            else "Cleared their name."
        )
        return {"applied": True, **self._cast_payload(), "note": note}

    def start_folder_export(self, folder: str, filename: str, encoder: str, quality: str) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}
        if self._folder is None or self._cast_selected is None:
            return {"started": False, "reason": "Choose a person first."}

        settings = ExportSettings(
            video_encoder=encoder,
            quality=quality_for(encoder, quality),
        )
        self._start(self._folder_export_worker, output_path(folder, filename), settings)
        return {"started": True}

    def _folder_export_worker(self, output: Path, settings: ExportSettings) -> None:
        def report(fraction: float, done: int, total: int) -> None:
            self._emit("onExportProgress", {"fraction": fraction, "done": done, "total": total})

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            _, left_out = export_cast(
                self._folder,
                self._cast_selected,
                output,
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

        self._emit(
            "onExported",
            {
                "path": str(output),
                "leftOut": [{"video": path.name, "reason": reason} for path, reason in left_out],
            },
        )

    # --------------------------------------------------------------- other

    def cancel(self) -> dict:
        """Stops a scan at the next frame, or an export after this cut."""
        self._cancel.set()
        return {"cancelling": True}

    def shutdown(self) -> dict:
        self._cancel.set()
        if self._scan_result is not None:
            self._scan_result.close()
        if self._folder is not None:
            self._folder.close()
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
