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
import dataclasses
import io
import json
import sys
import threading
from pathlib import Path

import webview

from app.modes import DEFAULT_MODE, MODES, availability, get_mode, mode_ids
from app.ui.macos import relaunch_from_bundle, set_application_icon, set_application_name
from app.ui.worker import (
    apply_edit,
    frames_at,
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
from app.faces import library
from app.ui import tuning
from app.faces.cast import Answers, CardRef
from app.faces.edits import EditError
from app.ui.folder import (
    CastPerson,
    FolderScan,
    cards_of,
    cast_of,
    detach_cards,
    discard_people,
    join_people,
    together,
    cast_preview_frames,
    export_cast,
    load_answers,
    name_cast_person,
    plan_cast_by_video,
    remember_answer,
    repeated_seconds,
    scan_folder,
)
from app.video import cuts
from app.video.cutter import CutterError, probe_clip
from app.video.loader import PICKER_PATTERN, VideoLoadError
from app.video.source import SourceMismatch
from app.video.timeline import format_timestamp

DEFAULT_OUTPUT_DIR = Path.home() / "Movies"
DEFAULT_FILENAME = "reel.mp4"
# A cut list's pictures. Small: there are two per row and a season reel
# has hundreds of rows, and the question a row answers -- is this the
# right person, does it open mid-turn -- is answerable at this size.
CUT_FRAME_WIDTH = 128
# What a "frame" nudge is worth when the footage will not say. Between
# 24 and 30 fps, so it is a nudge rather than a jump whatever the rate.
FALLBACK_FRAME_SECONDS = 1.0 / 25.0
QUALITY_LEVELS = ["Standard", "High", "Maximum"]
SAMPLE_INTERVALS = [0.25, 0.5, 1.0, 2.0]

WINDOW_TITLE = "FluxCutter"
# The artwork the built app's icons are made from (packaging/make_icon.py).
APP_ICON = Path(__file__).resolve().parents[2] / "packaging" / "icon.png"
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
        # A list, as in one video: a reel can be of several people.
        self._cast_chosen: list[CastPerson] = []
        # "Not them" to a suggested name: which card, by what does not
        # change when the gallery renumbers, and which name. Saved in the
        # people library so it holds after the window closes; this set is
        # the fallback when the library cannot be written.
        self._declined: set[tuple] = set()
        # The reel the current selection would cut, as a list that can be
        # edited (app/video/cuts.py). Rebuilt whenever the selection
        # changes, because it describes that selection's reel and nothing
        # else; an export cuts this rather than planning again.
        self._cuts: list[cuts.Cut] = []
        # Which selection the list belongs to: a cut's video number means
        # a position in the folder in one view and nothing at all in the
        # other, and both views can hold a selection at once.
        self._cuts_view = "video"
        # video position -> how long one frame of it lasts, since probing
        # opens the file and the answer cannot change under us.
        self._frame_seconds: dict[int, float] = {}

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
            "tuning": tuning.describe(self._mode),
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
            file_types=(f"Video files ({PICKER_PATTERN})", "All files (*.*)"),
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
        return {"status": self._availability_text(mode), "tuning": tuning.describe(mode)}

    # ------------------------------------------------------------- tuning

    def _tuned(self, interval: float) -> ScanSettings:
        """The mode's settings, with whatever someone changed in Advanced."""
        return tuning.apply(
            ScanSettings.for_mode(self._mode, sample_interval=interval),
            tuning.changed(self._mode),
        )

    def set_tuning(self, key: str, value=None) -> dict:
        """Changes one Advanced setting for the current mode, or puts it back."""
        if self._busy():
            return {"applied": False, "reason": "Not while a job is running.", "tuning": tuning.describe(self._mode)}
        try:
            tuning.set_value(self._mode, key, value)
        except (KeyError, ValueError, TypeError):
            return {"applied": False, "reason": "That is not a setting.", "tuning": tuning.describe(self._mode)}
        except OSError:
            return {"applied": False, "reason": "The setting could not be saved.", "tuning": tuning.describe(self._mode)}
        return {"applied": True, "tuning": tuning.describe(self._mode)}

    def reset_tuning(self) -> dict:
        """Puts every Advanced setting back to the current mode's own."""
        if self._busy():
            return {"applied": False, "reason": "Not while a job is running.", "tuning": tuning.describe(self._mode)}
        try:
            tuning.reset(self._mode)
        except OSError:
            return {"applied": False, "reason": "The settings could not be saved.", "tuning": tuning.describe(self._mode)}
        return {"applied": True, "tuning": tuning.describe(self._mode)}

    # ---------------------------------------------------------------- scan

    def start_scan(self, video: str, mode: str, interval: float) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}

        path = Path(video.strip()) if video else None
        if path is None or not path.is_file():
            return {"started": False, "reason": "Choose a video first."}

        if mode in MODES:
            self._mode = mode
        settings = self._tuned(float(interval))
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

    # ------------------------------------------------------- suggestions

    @staticmethod
    def _card_identity(result: ScanResult, person: Person) -> tuple:
        return (result.video_path.name, round(person.first_seen, 3), person.detection_count)

    @classmethod
    def _card_key(cls, result: ScanResult, person: Person) -> str:
        """The same identity as a string, the form the library keeps."""
        return "|".join(str(part) for part in cls._card_identity(result, person))

    def _is_declined(self, result: ScanResult, person: Person, name: str, saved: dict) -> bool:
        return (self._card_identity(result, person), name) in self._declined or (
            self._card_key(result, person) in saved.get(name.casefold(), set())
        )

    def _decline(self, result: ScanResult, person: Person, name: str) -> None:
        self._declined.add((self._card_identity(result, person), name))
        library.decline(name, self._card_key(result, person))

    def _suggestions(self, result: ScanResult) -> dict[int, str]:
        """Names from earlier videos for this scan's unnamed cards, by index.

        From the people library (app/faces/library.py), with the matching
        rule used everywhere else. A card someone said "not them" to keeps
        quiet about that name for the rest of the session.
        """
        floor = get_mode(self._mode).grouping.similarity_threshold
        try:
            found = library.suggest([person.group for person in result.people], floor)
        except Exception:
            return {}
        saved = library.declined()
        names = {}
        for suggestion in found:
            person = result.people[suggestion.card]
            if self._is_declined(result, person, suggestion.name, saved):
                continue
            names[person.index] = suggestion.name
        return names

    def decline_suggestion(self, index: int, name: str) -> dict:
        """Says a suggested name is wrong for one card in the video view."""
        if self._busy() or self._scan_result is None:
            return {"applied": False}
        person = next((p for p in self._scan_result.people if p.index == int(index)), None)
        if person is None:
            return {"applied": False}
        self._decline(self._scan_result, person, name)
        return {"applied": True, **self._scan_payload(self._scan_result)}

    def decline_cast_suggestion(self, index: int, name: str) -> dict:
        """Says a suggested name is wrong for one person in the folder view."""
        if self._busy() or self._folder is None:
            return {"applied": False}
        member = next((p for p in self._cast if p.index == int(index)), None)
        if member is None:
            return {"applied": False}
        for video, card in member.appearances:
            self._decline(self._folder.videos[video], card, name)
        return {"applied": True, **self._cast_payload()}

    def _cast_suggestions(self) -> dict[int, str]:
        """A name from earlier videos for each unnamed person in the cast.

        Suggested per video, and a person takes the name any of their cards
        was given, the strongest if more than one. A name only goes to one
        person: if two claim it, neither is sure enough to be offered it.
        """
        assert self._folder is not None
        by_card: dict[tuple[int, int], tuple[str, float]] = {}
        saved = library.declined()
        floor = get_mode(self._folder.settings.mode).grouping.similarity_threshold
        for video, result in enumerate(self._folder.videos):
            try:
                found = library.suggest([p.group for p in result.people], floor)
            except Exception:
                continue
            for suggestion in found:
                person = result.people[suggestion.card]
                if self._is_declined(result, person, suggestion.name, saved):
                    continue
                by_card[(video, person.index)] = (suggestion.name, suggestion.similarity)

        claims: dict[str, list[tuple[float, int]]] = {}
        for member in self._cast:
            if member.name:
                continue
            offered = [by_card[(v, c.index)] for v, c in member.appearances if (v, c.index) in by_card]
            if offered:
                name, score = max(offered, key=lambda o: o[1])
                claims.setdefault(name, []).append((score, member.index))
        return {members[0][1]: name for name, members in claims.items() if len(members) == 1}

    def _scan_payload(self, result: ScanResult) -> dict:
        suggested = self._suggestions(result)
        return {
            "people": [
                {
                    "index": person.index,
                    "name": person.name,
                    "label": person.label,
                    "suggestion": suggested.get(person.index),
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

    # ---------------------------------------------------- the named people

    def named_people(self) -> dict:
        """Everyone named so far, for the People panel.

        Each with a face from every video they were named in, so two
        entries that are the same person look it -- which is how "Coach"
        and "Bald Man" get noticed and merged.
        """
        try:
            everyone = library.known()
        except Exception:
            everyone = []
        return {
            "people": [
                {
                    "name": person.name,
                    "videos": person.videos,
                    "kind": "animation" if person.space.startswith("ccip") else "live action",
                    "faces": [
                        {"video": video, "image": "data:image/jpeg;base64," + picture}
                        for video, picture in person.pictures
                    ],
                    "videoNames": list(person.video_names),
                    # Only someone in the same embedding space: a drawn face
                    # and a filmed one are not comparable, so not mergeable.
                    "mergeInto": [
                        other.name
                        for other in everyone
                        if other.space == person.space and other.name != person.name
                    ],
                }
                for person in everyone
            ]
        }

    def rename_named(self, old: str, new: str) -> dict:
        """Renames someone everywhere, or merges them into someone else.

        A merge asks first, since two people made one cannot be told apart
        again from here -- the faces are pooled under one name.
        """
        if self._busy():
            return {"applied": False, "reason": "Not while a job is running."}
        new = (new or "").strip()
        existing = next(
            (p for p in library.known() if p.name.casefold() == new.casefold()), None
        )
        merging = existing is not None and existing.name.casefold() != old.casefold()
        if merging and self.window is not None:
            if not self.window.create_confirmation_dialog(
                f"Merge {old} into {existing.name}?",
                f"Every face saved as {old} becomes {existing.name}, and the name "
                f"changes on their cards in every video. Later scans will offer "
                f"{existing.name} only.",
            ):
                return {"applied": False, "reason": None}
        try:
            count = library.rename(old, new)
        except EditError as error:
            return {"applied": False, "reason": str(error)}
        if not count:
            return {"applied": False, "reason": f"Nobody called {old} is saved."}
        final = existing.name if merging else new
        self._rename_on_screen(old, final)
        note = f"Merged {old} into {final}." if merging else f"{old} is now {final}."
        return {"applied": True, "note": note, **self.named_people(), **self._redraw()}

    def forget_named(self, name: str) -> dict:
        """Forgets someone, and takes the name off their cards so it stays forgotten."""
        if self._busy():
            return {"applied": False, "reason": "Not while a job is running."}
        person = next((p for p in library.known() if p.name.casefold() == name.casefold()), None)
        if person is None:
            return {"applied": False, "reason": f"Nobody called {name} is saved."}
        if self.window is not None and not self.window.create_confirmation_dialog(
            f"Forget {person.name}?",
            f"The name comes off their cards in {person.videos} "
            f"video{'s' if person.videos != 1 else ''}, and later scans will not "
            f"suggest it. The cards themselves stay.",
        ):
            return {"applied": False, "reason": None}
        library.forget(person.name, from_scans=True)
        self._rename_on_screen(person.name, None)
        return {
            "applied": True,
            "note": f"Forgot {person.name}.",
            **self.named_people(),
            **self._redraw(),
        }

    def _rename_on_screen(self, old: str, new: str | None) -> None:
        """The scans on screen, renamed as the kept ones just were.

        The kept scans were rewritten on disk; these are the copies the
        window drew from, and a correction or an export made next would
        otherwise save the old name straight back.
        """
        def renamed(result: ScanResult) -> ScanResult:
            people = []
            for person in result.people:
                if person.name and person.name.casefold() == old.casefold():
                    person.group.name = new
                    person = dataclasses.replace(person, name=new)
                people.append(person)
            return dataclasses.replace(result, people=people)

        if self._scan_result is not None:
            chosen = {p.index for p in self._selected}
            self._scan_result = renamed(self._scan_result)
            self._selected = [p for p in self._scan_result.people if p.index in chosen]
        if self._folder is not None:
            self._folder = dataclasses.replace(
                self._folder, videos=[renamed(result) for result in self._folder.videos]
            )
            self._rebuild_cast()

    def _redraw(self) -> dict:
        """What each view should now draw, for whichever has something."""
        return {
            "video": self._scan_payload(self._scan_result) if self._scan_result is not None else None,
            "folder": self._cast_payload() if self._folder is not None else None,
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
        self._cuts_view = "video"
        if not self._selected:
            self._cuts = []
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
        # Edits belong to the reel they were made on, so a new selection
        # starts from the plan rather than carrying the last one's
        # dropped rows onto cuts that have nothing to do with them.
        self._cuts = cuts.cuts_from_plans([(0, segments)])
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
        return self._tuned(interval)

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
        # Whoever is selected now is made of different cards than a moment
        # ago, so their reel is a different reel: the cut list is rebuilt
        # rather than left describing the one before the correction.
        self._rebuild_cuts()
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

    # ------------------------------------------------------------ the cuts

    def cut_list(self) -> dict:
        """Every cut the reel is made of, in the order it plays.

        The list is the plan the rail already summarised -- the same cuts
        behind "14 cuts, about 4:31" -- only itemised, so the one that
        opens on the back of someone's head can be found and dropped
        rather than corrected for in the gallery.
        """
        return self._cut_payload()

    def edit_cut(self, index: int, action: str, step: str = "second") -> dict:
        """Drops one cut, puts it back, or moves one of its ends.

        `action` is one of drop, restore, start-, start+, end- and end+;
        `step` is "frame" or "second". The page sends which button was
        pressed and Python works out what that means in seconds, because
        a frame is the video's own frame and the page has no business
        knowing how long one lasts.

        Refused while an export runs: that job was handed its cuts when it
        started, so accepting an edit would change a list the encode is
        not using and say it had taken effect.
        """
        if self._busy() or not self._cuts:
            return {"accepted": False}
        index = int(index)
        if index < 0 or index >= len(self._cuts):
            return {"accepted": False}

        video = self._cuts[index].video
        result = self._scan_for(video)
        if result is None:
            return {"accepted": False}
        delta = self._frame_of(video) if step == "frame" else cuts.STEP_SECONDS

        if action == "drop":
            self._cuts = cuts.drop(self._cuts, index)
        elif action == "restore":
            self._cuts = cuts.restore(self._cuts, index)
        elif action in ("start-", "start+", "end-", "end+"):
            edge = cuts.START if action.startswith("start") else cuts.END
            self._cuts = cuts.move(
                self._cuts,
                index,
                edge,
                delta if action.endswith("+") else -delta,
                video_duration=result.video_duration,
            )
        else:
            return {"accepted": False}

        return self._cut_payload()

    def restore_cuts(self) -> dict:
        """Back to the plan: nothing dropped, no end moved."""
        if self._busy():
            return {"accepted": False}
        self._rebuild_cuts()
        return self._cut_payload()

    def cut_frames(self, indexes=None) -> dict:
        """The first and last frame of each of these cuts.

        Asked for a handful of rows at a time, as they come into view: a
        season reel is hundreds of cuts and two seeks each, which is a
        minute of decoding nobody asked for if it is done up front. The
        ends rather than the middle, because the ends are what the buttons
        beside them move.
        """
        if self._busy() or not self._cuts:
            return {"frames": []}
        wanted = [
            position
            for position in {int(i) for i in (indexes or [])}
            if 0 <= position < len(self._cuts)
        ]
        by_video: dict[int, list[int]] = {}
        for position in sorted(wanted):
            by_video.setdefault(self._cuts[position].video, []).append(position)

        frames = []
        for video, positions in by_video.items():
            result = self._scan_for(video)
            if result is None:
                continue
            asked = []
            for position in positions:
                cut = self._cuts[position]
                # A frame at the very end of a cut is the first frame of
                # what comes next; step back one so the picture is of
                # footage the cut actually contains.
                asked += [cut.start, max(cut.start, cut.end - self._frame_of(video))]
            pictures = frames_at(result, asked, width=CUT_FRAME_WIDTH)
            for position, first, last in zip(positions, pictures[::2], pictures[1::2]):
                frames.append(
                    {
                        "i": position,
                        "start": _data_uri(first) if first is not None else None,
                        "end": _data_uri(last) if last is not None else None,
                    }
                )
        return {"frames": frames}

    def _cut_payload(self) -> dict:
        """The cut list as the page draws it."""
        kept = cuts.kept(self._cuts)
        folder = self._cuts_view == "folder"
        return {
            "accepted": True,
            "cuts": [
                {
                    "i": position,
                    "video": self._video_name(cut.video) if folder else None,
                    "start": format_timestamp(cut.start),
                    "end": format_timestamp(cut.end),
                    "length": f"{cut.seconds:.1f}s",
                    "dropped": cut.dropped,
                    "changed": cut.changed,
                }
                for position, cut in enumerate(self._cuts)
            ],
            "kept": len(kept),
            "dropped": len(self._cuts) - len(kept),
            "planned": len(self._cuts),
            "reel": _clock(cuts.reel_seconds(self._cuts)),
            "edited": any(cut.changed for cut in self._cuts),
        }

    def _rebuild_cuts(self) -> None:
        """The plan for whatever is selected now, with nothing edited."""
        if self._cuts_view == "folder":
            chosen = self._cast_selected
            self._cuts = (
                cuts.cuts_from_plans(plan_cast_by_video(self._folder, chosen))
                if chosen is not None and self._folder is not None
                else []
            )
            return
        if self._scan_result is None or not self._selected:
            self._cuts = []
            return
        _, segments = plan_export(
            self._selected,
            video_duration=self._scan_result.video_duration,
            sample_interval=self._scan_result.sample_interval,
        )
        self._cuts = cuts.cuts_from_plans([(0, segments)])

    def _cut_segments(self) -> list:
        """One video's cut list as segments to encode.

        What the window last showed: untouched this is the plan itself,
        edited it is the plan minus the dropped rows and with the ends
        where they were left. Read before an export starts rather than
        inside the worker, so an edit made while it runs cannot change
        what that run is cutting halfway through.
        """
        return [
            interval
            for _, segments in cuts.plans_from_cuts(self._cuts)
            for interval in segments
        ]

    def _scan_for(self, video: int) -> ScanResult | None:
        """The scan a cut's video position refers to."""
        if self._cuts_view == "folder":
            if self._folder is None or not 0 <= video < len(self._folder.videos):
                return None
            return self._folder.videos[video]
        return self._scan_result

    def _video_name(self, video: int) -> str:
        result = self._scan_for(video)
        return result.video_path.name if result is not None else ""

    def _frame_of(self, video: int) -> float:
        """How long one frame of this video lasts.

        Asked of the footage once and kept: a frame is the smallest useful
        nudge, and what it is worth in seconds is a property of the video,
        not a number this window gets to choose. Footage that cannot be
        probed falls back to a step small enough to be a nudge on any
        ordinary rate.
        """
        if video not in self._frame_seconds:
            result = self._scan_for(video)
            seconds = FALLBACK_FRAME_SECONDS
            if result is not None:
                try:
                    profile = probe_clip(result.source or result.video_path, include_audio=False)
                    seconds = 1.0 / float(profile.frame_rate)
                except (CutterError, ZeroDivisionError):
                    pass
            self._frame_seconds[video] = seconds
        return self._frame_seconds[video]

    # -------------------------------------------------------------- export

    def start_export(self, folder: str, filename: str, encoder: str, quality: str) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}
        if self._scan_result is None or not self._selected:
            return {"started": False, "reason": "Choose a person first."}
        if not self._ensure_source_available():
            return {"started": False, "reason": None}

        segments = self._cut_segments()
        if not segments:
            return {"started": False, "reason": "Every cut has been dropped."}

        settings = ExportSettings(
            video_encoder=encoder,
            quality=quality_for(encoder, quality),
        )
        self._start(self._export_worker, output_path(folder, filename), settings, segments)
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
            file_types=(f"Video files ({PICKER_PATTERN})", "All files (*.*)"),
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

    def _export_worker(self, output_path: Path, settings: ExportSettings, segments) -> None:
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
                segments=segments,
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
        settings = self._tuned(float(interval))
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
        # Answers given the last time this folder was open still hold.
        self._answers = load_answers(scanned)
        self._cast_chosen = []
        self._rebuild_cast()
        self._emit("onFolderScanned", {**self._cast_payload(), "folderName": folder.name})

    def _rebuild_cast(self) -> None:
        assert self._folder is not None
        self._cast, self._questions = cast_of(self._folder, self._answers)
        # Keep the same people chosen by what they are made of, since their
        # positions in the cast move as answers change it. Two chosen people
        # who have just been joined are one choice now.
        chosen = []
        for before in self._cast_chosen:
            wanted = {(v, c.index) for v, c in before.appearances}
            now = next(
                (p for p in self._cast if wanted & {(v, c.index) for v, c in p.appearances}),
                None,
            )
            if now is not None and all(now.index != p.index for p in chosen):
                chosen.append(now)
        self._cast_chosen = sorted(chosen, key=lambda p: p.index)
        if self._cuts_view == "folder":
            self._rebuild_cuts()

    @property
    def _cast_selected(self) -> CastPerson | None:
        """The chosen people as one, for a reel of every scene any is in."""
        return together(self._cast_chosen) if self._cast_chosen else None

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

        suggested = self._cast_suggestions()
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
                    "suggestion": suggested.get(member.index),
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
            "selected": [p.index for p in self._cast_chosen],
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
        remember_answer(self._folder, asked.first, asked.second, bool(same))
        self._rebuild_cast()
        return {
            "applied": True,
            **self._cast_payload(),
            "note": "Joined them into one person." if same else "Kept them apart.",
        }

    def select_cast_person(self, index: int, current_filename: str = "") -> dict:
        """Adds or removes one person, and describes the reel that results.

        A toggle, as in one video: a second person joins the first, and the
        reel is every scene either of them is in, across the folder.
        """
        if self._busy() or self._folder is None:
            return {"accepted": False}
        clicked = next((p for p in self._cast if p.index == int(index)), None)
        if clicked is None:
            return {"accepted": False}

        self._preview_token += 1
        self._cuts_view = "folder"
        if any(p.index == clicked.index for p in self._cast_chosen):
            self._cast_chosen = [p for p in self._cast_chosen if p.index != clicked.index]
        else:
            self._cast_chosen = sorted(self._cast_chosen + [clicked], key=lambda p: p.index)
        if not self._cast_chosen:
            self._cuts = []
            return {"accepted": True, "indexes": [], "summary": "Choose a person to export."}

        chosen = self._cast_selected
        plans = plan_cast_by_video(self._folder, chosen)
        self._cuts = cuts.cuts_from_plans(plans)
        planned = len(self._cuts)
        reel = cuts.reel_seconds(self._cuts)
        repeated = repeated_seconds(self._folder, chosen)
        self._start_cast_preview()

        return {
            "accepted": True,
            "indexes": [p.index for p in self._cast_chosen],
            "token": self._preview_token,
            "name": chosen.label,
            "cuts": planned,
            "reel": _clock(reel),
            "onScreen": _clock(chosen.detection_count * self._folder.settings.sample_interval),
            "detections": chosen.detection_count,
            "videos": len(plans),
            "filename": self._suggest_cast_filename(self._cast_chosen, current_filename),
            "repeated": _clock(repeated) if repeated >= 0.5 else None,
            "summary": (
                f"{chosen.label} selected - {planned} cuts from {len(plans)} "
                f"video{'s' if len(plans) != 1 else ''}, about {_clock(reel)} of footage."
            ),
        }

    def _suggest_cast_filename(self, people: list[CastPerson], current: str) -> str | None:
        """`<folder>-<name>.mp4`, unless the box holds something typed by hand."""
        assert self._folder is not None
        stem = (
            self._folder.videos[0].video_path.parent.name if self._folder.videos else "reel"
        ) or "reel"
        part = "+".join(
            _filename_part(person.name) or f"person-{person.index + 1}" for person in people
        )
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
        if len(self._cast_chosen) != 1:
            return {"applied": False, "reason": "Choose one person to name."}
        try:
            self._folder = name_cast_person(self._folder, self._cast_chosen[0], name)
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

    def cards_of_cast(self, index: int) -> dict:
        """One face per card a person is made of, for choosing what to split off."""
        if self._busy() or self._folder is None:
            return {"cards": []}
        person = next((p for p in self._cast if p.index == int(index)), None)
        if person is None or len(person.appearances) < 2:
            return {"cards": []}
        return {
            "index": person.index,
            "cards": [
                {
                    "card": f"{video}:{card.index}",
                    "video": self._folder.videos[video].video_path.name,
                    "image": _data_uri(card.thumbnail),
                    "onScreen": _clock(card.detection_count * self._folder.settings.sample_interval),
                }
                for video, card in person.appearances
            ],
        }

    def edit_cast(self, operation: str, cards=None) -> dict:
        """Merges, splits or discards people across the whole folder.

        Merge and split are recorded as answers, and kept, so they hold the
        next time the folder is opened. Discard removes the cards from each
        video's kept scan, as it does in one video.
        """
        if self._busy() or self._folder is None:
            return {"applied": False, "reason": "Not while a job is running."}
        if not self._cast_chosen:
            return {"applied": False, "reason": "Choose a person first."}

        chosen = list(self._cast_chosen)
        try:
            if operation == "merge":
                join_people(self._folder, self._answers, chosen)
                note = f"Joined {len(chosen)} people into one."
            elif operation == "split":
                if len(chosen) != 1:
                    return {"applied": False, "reason": "Choose one person to split."}
                refs = []
                for token in cards or []:
                    video, _, card = str(token).partition(":")
                    refs.append(CardRef(int(video), int(card)))
                self._folder = detach_cards(self._folder, self._answers, chosen[0], refs)
                note = f"Split {len(refs)} of their faces off as someone else."
            elif operation == "discard":
                self._folder, refused = discard_people(self._folder, chosen)
                # Discarding renumbers each gallery it touched, so answers
                # held by position are read again from what was kept.
                self._answers = load_answers(self._folder)
                self._cast_chosen = []
                note = "Removed them from every video." + (
                    f" Kept in {', '.join(refused)}: removing them would leave nobody."
                    if refused else ""
                )
            else:
                return {"applied": False, "reason": f"Unknown edit: {operation}"}
        except (EditError, ValueError) as error:
            return {"applied": False, "reason": str(error)}

        self._rebuild_cast()
        self._preview_token += 1
        if operation == "merge" and len(self._cast_chosen) != 1:
            # Never report a join the cast does not show.
            return {
                "applied": False,
                "reason": "They could not be joined: something said about them earlier keeps them apart.",
                **self._cast_payload(),
            }
        return {"applied": True, **self._cast_payload(), "note": note}

    def save_report(self, folder: str) -> dict:
        """Writes the folder's season report beside the reels, and opens it.

        Built from the cast as it stands -- names, merges and answers
        included -- so it says what the window shows.
        """
        if self._busy() or self._folder is None:
            return {"saved": False, "reason": "Scan a folder first."}
        from app.report import season_report, write_csv, write_html

        title = (
            self._folder.videos[0].video_path.parent.name if self._folder.videos else "season"
        ) or "season"
        report = season_report(self._folder, self._answers, title=title)
        directory = Path(folder.strip() or DEFAULT_OUTPUT_DIR).expanduser()
        stem = _filename_part(title) or "season"
        try:
            csv_path = write_csv(report, directory / f"{stem}-report.csv")
            html_path = write_html(report, directory / f"{stem}-report.html")
        except OSError as error:
            return {"saved": False, "reason": f"The report could not be written: {error}"}
        try:
            import webbrowser

            webbrowser.open(html_path.resolve().as_uri())
        except Exception:
            pass
        return {
            "saved": True,
            "html": str(html_path),
            "csv": str(csv_path),
            "people": len(report.rows),
            "videos": len(report.videos),
        }

    def start_folder_export(self, folder: str, filename: str, encoder: str, quality: str) -> dict:
        if self._busy():
            return {"started": False, "reason": "already running"}
        if self._folder is None or self._cast_selected is None:
            return {"started": False, "reason": "Choose a person first."}

        plans = cuts.plans_from_cuts(self._cuts)
        if not plans:
            return {"started": False, "reason": "Every cut has been dropped."}

        settings = ExportSettings(
            video_encoder=encoder,
            quality=quality_for(encoder, quality),
        )
        self._start(self._folder_export_worker, output_path(folder, filename), settings, plans)
        return {"started": True}

    def _folder_export_worker(self, output: Path, settings: ExportSettings, plans) -> None:
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
                plans=plans,
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
    # From a checkout on macOS, start again from a bundle named FluxCutter,
    # or the Dock labels the icon "Python" whatever the process calls
    # itself (app/ui/macos.py). Returns straight away everywhere else.
    relaunch_from_bundle(
        ["-m", "app", "ui", *([str(video_path)] if video_path else [])],
        APP_ICON.with_suffix(".icns"),
    )
    # Before the window exists: macOS reads the bundle name once, and an
    # unbundled Python process is called "Python" in the menu bar and the
    # Dock until told otherwise (app/ui/macos.py).
    set_application_name(WINDOW_TITLE)
    # A built app's icon comes from its bundle; a checkout has no bundle of
    # its own, so the Dock is handed the artwork directly.
    if not getattr(sys, "frozen", False):
        set_application_icon(APP_ICON)

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
