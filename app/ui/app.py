"""A small desktop front end for FluxCutter.

The whole app is one screen and one sentence of workflow: pick a video,
scan it for people, click a face, export that person's reel.

Threading is the only structurally interesting part. Scanning a
22-minute video takes minutes and encoding takes minutes more, so both
run on a daemon worker thread while Tk keeps drawing. The two sides
communicate through a Queue that the main thread drains on a timer --
worker code never touches a widget, because Tk is not thread-safe and
the failure mode when you get that wrong is a hang or a crash rather
than an exception anyone can read.

The pipeline itself lives in app/main.py and app/ui/worker.py. Nothing
here decides anything about faces; this module is windows and buttons.
"""

import queue
import sys
import threading
import tkinter
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from app import settings as user_settings
from app.modes import MODES, availability, mode_ids
from app.ui.macos import set_application_name
from app.ui.worker import (
    DEFAULT_QUALITY_LEVEL,
    QUALITY_LEVELS,
    Cancelled,
    ExportSettings,
    Person,
    ScanResult,
    ScanSettings,
    available_encoders,
    export,
    plan_export,
    quality_for,
    scan,
)
from app.models import ModelDownloadError
from app.video.cutter import CutterError
from app.video.loader import VideoLoadError
from app.video.source import SourceMismatch

def _icon_file() -> Path | None:
    """Where the window icon is, frozen or from a checkout.

    PyInstaller unpacks bundled data into a temporary directory it names in
    sys._MEIPASS, so the path that works in development is not the path that
    works in the shipped app.
    """
    root = Path(getattr(sys, "_MEIPASS", "")) if hasattr(sys, "_MEIPASS") else None
    candidates = [
        root / "icon.png" if root is not None else None,
        Path(__file__).resolve().parents[2] / "packaging" / "icon.png",
    ]
    return next((c for c in candidates if c is not None and c.is_file()), None)


# How often the main thread checks the worker's mailbox. Fast enough that
# a progress bar looks live, slow enough to stay off the CPU.
POLL_INTERVAL_MS = 80

CARD_COLUMNS = 4
CARD_THUMBNAIL_SIZE = (112, 112)

# Sampling intervals worth offering. Denser sampling finds people who are
# on screen briefly and costs proportionally more time; it does not make
# grouping better per se, since faces-per-frame is a property of the
# footage rather than of how often it is sampled.
INTERVAL_CHOICES = {
    "0.25s - thorough": 0.25,
    "0.5s - balanced": 0.5,
    "1.0s - quick": 1.0,
    "2.0s - fastest": 2.0,
}
DEFAULT_INTERVAL_LABEL = "0.5s - balanced"

# Matches the CLI's --output default, so the two front ends put their
# reels in the same place unless told otherwise.
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_FILENAME = "reel.mp4"


def _clock(seconds: float) -> str:
    """Formats seconds as a clock time for display: 4:07, or 1:04:07."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class PersonCard(ctk.CTkFrame):
    """One face in the results grid, clickable to select that person."""

    def __init__(self, master, person: Person, on_select):
        super().__init__(master, corner_radius=10, border_width=2)
        self._person = person
        self._on_select = on_select
        self._selected = False

        # Held as an attribute because CTkImage is only weakly held by the
        # label; letting it fall out of scope leaves an empty tile.
        self._image = ctk.CTkImage(
            light_image=person.thumbnail,
            dark_image=person.thumbnail,
            size=CARD_THUMBNAIL_SIZE,
        )

        self._thumbnail = ctk.CTkLabel(self, image=self._image, text="")
        self._thumbnail.pack(padx=10, pady=(10, 6))

        self._title = ctk.CTkLabel(
            self,
            text=f"Person #{person.index + 1}",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._title.pack()

        self._detail = ctk.CTkLabel(
            self,
            text=(
                f"{person.detection_count} detections\n"
                f"{_clock(person.first_seen)} - {_clock(person.last_seen)}"
            ),
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray65"),
        )
        self._detail.pack(padx=10, pady=(0, 10))

        # Bound on the children too: a click landing on the thumbnail or a
        # label would otherwise never reach the frame underneath it.
        for widget in (self, self._thumbnail, self._title, self._detail):
            widget.bind("<Button-1>", self._clicked)

        self.set_selected(False)

    @property
    def person(self) -> Person:
        return self._person

    def _clicked(self, _event=None) -> None:
        self._on_select(self._person)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            self.configure(border_color=("#3a7ebf", "#1f6aa5"), fg_color=("gray86", "gray20"))
        else:
            self.configure(border_color=("gray78", "gray28"), fg_color=("gray92", "gray17"))


class FluxCutterApp(ctk.CTk):
    """The main window."""

    def __init__(self, video_path: Path | None = None):
        # Before super(), which starts Tk: Tk reads the application name once,
        # while building the macOS menu bar. Run from a checkout the answer
        # would otherwise be "Python", after a framework build re-execs this
        # process through a stub app inside the framework (app/ui/macos.py).
        set_application_name("FluxCutter")
        super().__init__()

        self.title("FluxCutter")
        self.geometry("900x760")
        self.minsize(720, 620)
        self._set_window_icon()

        self._messages: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._drain_job: str | None = None
        self._scan_result: ScanResult | None = None
        self._selected: Person | None = None
        self._cards: list[PersonCard] = []
        # Restored from the settings file, so the choice survives a restart.
        self._mode = user_settings.load().mode

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._drain_job = self.after(POLL_INTERVAL_MS, self._drain_messages)

        if video_path is not None:
            self._video_var.set(str(video_path))
            self._set_status("Ready to scan.")

    def _set_window_icon(self) -> None:
        """Puts the app icon on the window itself.

        macOS takes the Dock icon from the bundle and ignores this, but
        Windows draws the title bar and taskbar button from whatever Tk was
        given -- and given nothing, that is Tk's default feather. Cosmetic
        either way, so a missing or unreadable icon is not worth failing a
        launch over.
        """
        icon = _icon_file()
        if icon is None:
            return
        try:
            self.iconphoto(True, tkinter.PhotoImage(file=str(icon)))
        except tkinter.TclError:
            pass

    # ---------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        # Only the results grid grows; the control rows keep their height.
        self.grid_rowconfigure(2, weight=1)

        self._build_source_row()
        self._build_progress_row()
        self._build_results_area()
        self._build_export_row()

    def _build_source_row(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="FluxCutter", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=(16, 8), pady=(14, 0), sticky="w")
        ctk.CTkLabel(
            frame,
            text="Find someone in a video, then cut every appearance into one clip.",
            text_color=("gray40", "gray65"),
        ).grid(row=0, column=1, columnspan=2, padx=8, pady=(14, 0), sticky="w")

        ctk.CTkLabel(frame, text="Video").grid(row=1, column=0, padx=(16, 8), pady=(14, 8), sticky="w")
        self._video_var = ctk.StringVar()
        self._video_entry = ctk.CTkEntry(
            frame, textvariable=self._video_var, placeholder_text="Choose an .mp4 or .mov file"
        )
        self._video_entry.grid(row=1, column=1, padx=8, pady=(14, 8), sticky="ew")
        ctk.CTkButton(frame, text="Browse...", width=100, command=self._browse_video).grid(
            row=1, column=2, padx=(8, 16), pady=(14, 8)
        )

        ctk.CTkLabel(frame, text="Sampling").grid(row=2, column=0, padx=(16, 8), pady=(0, 16), sticky="w")
        controls = ctk.CTkFrame(frame, fg_color="transparent")
        controls.grid(row=2, column=1, columnspan=2, padx=8, pady=(0, 16), sticky="ew")

        self._interval_menu = ctk.CTkOptionMenu(
            controls, values=list(INTERVAL_CHOICES), width=170
        )
        self._interval_menu.set(DEFAULT_INTERVAL_LABEL)
        self._interval_menu.pack(side="left")

        # Content type is the user's call and never inferred from the video.
        # It sits next to sampling rather than behind an Advanced panel
        # because picking the wrong one does not fail loudly -- it quietly
        # returns a gallery of the wrong things.
        self._mode_labels = {MODES[m].display_name: m for m in mode_ids()}
        self._mode_menu = ctk.CTkOptionMenu(
            controls,
            values=list(self._mode_labels),
            width=150,
            command=self._on_mode_changed,
        )
        self._mode_menu.set(MODES[self._mode].display_name)
        self._mode_menu.pack(side="left", padx=(8, 0))

        self._scan_button = ctk.CTkButton(
            controls, text="Scan for people", width=150, command=self._on_scan_clicked
        )
        self._scan_button.pack(side="right", padx=(8, 8))

    def _build_progress_row(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        self._progress = ctk.CTkProgressBar(frame)
        self._progress.set(0)
        self._progress.grid(row=0, column=0, sticky="ew")

        self._status = ctk.CTkLabel(
            frame, text="Choose a video to begin.", anchor="w", text_color=("gray40", "gray65")
        )
        self._status.grid(row=1, column=0, pady=(6, 0), sticky="ew")

    def _build_results_area(self) -> None:
        self._results = ctk.CTkScrollableFrame(self, label_text="People found")
        self._results.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="nsew")
        for column in range(CARD_COLUMNS):
            self._results.grid_columnconfigure(column, weight=1)

        self._empty_label = ctk.CTkLabel(
            self._results,
            text="No scan yet.\nPick a video and press Scan for people.",
            text_color=("gray50", "gray55"),
        )
        self._empty_label.grid(row=0, column=0, columnspan=CARD_COLUMNS, pady=40)

    def _build_export_row(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        self._selection_label = ctk.CTkLabel(
            frame, text="Select a person to export.", anchor="w"
        )
        self._selection_label.grid(
            row=0, column=0, columnspan=3, padx=16, pady=(12, 4), sticky="ew"
        )

        ctk.CTkLabel(frame, text="Folder").grid(row=1, column=0, padx=(16, 8), pady=8, sticky="w")
        self._output_dir_var = ctk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        ctk.CTkEntry(
            frame, textvariable=self._output_dir_var, placeholder_text="Where to save reels"
        ).grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(
            frame, text="Choose...", width=100, command=self._browse_output_dir
        ).grid(row=1, column=2, padx=(8, 16), pady=8)

        ctk.CTkLabel(frame, text="File name").grid(
            row=2, column=0, padx=(16, 8), pady=(0, 8), sticky="w"
        )
        self._filename_var = ctk.StringVar(value=DEFAULT_FILENAME)
        # Remembers what was last filled in automatically, so a name the
        # user typed themselves is never overwritten when they click a
        # different face -- but an untouched one still keeps up.
        self._suggested_filename = DEFAULT_FILENAME
        ctk.CTkEntry(frame, textvariable=self._filename_var).grid(
            row=2, column=1, columnspan=2, padx=(8, 16), pady=(0, 8), sticky="ew"
        )

        options = ctk.CTkFrame(frame, fg_color="transparent")
        options.grid(row=3, column=0, columnspan=3, padx=16, pady=(0, 14), sticky="ew")

        ctk.CTkLabel(options, text="Encoder").pack(side="left", padx=(0, 8))
        encoders = available_encoders()
        self._encoder_menu = ctk.CTkOptionMenu(options, values=encoders, width=180)
        self._encoder_menu.set(encoders[0])
        self._encoder_menu.pack(side="left")

        ctk.CTkLabel(options, text="Quality").pack(side="left", padx=(16, 8))
        self._quality_menu = ctk.CTkOptionMenu(
            options, values=list(QUALITY_LEVELS), width=120
        )
        self._quality_menu.set(DEFAULT_QUALITY_LEVEL)
        self._quality_menu.pack(side="left")

        self._audio_switch = ctk.CTkSwitch(options, text="Keep audio")
        self._audio_switch.select()
        self._audio_switch.pack(side="left", padx=16)

        self._export_button = ctk.CTkButton(
            options, text="Export reel", width=150, command=self._on_export_clicked
        )
        self._export_button.pack(side="right")
        self._export_button.configure(state="disabled")

    # ------------------------------------------------------------- callbacks

    def _browse_video(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.mov"), ("All files", "*.*")],
        )
        if chosen:
            self._video_var.set(chosen)
            self._set_status("Ready to scan.")

    def _browse_output_dir(self) -> None:
        current = Path(self._output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR)
        chosen = filedialog.askdirectory(
            title="Choose a folder for exported reels",
            initialdir=str(current if current.is_dir() else Path.cwd()),
            mustexist=False,
        )
        if chosen:
            self._output_dir_var.set(chosen)

    def _output_path(self) -> Path:
        """Where the next export will be written.

        The folder and the name are separate fields because they change on
        different rhythms: a folder is chosen once for a session's worth of
        reels, while the name wants to follow whichever face is selected.
        """
        directory = Path(self._output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR)
        name = self._filename_var.get().strip() or DEFAULT_FILENAME
        if not name.lower().endswith(".mp4"):
            name = f"{name}.mp4"
        return directory / name

    def _suggest_filename(self, person: Person) -> None:
        """Names the file after the video and the person, if that is free.

        Only replaces a name this method put there itself; anything typed
        by hand survives clicking through the gallery.

        The name comes from the scan's video rather than from the path box,
        which can have been edited since. Export always cuts from the
        scanned video, so reading the box here produced a file named after
        footage it does not contain.
        """
        source = (
            self._scan_result.video_path
            if self._scan_result is not None
            else Path(self._video_var.get().strip())
        )
        video_stem = source.stem or "reel"
        suggestion = f"{video_stem}-person-{person.index + 1}.mp4"

        if self._filename_var.get().strip() == self._suggested_filename:
            self._filename_var.set(suggestion)
        self._suggested_filename = suggestion

    def _on_mode_changed(self, label: str) -> None:
        """Records the chosen content type and reports what it needs.

        Selecting a mode whose weights are missing is allowed -- the scan
        will fetch them, with a progress bar, the same way live action does
        on a fresh install. What is not allowed is finding out only after a
        long scan, so the requirement is stated here.
        """
        self._mode = self._mode_labels.get(label, self._mode)

        remembered = user_settings.load()
        remembered.mode = self._mode
        user_settings.save(remembered)

        state = availability(self._mode)
        if state.usable:
            self._set_status(f"{MODES[self._mode].display_name} mode. Ready to scan.")
        else:
            self._set_status(f"{MODES[self._mode].display_name} mode. {state.describe()}.")

    def _on_scan_clicked(self) -> None:
        if self._is_busy():
            self._cancel.set()
            self._set_status("Stopping after the current frame...")
            return

        video_path = self._video_var.get().strip()
        if not video_path:
            messagebox.showwarning("No video", "Choose a video file first.")
            return
        if not Path(video_path).exists():
            messagebox.showerror("Not found", f"No such file:\n{video_path}")
            return

        spec = MODES[self._mode]
        settings = ScanSettings(
            sample_interval=INTERVAL_CHOICES[self._interval_menu.get()],
            mode=self._mode,
            confidence_threshold=spec.detection.confidence_threshold,
            min_confidence=spec.detection.min_confidence,
            min_face_size=spec.detection.min_face_size,
            similarity_threshold=spec.grouping.similarity_threshold,
            consolidation_threshold=spec.grouping.consolidation_threshold,
            min_group_eye_span=spec.grouping.min_group_eye_span,
        )

        self._clear_results()
        self._set_busy(True, scanning=True)
        self._set_status("Starting scan...")
        self._start_worker(self._scan_worker, Path(video_path), settings)

    def _on_export_clicked(self) -> None:
        if self._is_busy():
            self._cancel.set()
            self._set_status("Stopping after the current cut...")
            return

        if self._scan_result is None or self._selected is None:
            return

        if not self._ensure_source_available():
            return

        output_path = self._output_path()
        if output_path.exists() and not messagebox.askyesno(
            "Overwrite?", f"{output_path.name} already exists. Replace it?"
        ):
            return

        encoder = self._encoder_menu.get()
        settings = ExportSettings(
            video_encoder=encoder,
            quality=quality_for(encoder, self._quality_menu.get()),
            include_audio=bool(self._audio_switch.get()),
        )

        self._set_busy(True, scanning=False)
        self._set_status("Preparing cuts...")
        self._start_worker(
            self._export_worker, self._scan_result, self._selected, output_path, settings
        )

    def _ensure_source_available(self) -> bool:
        """Makes sure the scanned footage can still be read, asking if not.

        Checked before the encode starts rather than caught when it fails,
        so a video that has gone missing costs a dialog rather than a
        progress bar that runs to "Preparing cuts..." and then stops.

        On macOS and Linux this almost never fires: the scan holds a
        descriptor, which survives the file being renamed, moved, or
        deleted. It is the path that people actually lose -- a file moved
        while the app was closed, or a Windows scan, which holds no
        descriptor on purpose (app/video/source.py).

        Returns:
            True if the export may go ahead.
        """
        assert self._scan_result is not None
        source = self._scan_result.source
        if source is None or source.is_available():
            return True

        if not messagebox.askyesno(
            "Video moved",
            f"FluxCutter cannot find {source.path.name} where it was scanned:\n\n"
            f"{source.path.parent}\n\n"
            "Locate it to export without scanning again?",
        ):
            return False

        chosen = filedialog.askopenfilename(
            title=f"Where is {source.path.name}?",
            initialfile=source.path.name,
            filetypes=[("Video files", "*.mp4 *.mov"), ("All files", "*.*")],
        )
        if not chosen:
            return False

        try:
            source.relocate(chosen)
        except SourceMismatch as error:
            messagebox.showerror("Not the same video", str(error))
            return False
        except VideoLoadError as error:
            messagebox.showerror("Cannot use that file", str(error))
            return False

        # The path box is what the user reads to know what is loaded, so it
        # should not go on naming a folder the video left.
        self._video_var.set(str(source.path))
        self._set_status(f"Found it. Exporting from {source.path.parent}.")
        return True

    def _on_person_selected(self, person: Person) -> None:
        # The gallery stays visible during an export but must not accept a
        # new selection: the running job already holds its own person, so
        # letting the click through changed the label and the filename to
        # describe someone the encode is not cutting.
        if self._is_busy():
            return

        self._selected = person
        for card in self._cards:
            card.set_selected(card.person.index == person.index)
        self._suggest_filename(person)

        assert self._scan_result is not None
        _, segments = plan_export(
            person,
            video_duration=self._scan_result.video_duration,
            sample_interval=self._scan_result.sample_interval,
        )
        reel_seconds = sum(s.end_time - s.start_time for s in segments)
        self._selection_label.configure(
            text=(
                f"Person #{person.index + 1} selected - "
                f"{len(segments)} cuts, about {_clock(reel_seconds)} of footage."
            )
        )
        self._export_button.configure(state="normal")

    def _on_close(self) -> None:
        # A running encode holds an ffmpeg subprocess; ask it to stop, but
        # do not block the close on it. The worker is a daemon, so the
        # process exits regardless.
        self._cancel.set()
        # The drain loop reschedules itself, so without this the timer
        # fires once more against a half-destroyed window.
        if self._drain_job is not None:
            self.after_cancel(self._drain_job)
            self._drain_job = None
        if self._scan_result is not None:
            self._scan_result.close()
        self.destroy()

    # ---------------------------------------------------------------- workers

    def _start_worker(self, target, *args) -> None:
        self._cancel.clear()
        self._worker = threading.Thread(target=target, args=args, daemon=True)
        self._worker.start()

    def _scan_worker(self, video_path: Path, settings: ScanSettings) -> None:
        """Runs on the worker thread. Only posts to the queue."""
        def report(fraction: float, timestamp: float) -> None:
            self._messages.put(
                ("progress", fraction, f"Scanning {_clock(timestamp)}...")
            )

        def downloading(description: str, fraction: float, done: int, total: int) -> None:
            self._messages.put((
                "progress",
                fraction,
                f"First run: downloading the {description} "
                f"({done / 1_000_000:.0f} of {total / 1_000_000:.0f} MB)...",
            ))

        try:
            result = scan(
                video_path,
                settings,
                on_progress=report,
                cancel=self._cancel,
                on_download=downloading,
            )
        except Cancelled:
            self._messages.put(("cancelled", "Scan stopped."))
        except ModelDownloadError as error:
            self._messages.put(("error", "Could not get the face models", str(error)))
        except (VideoLoadError, ValueError) as error:
            self._messages.put(("error", "Could not scan that video", str(error)))
        except Exception as error:  # noqa: BLE001 - a UI must not die silently
            self._messages.put(("error", "Scan failed", f"{type(error).__name__}: {error}"))
        else:
            self._messages.put(("scan_done", result))

    def _export_worker(
        self,
        scan_result: ScanResult,
        person: Person,
        output_path: Path,
        settings: ExportSettings,
    ) -> None:
        """Runs on the worker thread. Only posts to the queue."""
        def report(fraction: float, done: int, total: int) -> None:
            self._messages.put(("progress", fraction, f"Encoding cut {done} of {total}..."))

        try:
            result = export(
                scan_result,
                person,
                output_path,
                settings,
                on_progress=report,
                cancel=self._cancel,
            )
        except Cancelled:
            self._messages.put(("cancelled", "Export stopped."))
        except CutterError as error:
            self._messages.put(("error", "Could not export", str(error)))
        except Exception as error:  # noqa: BLE001 - a UI must not die silently
            self._messages.put(("error", "Export failed", f"{type(error).__name__}: {error}"))
        else:
            self._messages.put(("export_done", result))

    # ------------------------------------------------------- message pumping

    def _drain_messages(self) -> None:
        """Applies whatever the worker posted. Main thread only.

        Drains the whole queue each tick rather than one item, so a burst
        of progress updates cannot fall behind the work producing them.
        """
        try:
            while True:
                message = self._messages.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
        finally:
            self._drain_job = self.after(POLL_INTERVAL_MS, self._drain_messages)

    def _handle_message(self, message: tuple) -> None:
        kind = message[0]

        if kind == "progress":
            _, fraction, text = message
            self._progress.set(fraction)
            self._set_status(text)

        elif kind == "scan_done":
            result: ScanResult = message[1]
            self._scan_result = result
            self._show_people(result)
            self._set_busy(False)
            self._progress.set(1.0)
            self._set_status(
                f"Found {len(result.people)} "
                f"{'person' if len(result.people) == 1 else 'people'} in "
                f"{result.frame_count} frames "
                f"({result.detection_count} detections, "
                f"{result.unassigned_count} unassigned) "
                f"in {_clock(result.elapsed_seconds)}."
            )

        elif kind == "export_done":
            result = message[1]
            self._set_busy(False)
            self._progress.set(1.0)
            self._set_status(
                f"Wrote {result.output_path.name} - "
                f"{_clock(result.exported_seconds)} from {result.segment_count} cuts "
                f"in {_clock(result.encode_seconds)}."
            )
            messagebox.showinfo("Export complete", f"Saved to:\n{result.output_path}")

        elif kind == "cancelled":
            self._set_busy(False)
            self._progress.set(0)
            self._set_status(message[1])

        elif kind == "error":
            _, title, detail = message
            self._set_busy(False)
            self._progress.set(0)
            self._set_status(f"{title}: {detail}")
            messagebox.showerror(title, detail)

    # ------------------------------------------------------------ ui helpers

    def _is_busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _set_busy(self, busy: bool, scanning: bool = True) -> None:
        """Turns the action buttons into Cancel while work is running."""
        if busy:
            if scanning:
                self._scan_button.configure(text="Cancel")
                self._export_button.configure(state="disabled")
            else:
                self._export_button.configure(text="Cancel")
                self._scan_button.configure(state="disabled")
            self._set_settings_enabled(False)
        else:
            self._scan_button.configure(text="Scan for people", state="normal")
            self._export_button.configure(
                text="Export reel",
                state="normal" if self._selected is not None else "disabled",
            )
            self._set_settings_enabled(True)
            self._worker = None

    def _set_settings_enabled(self, enabled: bool) -> None:
        """Freezes every knob while a job runs.

        The running job captured its settings when it started, so a control
        that still moves is telling the user something untrue about what is
        happening.
        """
        state = "normal" if enabled else "disabled"
        for widget in (
            self._interval_menu,
            self._mode_menu,
            self._encoder_menu,
            self._quality_menu,
            self._audio_switch,
        ):
            widget.configure(state=state)

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)

    def _clear_results(self) -> None:
        for card in self._cards:
            card.destroy()
        self._cards.clear()
        self._selected = None
        # A scan holds an open descriptor on its footage; dropping the
        # reference without closing it leaks one per scan, and on Windows
        # would keep the user's own file locked after they moved on.
        if self._scan_result is not None:
            self._scan_result.close()
        self._scan_result = None
        self._export_button.configure(state="disabled")
        self._selection_label.configure(text="Select a person to export.")
        self._empty_label.grid()

    def _show_people(self, result: ScanResult) -> None:
        self._clear_results()
        self._scan_result = result

        if not result.people:
            self._empty_label.configure(
                text=(
                    "No one appeared for long enough to report.\n"
                    f"Identities need at least {result.min_detections} detections "
                    f"({result.min_detections * result.sample_interval:.0f}s on screen)."
                )
            )
            return

        self._empty_label.grid_remove()
        for position, person in enumerate(result.people):
            card = PersonCard(self._results, person, self._on_person_selected)
            card.grid(
                row=position // CARD_COLUMNS,
                column=position % CARD_COLUMNS,
                padx=8,
                pady=8,
                sticky="nsew",
            )
            self._cards.append(card)


def launch(video_path: Path | None = None) -> None:
    """Opens the FluxCutter window, optionally with a video preloaded."""
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    try:
        app = FluxCutterApp(video_path=video_path)
    except tkinter.TclError as error:
        print(f"Error: could not open a window ({error}).", file=sys.stderr)
        sys.exit(1)

    app.mainloop()


if __name__ == "__main__":
    launch()
