"""The queue of scans and exports, run one at a time.

The window used to run one job and refuse the next while it ran, so three
scene packs of three people meant sitting through each encode before asking
for the following one. This holds the jobs instead and runs them in order,
and a job can be taken out, moved up or down, or cancelled while the rest
wait.

One at a time rather than several at once, and that is the design rather
than a simplification. Two jobs reading one video would be two readers of
one descriptor, which share a file offset and seek each other sideways
(app/video/source.py); and two encodes are two halves of the same machine,
so the second does not finish sooner for having been started early.

Nothing here knows what a scan or an export is: a job is a callable, a
cancel flag and some words to draw it with. That is what lets the order,
the removal, the reordering and the cancellation be tested without
encoding any video at all, and it keeps every decision about footage in
app/ui/worker.py, where it already was.

Like app/ui/worker.py this imports no toolkit, so all of it runs with no
display attached.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

# Which way `JobQueue.move` takes a job.
UP = -1
DOWN = 1


@dataclass
class Job:
    """One scan or export: running, or waiting its turn."""

    id: int
    # "scan", "export" or "shots". The queue's only concession to knowing
    # what is in it, because the window draws the three differently.
    kind: str
    # What the queue calls it, and the one line of detail under it: "Export
    # scene pack" and "jamie-pack.mp4".
    title: str
    detail: str
    # What the status bar says while this one runs.
    status: str
    # The work. Handed its own cancel flag, so cancelling the job that is
    # running cannot stop the one behind it.
    run: Callable[[threading.Event], None]
    cancel: threading.Event = field(default_factory=threading.Event)
    # The scans this job reads. A scan the window has replaced still has
    # footage open, and this is what says it may not be closed yet.
    needs: tuple = ()
    # What two jobs asking for the same thing have in common, so the second
    # is never queued: clicking through four cards would otherwise queue
    # four reads of the same video's camera cuts. None for anything the
    # user asked for by hand -- three exports of one reel are three
    # exports, and it is not this code's place to say otherwise.
    key: tuple | None = None
    state: str = "waiting"
    # What the job raised, if the worker let something through. Workers
    # report their own failures; this is for the ones that do not.
    error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self.state == "running"


class JobQueue:
    """Jobs in the order they will run, with one runner thread behind them.

    Args:
        on_change: Called with no arguments after anything about the queue
            changes -- a job added, started, finished, removed or moved.
            The window redraws the list from `jobs()`.
        on_finished: Called with the job once it is done, cancelled or
            failed, after it has left the queue. Where the window lets go
            of footage no job needs any more.

    Both are called from the runner thread as well as the caller's, and
    neither may raise: a callback that threw would take the runner with it
    and strand everything still waiting.
    """

    def __init__(self, on_change=None, on_finished=None):
        self._lock = threading.Lock()
        # The running job first, when there is one, then the waiting ones in
        # the order they will run. A job leaves the list when it finishes,
        # so this is the queue and nothing else.
        self._jobs: list[Job] = []
        self._runner: threading.Thread | None = None
        self._next_id = 1
        self._on_change = on_change
        self._on_finished = on_finished

    # ----------------------------------------------------------- the queue

    def add(
        self,
        run: Callable[[threading.Event], None],
        kind: str = "scan",
        title: str = "",
        detail: str = "",
        status: str = "",
        needs=(),
        key: tuple | None = None,
    ) -> Job | None:
        """Puts a job at the end of the queue, and starts it if nothing runs.

        Returns the job, or None when `key` names something already queued.
        """
        with self._lock:
            if key is not None and any(job.key == key for job in self._jobs):
                return None
            job = Job(
                id=self._next_id,
                kind=kind,
                title=title,
                detail=detail,
                status=status,
                run=run,
                needs=tuple(needs),
                key=key,
            )
            self._next_id += 1
            self._jobs.append(job)
            self._wake()
        self._changed()
        return job

    def jobs(self) -> list[Job]:
        """The running job, then the waiting ones, in the order they run."""
        with self._lock:
            return list(self._jobs)

    def running(self) -> Job | None:
        with self._lock:
            return next((job for job in self._jobs if job.running), None)

    def waiting(self) -> list[Job]:
        with self._lock:
            return [job for job in self._jobs if not job.running]

    def active(self) -> bool:
        """Whether anything is running or waiting to."""
        with self._lock:
            return bool(self._jobs)

    def holds(self, owned) -> bool:
        """Whether any job still reads this scan. By identity, not equality:
        the question is whether this very footage handle is in use."""
        with self._lock:
            return any(owned is needed for job in self._jobs for needed in job.needs)

    def find(self, job_id: int) -> Job | None:
        with self._lock:
            return next((job for job in self._jobs if job.id == job_id), None)

    def keyed(self, key: tuple) -> Job | None:
        """The queued job with this key, for asking before adding one.

        `add` refuses a repeated key by itself, in silence. This is for the
        caller that owes the user a reason.
        """
        with self._lock:
            return next((job for job in self._jobs if job.key == key), None)

    # --------------------------------------------------------- changing it

    def remove(self, job_id: int) -> Job | None:
        """Takes a waiting job out of the queue, so it never runs.

        Returns the job, or None if there is no such job or it is the one
        running -- that one is cancelled, which is not the same thing and
        has to be asked for by name.
        """
        with self._lock:
            job = next(
                (j for j in self._jobs if j.id == job_id and not j.running), None
            )
            if job is None:
                return None
            self._jobs.remove(job)
        self._changed()
        return job

    def move(self, job_id: int, direction: int) -> bool:
        """Moves a waiting job one place up or down the queue.

        The running job never moves, and nothing moves above it: it has
        already started, so there is no queue position left to win.
        """
        with self._lock:
            waiting = [job for job in self._jobs if not job.running]
            position = next(
                (i for i, job in enumerate(waiting) if job.id == job_id), None
            )
            if position is None:
                return False
            wanted = position + (1 if direction > 0 else -1)
            if not 0 <= wanted < len(waiting):
                return False
            waiting[position], waiting[wanted] = waiting[wanted], waiting[position]
            self._jobs = [job for job in self._jobs if job.running] + waiting
        self._changed()
        return True

    def cancel(self, job_id: int | None = None) -> Job | None:
        """Stops the running job, or the one named.

        A waiting job named here is removed instead: it has nothing to stop.
        Returns the job acted on, or None when there is nothing to cancel.
        With no id this is the window's Cancel button, which means "the one
        that is running".
        """
        with self._lock:
            if job_id is None:
                job = next((j for j in self._jobs if j.running), None)
            else:
                job = next((j for j in self._jobs if j.id == job_id), None)
            if job is None:
                return None
            if job.running:
                job.cancel.set()
            else:
                self._jobs.remove(job)
        self._changed()
        return job

    def stop(self) -> list[Job]:
        """Cancels the running job and drops the rest. For a closing window.

        Returns the jobs that were waiting, which never ran, so their
        footage can be let go of.
        """
        with self._lock:
            dropped = [job for job in self._jobs if not job.running]
            for job in self._jobs:
                job.cancel.set()
            self._jobs = [job for job in self._jobs if job.running]
        self._changed()
        return dropped

    def wait(self, timeout: float | None = None) -> bool:
        """Waits for the queue to empty. For tests, and for nothing else.

        Returns whether it emptied within `timeout`.
        """
        with self._lock:
            runner = self._runner
        if runner is not None:
            runner.join(timeout)
        return not self.active()

    # ---------------------------------------------------------- the runner

    def _wake(self) -> None:
        """Starts the runner if it is not already going. Called under the lock.

        Under the lock on purpose: the runner clears `_runner` under the
        same lock on its way out, so there is no moment where a job is
        added to a queue whose runner has decided to stop.
        """
        if self._runner is not None and self._runner.is_alive():
            return
        self._runner = threading.Thread(target=self._loop, daemon=True)
        self._runner.start()

    def _loop(self) -> None:
        while True:
            with self._lock:
                job = next((j for j in self._jobs if not j.running), None)
                if job is None:
                    # Nothing left. Said under the lock, so `_wake` either
                    # sees a live runner or starts another.
                    self._runner = None
                    return
                job.state = "running"
            self._changed()

            try:
                job.run(job.cancel)
            except BaseException as error:  # noqa: BLE001 - see below
                # A worker reports its own failures; what reaches here is a
                # worker that did not. Kept on the job rather than raised:
                # the runner has a queue behind it, and the jobs waiting
                # did nothing wrong.
                job.error = error

            with self._lock:
                if job in self._jobs:
                    self._jobs.remove(job)
            self._finished(job)
            self._changed()

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _finished(self, job: Job) -> None:
        if self._on_finished is not None:
            self._on_finished(job)
