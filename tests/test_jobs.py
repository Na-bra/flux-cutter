"""The queue of scans and exports: what runs, in what order, and what stops.

A job here is a callable and a cancel flag, which is the whole point of
app/ui/jobs.py keeping it that way: the order, the removal, the reordering
and the cancellation are all proved without decoding a frame or encoding a
file. What a real scan or export does with its turn is tests/test_ui_web.py.
"""

import threading
import time

import pytest

from app.ui import jobs


class Held:
    """A job that runs until the test lets go of it, or cancels it.

    Real work on the queue's own runner thread, rather than a stand-in: a
    job that is "running" because an attribute says so proves nothing about
    what runs when.
    """

    def __init__(self, queue, name="job", **about):
        self.name = name
        self.ran = threading.Event()
        self.let_go = threading.Event()
        self.cancelled = False
        self.job = queue.add(self._run, title=name, **about)

    def _run(self, cancel: threading.Event) -> None:
        self.ran.set()
        while not self.let_go.is_set():
            if cancel.is_set():
                self.cancelled = True
                return
            time.sleep(0.005)

    def release(self) -> None:
        self.let_go.set()


@pytest.fixture
def queue():
    made = jobs.JobQueue()
    yield made
    made.stop()
    made.wait(5)


def ran_in_order(queue, names):
    """Runs jobs that only record their own name, and returns the order."""
    order = []
    for name in names:
        queue.add(lambda cancel, name=name: order.append(name), title=name)
    assert queue.wait(5), "the queue did not empty"
    return order


# ------------------------------------------------------------------- order


def test_jobs_run_one_after_another_in_the_order_they_were_added(queue):
    """Three scene packs of three people, asked for one after another."""
    assert ran_in_order(queue, ["first", "second", "third"]) == [
        "first",
        "second",
        "third",
    ]


def test_a_job_added_while_one_runs_waits_rather_than_starting(queue):
    """Two jobs at once would be two readers of one video, which share a
    file offset and seek each other sideways (app/video/source.py)."""
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")

    assert not second.ran.is_set()
    assert queue.running() is first.job
    assert [job.title for job in queue.waiting()] == ["second"]

    first.release()
    assert second.ran.wait(5)
    second.release()
    assert queue.wait(5)


def test_the_queue_is_empty_once_its_jobs_are_done(queue):
    """A job leaves the list when it finishes, so the list is the queue."""
    ran_in_order(queue, ["one"])

    assert queue.jobs() == []
    assert not queue.active()


def test_a_second_runner_is_started_for_a_job_added_after_the_queue_emptied(queue):
    """The runner returns when nothing is left, so the next add has to start
    another -- otherwise the queue silently stops after its first batch."""
    assert ran_in_order(queue, ["one"]) == ["one"]

    assert ran_in_order(queue, ["two"]) == ["two"]


def test_the_order_is_told_running_first(queue):
    held = Held(queue, "running")
    assert held.ran.wait(5)
    Held(queue, "waiting")

    assert [job.title for job in queue.jobs()] == ["running", "waiting"]
    assert queue.jobs()[0].running and not queue.jobs()[1].running


# ----------------------------------------------------------------- removal


def test_a_job_removed_before_it_starts_never_runs(queue):
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")

    assert queue.remove(second.job.id) is second.job
    first.release()
    assert queue.wait(5)

    assert not second.ran.is_set()


def test_the_running_job_is_not_removed(queue):
    """It is cancelled, which is a different thing and has to be asked for
    by name: removing it would leave the work going with nothing listing it."""
    held = Held(queue, "running")
    assert held.ran.wait(5)

    assert queue.remove(held.job.id) is None
    assert queue.running() is held.job


def test_removing_a_job_that_is_not_there_says_so(queue):
    assert queue.remove(404) is None


# -------------------------------------------------------------- reordering


def test_a_waiting_job_moves_down_the_queue(queue):
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")
    third = Held(queue, "third")

    assert queue.move(second.job.id, jobs.DOWN)
    assert [job.title for job in queue.waiting()] == ["third", "second"]

    for job in (first, second, third):
        job.release()
    assert queue.wait(5)


def test_a_waiting_job_moves_up_the_queue_and_then_runs_sooner(queue):
    """What reordering is for: the pack wanted tonight, ahead of the two
    queued before it."""
    running = Held(queue, "running")
    assert running.ran.wait(5)
    order = []
    for name in ("second", "third"):
        queue.add(lambda cancel, name=name: order.append(name), title=name)
    third = queue.waiting()[-1]

    assert queue.move(third.id, jobs.UP)
    running.release()
    assert queue.wait(5)

    assert order == ["third", "second"]


def test_a_job_does_not_move_past_the_ends_of_the_queue(queue):
    held = Held(queue, "running")
    assert held.ran.wait(5)
    first = Held(queue, "first")
    last = Held(queue, "last")

    assert queue.move(first.job.id, jobs.UP) is False
    assert queue.move(last.job.id, jobs.DOWN) is False
    assert [job.title for job in queue.waiting()] == ["first", "last"]

    for job in (held, first, last):
        job.release()
    assert queue.wait(5)


def test_the_running_job_never_moves(queue):
    """It has started; there is no queue position left to win."""
    held = Held(queue, "running")
    assert held.ran.wait(5)
    Held(queue, "waiting")

    assert queue.move(held.job.id, jobs.DOWN) is False
    assert queue.jobs()[0] is held.job


# ------------------------------------------------------------ cancellation


def test_cancelling_the_running_job_lets_the_next_one_run(queue):
    """And the cancelled one stops: a scan at the next frame, an export
    after the cut it is in the middle of."""
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")

    assert queue.cancel() is first.job
    assert second.ran.wait(5), "the job behind the cancelled one never ran"
    second.release()
    assert queue.wait(5)

    assert first.cancelled
    assert not second.cancelled


def test_each_job_has_its_own_cancel_flag(queue):
    """One flag for the whole queue would have stopped the job behind the
    one cancelled, before it had run a frame."""
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")

    queue.cancel(first.job.id)

    assert first.job.cancel.is_set()
    assert not second.job.cancel.is_set()
    second.release()
    assert queue.wait(5)


def test_cancelling_a_waiting_job_takes_it_out_instead(queue):
    """It has nothing to stop, so "cancel" can only mean "do not run it"."""
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")

    assert queue.cancel(second.job.id) is second.job
    first.release()
    assert queue.wait(5)

    assert not second.ran.is_set()


def test_cancelling_with_an_empty_queue_cancels_nothing(queue):
    assert queue.cancel() is None
    assert queue.cancel(404) is None


def test_stopping_cancels_what_runs_and_drops_what_waits(queue):
    """A closing window: nothing is left running, and nothing waiting is
    started on the way out."""
    first = Held(queue, "first")
    assert first.ran.wait(5)
    second = Held(queue, "second")

    dropped = queue.stop()

    assert [job.title for job in dropped] == ["second"]
    assert queue.wait(5)
    assert first.cancelled and not second.ran.is_set()


# ------------------------------------------------- what the window is told


def test_every_change_is_announced():
    """The window redraws the list from this, so a change it is not told
    about is a queue the user cannot see or act on."""
    changes = []
    told = jobs.JobQueue(on_change=lambda: changes.append(len(told.jobs())))
    held = Held(told, "running")
    assert held.ran.wait(5)
    waiting = Held(told, "waiting")
    told.move(waiting.job.id, jobs.UP)
    told.remove(waiting.job.id)
    held.release()
    assert told.wait(5)

    # Added, started, added, moved (refused, and still announced), removed,
    # finished: the count after each, ending with an empty queue.
    assert changes[0] == 1
    assert changes[-1] == 0
    assert len(changes) >= 5


def test_a_finished_job_is_handed_back_once_it_has_left_the_queue():
    """Where the window lets go of footage no job needs any more, so it has
    to be able to see that this one is gone."""
    done = []
    told = jobs.JobQueue(on_finished=lambda job: done.append((job, told.jobs())))
    told.add(lambda cancel: None, title="one")
    assert told.wait(5)

    job, still_queued = done[0]
    assert job.title == "one"
    assert still_queued == []


def test_a_job_that_raises_does_not_take_the_queue_with_it():
    """Workers report their own failures; this is for the one that does not.
    The jobs waiting behind it did nothing wrong."""
    done = []
    told = jobs.JobQueue(on_finished=done.append)

    def broken(cancel):
        raise RuntimeError("the encoder is not available")

    told.add(broken, title="broken")
    order = []
    told.add(lambda cancel: order.append("after"), title="after")
    assert told.wait(5)

    assert isinstance(done[0].error, RuntimeError)
    assert "the encoder is not available" in str(done[0].error)
    assert order == ["after"]


# ------------------------------------------------------- the same job twice


def test_a_job_with_a_key_already_queued_is_not_queued_again(queue):
    """Clicking through four cards of one set would otherwise queue four
    reads of the same video's camera cuts."""
    first = Held(queue, "shots", key=("shots", 1))
    assert first.ran.wait(5)

    assert queue.add(lambda cancel: None, key=("shots", 1)) is None
    assert queue.keyed(("shots", 1)) is first.job
    assert [job.title for job in queue.jobs()] == ["shots"]


def test_a_key_is_free_again_once_its_job_is_done(queue):
    """The second read of the same video is refused while the first is
    queued, not forever: footage changes, and so does what is asked of it."""
    ran_in_order(queue, ["one"])
    queue.add(lambda cancel: None, key=("shots", 1))
    assert queue.wait(5)

    assert queue.keyed(("shots", 1)) is None
    assert queue.add(lambda cancel: None, key=("shots", 1)) is not None


def test_jobs_without_a_key_are_never_held_to_be_the_same(queue):
    """Three exports of one reel are three exports, and it is not the
    queue's place to say otherwise."""
    order = ran_in_order(queue, ["same", "same", "same"])

    assert order == ["same", "same", "same"]


# -------------------------------------------------- the footage jobs hold


def test_the_footage_a_queued_job_will_read_is_known_to_be_in_use(queue):
    """A new scan replaces the one on screen and would close its footage
    there and then. With a queue it can still be what a queued export is
    going to cut."""
    scan = object()
    other = object()
    held = Held(queue, "export", needs=(scan,))
    assert held.ran.wait(5)

    assert queue.holds(scan)
    assert not queue.holds(other)

    held.release()
    assert queue.wait(5)
    assert not queue.holds(scan)


def test_footage_is_matched_by_identity_rather_than_equality(queue):
    """The question is whether this very handle is in use, and two scans of
    one video can compare equal while only one of them is open."""

    class Same:
        def __eq__(self, other):
            return True

        __hash__ = None

    needed, twin = Same(), Same()
    held = Held(queue, "export", needs=(needed,))
    assert held.ran.wait(5)

    assert queue.holds(needed)
    assert not queue.holds(twin)

    held.release()
    assert queue.wait(5)
