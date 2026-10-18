# The nightly run finishes, or leaves its work where it belongs

## What went wrong on the first run

The first run, on issue #42, was started by hand on 17 October. It was
going well. It reproduced both bad cards and found a likely cause: animation
borrows a similarity limit from live action that doesn't fit drawn faces.
Then, ten minutes in, it started an evaluation in the background and waited
for it.

An unattended Claude session ends as soon as Claude stops to wait. So the
session ended, taking the evaluation with it. The script then switched back
to `main`, and the run's unfinished edits (5 files) came along, uncommitted.
That left your `main` with changes you didn't make. It would also have
stopped every later run, since the job never runs on a folder with
uncommitted work. Nothing was written to the issue, so it looked like
nothing had happened.

Those edits are now committed on the run's own branch,
`claude/issue-42-animation-card-merges`, and your `main` is clean.

## What changed

- **Long commands run to the end.** The run is told to wait for every
  command in the foreground, never in the background, and commands may now
  take up to an hour. Measuring test_3 takes about 16 minutes; before, a
  command was cut off at 10.
- **A stopped run picks up where it left off.** If a branch for the issue
  already exists, the next run checks it out and carries on from its last
  commit instead of starting over.
- **Unfinished work stays on its branch.** Whatever the run hasn't
  committed when it ends is committed on its `claude/…` branch as
  "unfinished" and pushed. It is never carried onto `main`. If it ever
  changed files while on some other branch, those changes are stashed, not
  left lying around.
- **The issue is told.** If a run ends without a pull request, the script
  comments on the issue: which branch the work is on, and where the log is.

Nothing about what the run is *allowed* to do has changed, apart from one
addition: it may now switch to an existing `claude/…` branch, so it can
continue one.

## Still needed from you

**The 02:00 job hasn't run yet.** It needs your permission to read the
Documents folder (System Settings → Privacy & Security → Full Disk Access →
add `/bin/zsh`), and the Mac has to be on, or asleep rather than shut down,
at 02:00. Until then you can start a run by hand from Terminal:
`scripts/nightly/run.sh 42`.

**This change only takes effect once it's merged**, since the job runs from
`main`.

## How to undo it

Revert this pull request. The job keeps running with the old script.
