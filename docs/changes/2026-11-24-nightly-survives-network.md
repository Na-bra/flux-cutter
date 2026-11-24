# The nightly run waits out a dropped connection

**Status:** on branch `fix/nightly-survives-network`, waiting for you to merge.

## What changed

Your connection kept dropping on 24 November, and three nightly runs lost
time because of it:

- **2am:** GitHub was unreachable, so the run gave up for the night.
- **Two daytime runs** on #54 stopped partway when Claude's servers stopped
  answering. Each time, the script then failed to save the work to GitHub and
  left your folder on the run's branch.

Now:

1. **At the start, it waits for the internet.** It tries again every 5 minutes
   for up to an hour before giving up for the night.
2. **At the end, it keeps trying to save.** It tries the push five times, a
   minute apart. If all five fail, the work is still committed on this Mac.
   Either way, your folder goes back to `main`.
3. **The next run finds work that never reached GitHub.** It looks for the
   run's branch on this Mac first, then on GitHub.
4. **The Mac stays awake for the whole run,** including the waiting and the
   saving, not just while Claude is working.

## What it means for your videos

Nothing in FluxCutter changes. Only the nightly script does.

## What it can't fix

A run that stops because Claude's servers didn't answer still stops. Its work
is saved and the next run carries on, but it doesn't finish that night.

## How it was checked

- The script parses.
- The retry step was tested on its own, under the same strict settings the
  script uses:
  - a command that fails twice then succeeds;
  - a command that always fails;
  - capturing a command's output. This test caught a bug, now fixed: the
    "retrying" messages would have been read as issue numbers.
- Finding a branch was tested against `claude/issue-54-join-order`, which is
  on this Mac.

Nobody has watched a full run with a dropped connection yet.

## How to undo it

Revert the merge.
