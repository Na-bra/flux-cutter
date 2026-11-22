# The nightly run keeps the Mac awake until it finishes

**Status:** on branch `fix/nightly-stays-awake`, waiting for you to merge.

## What changed

**The Mac stays awake for the whole run.** The 1:55am wake starts the nightly
run, but nothing kept the Mac awake after that. On 22 November it fell asleep
mid-run, again and again: the run took three hours and spent most of them
asleep, with one 53-minute gap. The Mac now stays awake until the run ends,
then sleeps as usual. Plugged in or on battery, it can't sleep while a run is
going.

**A run that stops early no longer points at `main`.** When a run failed
before making its branch, its comment on the issue said "Its work so far is on
`main`", which sent you looking for work that didn't exist. It now says the run
stopped before making a branch, so the next run starts the issue afresh. A run
that stops after making its branch still names it.

## What it means for your videos

Nothing in FluxCutter changes. Only the nightly script does.

## How it was checked

- The script parses.
- Both issue messages were printed and read.
- The Mac reports a no-sleep hold while a command runs under `caffeinate`.

Nobody has watched a full nightly run with it yet. The first will be the next
2am run after you merge.

## How to undo it

Revert the merge.
