# Up and down buttons for the scene pack length

**Status:** on branch `claude/issue-51-pack-length-steps`, waiting for you to
merge.

## What changed

The **Scene pack length** box now has two small arrows beside it, so a length
can be stepped as well as typed:

    SCENE PACK LENGTH — up to 11:53
    [ 0:45          ] ▲
                      ▼

- Each press moves 15 seconds, or 5 seconds below half a minute. The lengths
  it lands on are 0:05, 0:10 … 0:30, 0:45, 1:00, 1:15, and so on.
- **Up from an empty box gives 0:30**, or the longest pack there is of the
  person when there is less of them than that.
- The steps stop at both ends: **down at 0:05 does nothing**, and **up at the
  longest pack that can be made** — the `up to 11:53` figure the label has
  shown since the last change — **does nothing**. Pressing up walks right onto
  that figure, so the longest pack is now reachable without typing it.
- **The up and down arrow keys in the box do the same thing.** They used to
  only move the text cursor.
- Typing still works exactly as it did: `90`, `1:30`, `2m`, Enter. A length
  typed off the grid is tidied by the first press — 1:07 steps up to 1:15,
  not 1:22.
- Stepping down never turns the pack off. The box is still emptied by hand to
  go back to a whole reel.

One thing you may notice: **the pack is not rebuilt on every press.** It waits
350 ms after the last one. Hold the up arrow key down and the box runs up
through the lengths freely; the pack is built once, for the length you stopped
on. That matters because on a scan made before v1.17.0 a length can start a
one-off read of the video, and a held-down key is dozens of presses.

## What it does to your videos and your kept scans

Nothing. No video is read, nothing is written, and no kept scan is stored or
read differently. The arrows only work out a number and put it in the box; the
pack that number builds is the same pack typing it has always built, cut the
same way, to the byte. With the box empty, a reel export is untouched.

## The numbers

Nothing in `evaluation/baseline.json` can move: no detection, embedding,
tracking, grouping, timeline or export code was touched, so the evaluation
suite was not re-run. What was re-run is the measurement on the real
44-minute episode (`test_3.mp4`, its lead, 1530 faces of 40 people found),
before and after:

| | Before | After |
|---|---|---|
| Longest pack said | up to 11:53 | up to 11:53 |
| Whole shots available | 163 pieces | 163 pieces |
| Asking for exactly 11:53 | 11:18 in 149 pieces | 11:18 in 149 pieces |
| Asking for 23:46 | "short of 11:53" | "short of 11:53" |
| Scan time for that episode | 2 min 16 s | 2 min 16 s |
| Presses from an empty box to 11:53 | — (typing only) | **47** |
| Where the 48th press lands | — | nowhere: 11:53 is the stop |

The whole test suite passes: 895 tests, 2 skipped.

## Anything that got worse

Two small things, both judgement calls rather than faults:

1. **47 presses is a long way to the top of a 44-minute episode.** The arrows
   are for nudging a length you can nearly see, not for crossing twelve
   minutes; typing `11:53` is still one action. If that turns out to be the
   wrong trade, the step is one number — `PACK_STEP_SECONDS` in
   `app/video/scenepack.py`.
2. **The issue asked for 5-second steps below one minute, and this does 5
   seconds below *half* a minute.** The two cannot both be true: the issue's
   own acceptance check says "up from empty gives 0:30, up again gives 0:45",
   which is a 15-second step below one minute. I followed the acceptance
   check, because 5-second steps up to a minute would make 0:30 → 0:35 and
   then twelve presses to get to 1:00. Say the word and it is one constant.

**One thing is not done.** The issue asked for this to be driven in the real
window, not only through the bridge's tests. The nightly run that made it
cannot open the window — the app is not in its list of allowed commands and
there is no browser to load the page in — so that check is still owed. What
wants a look: that the two arrows sit beside the box without crowding the
rail, that holding the up arrow key runs up smoothly, and that the pack
rebuilds once when you stop. Everything on the Python side of the bridge is
covered by tests (`tests/test_ui_web.py`, `tests/test_scenepack.py`, and the
real-episode walk in `tests/test_scenepack_episode.py`).

## How to undo

Revert the commit. Nothing is stored differently and nothing was written, so
there is nothing to clean up. To keep the arrows but change how far they
move, the three numbers are at the top of `app/video/scenepack.py`:
`PACK_STEP_SECONDS` (15), `PACK_SHORT_STEP_SECONDS` (5),
`PACK_SHORT_STEP_BELOW` (30), with `PACK_FIRST_SECONDS` (30) for where up
from an empty box starts.
