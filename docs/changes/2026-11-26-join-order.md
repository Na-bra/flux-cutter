# Choose the order videos are joined in

**Status:** on branch `claude/issue-54-join-order`, waiting for you to
merge. Closes [#54](https://github.com/Na-bra/flux-cutter/issues/54).

## What changed

You could already cut one reel or one scene pack from several videos at
once — a folder of them, or, since 22 November, a set of files picked one
by one out of as many folders as you like. What you could not do was say
which one plays first. The order was whatever the set happened to be in:
name order for a folder, and for a picked set, the order the file dialogs
gave the files back.

**In the window.** Each video in the set under the toolbar is now numbered,
and has **↑ ↓** beside the **×**:

    IN THE SET  [ 1  animation5.mp4  show-a  ↑ ↓ × ]  [ 2  animation6.mp4  show-b  ↑ ↓ × ]
                [ 3  animation7.mp4  show-c  ↑ ↓ × ]
                Joined in this order — use ↑ ↓ to change it

That number is the video's place in the finished file. Press ↑ on
`animation7.mp4` twice and it becomes video 1; a reel or a scene pack of
whoever you have chosen then plays episode 7 first. Episodes picked in
whatever order three dialogs offered them can be put in the order they
should play.

Moving one takes effect at once and **scans nothing again**. The cast stays
exactly as it was, whoever you had chosen stays chosen, and the rail is
read again so it describes the reel the export would now cut. The status
line says where the video went — "animation7.mp4 is now video 1 of 3."

The ↑ on the first video and the ↓ on the last are drawn greyed out, and
the whole row is refused while a job is running, the same rule as adding or
removing a video: an export already in the queue was handed its own cuts
and will write the reel you asked for.

**On the command line.** `batch … --combine` already joined videos in the
order you named them — but the code said otherwise ("a sorted list") and
neither the README nor `--help` mentioned it, so it was not something you
could rely on. It is now written down and covered by tests:

    python -m app batch b/e7.mp4 a/e3.mp4 --person Jamie --combine reel.mp4

plays `e7` first. Name a folder and its own videos are still sorted by
name, since nothing else says which order a season goes in.

**The order also decides which video loses a recap.** A stretch of footage
two episodes share is cut out of whichever of them plays *second*, so that
you see it once. Move an episode and that flips with it — worked out from
what was already found, not by reading the footage again.

## What it does to your videos and your kept scans

Nothing. No video file is read, written or re-encoded by moving a video up
or down the list, and nothing is stored in a new place or a new shape.

- **No video is scanned again.** The same scans are simply held in another
  order. Everything the app keys by a video's position in the set — the
  "Same person?" answers you have given, the people you have chosen, the
  recaps it found, the cut list — is renumbered to follow the videos, in
  memory, in well under a second.
- **Kept scans, fingerprints, remembered answers and named people are
  written exactly as before.** Nothing on disk changed format, so folder
  view, the command line and an older build all keep reading them.
- **Single-video exports are untouched**, as are folder exports left in
  name order: nothing on those paths changed.
- **One thing deliberately does not follow the order.** The suggested file
  name comes from the first video's folder (`show-a-person-1.mp4`), and
  moving a video does *not* rewrite the name box — you may well have typed
  in it. Clear it and click a face if you want the suggestion refreshed.

## The numbers

### Nothing the evaluation suite measures is on this path

The three episodes the feature was measured on, re-run on the branch and
compared with `evaluation/baseline.json`:

| Video | Precision | Recall | Contamination | Cards holding someone else | Splits |
|---|---|---|---|---|---|
| animation5 before | 99.54% | 80.0% | 7.79% | 4/17 | 8 |
| animation5 **after** | **99.54%** | **80.0%** | **7.79%** | **4/17** | **8** |
| animation6 before | 98.92% | 100.0% | 5.41% | 3/21 | 8 |
| animation6 **after** | **98.92%** | **100.0%** | **5.41%** | **3/21** | **8** |
| animation7 before | 98.45% | 50.0% | 3.40% | 2/11 | 1 |
| animation7 **after** | **98.45%** | **50.0%** | **3.40%** | **2/11** | **1** |

Identical to four decimal places, which is what you would expect: no
detection, embedding, tracking, grouping or timeline code was touched, and
the only change outside the window's own layer is a docstring. To take it
again:

    .venv/bin/python -m app evaluate animation5 animation6 animation7

### What the feature does, on real episodes

Three consecutive 28-minute animation episodes, symlinked into three
separate folders and picked one at a time:

| | Before | After |
|---|---|---|
| Order the three are joined in | animation5, 6, 7 — not changeable | **animation5, 6, 7, and changeable** |
| After pressing ↑ on animation7 twice | not possible | **animation7, animation5, animation6** |
| Cost of moving it | — | **no rescan**; the status line answers at once |
| The lead's reel after the move | — | 57 cuts, 3:32, 0:06 of recap left out — **the same reel, reordered** |
| That reel, written out | not possible | **194.9 s, 69.2 MB, in 26.5 s** |
| The cast after the move | — | 34 people, 5 in all three — **unchanged** |
| A 0:30 scene pack of the lead | 28.9 s, 10.3 MB | **28.9 s, 10.3 MB**, and it plays its videos in the set's order |

Joins were checked the way the single-video sync test checks a reel, on
three videos cut in three different orders: every join lands on a whole
frame, nothing is carried in from beyond a cut, and sound stays with
picture to within 0.05 s — the same bound a single video is held to.

Both measurements are tests, so they can be taken again:

    .venv/bin/python -m pytest tests/test_multi_video_episode.py --measure -s
    .venv/bin/python -m pytest tests/test_multi_video.py -k order

The whole suite passes: **997 tests, 3 skipped** (974 and 3 before). The 23
new ones cover the reordering itself, the renumbering of answers, people
and recaps, the refusals, the rendered list of videos, the command line's
order, and the three-video join above.

## Anything that got worse

Nothing measured got worse, and nothing was taken away. Three things worth
knowing:

1. **The 0:30 scene pack in the measurement drew on two of the three
   episodes, not all three.** That is not new and not a fault: a pack is
   the clearest half-minute there is, and half a minute need not touch
   three 28-minute episodes. What the change guarantees is that whatever it
   does draw on, it plays in the set's order. The lead's reel does use all
   three.
2. **The set under the toolbar is now three buttons wide per video.** With
   six or seven videos the row is busier than it was. Each video's row is a
   little taller as a result.
3. **The order is not remembered between sessions.** Close the window and
   the next set you pick starts in the order the dialogs give it.

**One thing is not done.** This has not been driven in the real window —
the nightly run cannot open it. Everything behind the bridge is covered by
tests, and the list of videos is rendered under Node and checked, including
which arrows come out disabled. What wants your eye: that the number and
the two arrows sit comfortably beside the **×** without the row getting
cramped, that holding the pointer over an arrow says the right thing
("Play animation7.mp4 earlier"), that the arrows are big enough to hit, and
that moving a video with a big cast on screen redraws without a visible
flicker.

## Merging this

The branch was started on 24 November, before "the nightly run waits out a
dropped connection" (#68) went in, so GitHub asks you to resolve two
conflicts before you can merge. Both are documentation, both are two
changes adding text in the same place, and in both cases the answer is
**keep both** — nothing either change says contradicts the other:

- `docs/changes/README.md` — two new lines at the top of the list. Keep
  both, newest first: the 2026-11-26 join-order line above the 2026-11-24
  nightly line.
- `Instructions.md` — two new sections at the end. Keep both: #68's
  "### Dropped connections (24 November)", which belongs to section 68,
  and then "## 69. The order videos are joined in".

No code conflicts: #68 only touched `scripts/nightly/run.sh` and those two
documents.

## How to undo

Revert the commit. Nothing is stored differently, so there is nothing to
clean up.

To keep everything else but take the arrows off the screen, delete the two
`data-move` buttons from `drawPicks` in `app/ui/window.html`; the set then
behaves exactly as it did before, in the order the files were picked.
