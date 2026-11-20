# The Export panel says how long a scene pack of someone can be

**Status:** on branch `claude/issue-50-longest-pack`, waiting for you to merge.

## What changed

Choose someone and the **Scene pack length** label now tells you the longest
pack that can be made of them, before you type anything:

    SCENE PACK LENGTH — up to 11:53

Until now you only found that out by asking for too much: type `20:00` and the
panel answered "all there is of them in whole shots: about 11:53, short of
20:00". Same figure, said up front.

- It is the total of every whole shot the pack could use, once cut — exactly
  the number the shortfall note has always reported.
- It follows the selection: choose someone else, or a second person, and it
  changes with them.
- In **Folder** view it is the total across every episode they are in, because
  one pack is cut from all of them.
- Nothing else changed. With the length box empty, exporting a reel is the
  same reel it was, cut the same way, to the byte.

## What it does to your videos and your kept scans

Nothing. No file is written differently, no kept scan is stored differently,
and no video is decoded to work the figure out — it comes from where the
camera cuts, which every scan has recorded since v1.17.0, and from the faces
the scan already measured.

**The one exception: a scan kept from before v1.17.0.** Those do not know
where the camera cuts, and the figure cannot exist without that. So the label
says `— working it out…` and FluxCutter reads the video once to find them.
This is the same read the first scene pack of that video always needed; it
has simply moved from "when you ask for a pack" to "when you choose someone".
While it runs you get the usual progress bar and **Cancel**, and the export
button waits for it, as it already did. Once per video, ever — the cuts are
kept afterwards. Cancel it and the figure goes away rather than sitting there
claiming to be working; typing a length asks for the read again.

## Measured

The 44-minute test episode (`test_3.mp4`), its lead — 1530 faces, of 40
people found:

| | Before | After |
|---|---|---|
| Told the longest pack when chosen | — | **up to 11:53** |
| Whole shots of them available | (not shown) | 163 pieces |
| Asking for exactly 11:53 | — | 11:18 in 149 pieces — inside the usual 35.7 s slack |
| Asking for 23:46 | "short of 11:53" | "short of 11:53" — the same figure |
| Reel export with the box empty | unchanged | unchanged |
| Extra time to show the figure | — | none (the cuts were already known) |

The evaluation suite was not re-run: this change touches nothing it measures
— no detection, embedding, tracking, grouping, timeline or export code was
altered. Nothing in `evaluation/baseline.json` can move.

One small speed-up came with it. Working the figure out means finding the
usable shots on every selection, not just when a pack is asked for, and the
old search read the whole face list once per shot — 163 shots against 1530
faces on that episode. It now looks each shot's faces up directly, with the
same answer.

## Anything that got worse

Two things, both only on a scan kept from before v1.17.0:

1. Choosing a person starts that one-off read of the video sooner than it used
   to. The window is busy while it runs — that is a few seconds to half a
   minute depending on the video's length — and you can cancel it.
2. If you cancel it, the figure is not offered for that video again until you
   type a pack length.

A scan made by any recent version is unaffected: the figure is there the
instant you click a face.

## How to undo

Revert the commit. Nothing is stored differently, so there is nothing to
clean up.
