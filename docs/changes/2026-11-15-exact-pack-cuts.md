# Scene packs now cut on the exact frame

**Status:** on branch `feature/exact-pack-cuts`, waiting for you to merge.

## What changed

FluxCutter finds where the show cuts by looking at a frame every half second,
so it only knew each cut to within half a second. To be safe, each piece of a
scene pack started and ended slightly *inside* its shot, and up to half a
second of good footage was left behind at each edge.

Now, when you export a scene pack, FluxCutter reads the few frames around each
edge and moves it to the **exact frame the show cuts on**. Each piece is the
whole shot, first frame to last.

- Nothing changes in how you use it. Export a scene pack as before.
- It adds about 2 seconds to an export.
- Where a cut isn't clear (a dissolve, or two quick cuts in the same half
  second), that edge stays where it was, safely inside the shot. About 1 edge
  in 12.
- An edge you moved by hand in **Edit cuts…** is never touched.
- Reels (with the length box empty) are exported exactly as before.
- On animation, effects near the edge of a piece are no longer mistaken for
  cuts and trimmed away.

## Measured

A 2:00 pack of the lead in a 44-minute episode:

| | Before | After |
|---|---|---|
| Footage in the same 23 pieces | 1:47.8 | **1:59.2** |
| Edges on the exact cut | 0 of 46 | **43 of 46** |
| Extra time to export | — | 2.5 s |
| Flashes of a neighbouring shot | none | none |

How "exact" was checked: at each of the 43 moved edges, the frame before and
after were compared in the original video. Across each edge the picture
changes sharply (at least 34, where 12 counts as a cut). One frame further in,
it barely changes (at most 2.6). So each edge sits exactly on a cut, with
nothing of the next shot included.

The panel's "about" length now allows for the extra footage. In this run it
said about 1:57 and the pack came out 1:59.2.

### On animation

A 2:00 pack from a 28-minute animated episode: 31 edges moved to the exact cut
in 2.1 seconds, each one a sharp change in the source.

That run turned up one more thing to fix. When it cuts, FluxCutter normally
drops any flash of a neighbouring shot at the edge of each piece. On
animation it mistook the show's own effects near an edge (a red flash, speed
lines) for a cut, and threw away 2.4 seconds of real footage. A scene pack's
edges are already on the show's cuts, so that clean-up is now off for packs:

| 2:00 animation pack | Planned | Came out |
|---|---|---|
| With the clean-up | 1:54.1 | 1:51.7 |
| Without it (now) | 1:54.1 | **1:54.1** |

Live action came out the same either way (1:59.2). Reels still get the
clean-up as before.

Those effects (whole-frame flashes, speed lines drawn every couple of frames)
are inside the show's own shots, so they stay in your pack, as they are in
the episode.

## A note on the test videos

New videos now sit under two old names: `test_3.mp4` is the 44-minute
episode measured above, and `animation2.mp4` is a 28-minute animated
episode. The evaluation suite's labels describe the old files, so it rightly
refuses to score these two ("not the file its truth was labelled on"). Scoring
them would need new labels.

## How to undo

Revert the commit. Nothing is stored differently.
