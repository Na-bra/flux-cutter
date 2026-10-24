# A scene pack is one video of whole shots, with clean cuts

**Status:** on branch `feature/scene-pack-one-video`, waiting for you to merge.
It replaces the scene pack that shipped in v1.16.0 (see
[2026-10-22](2026-10-22-scene-pack.md)).

## What changed

v1.16.0's scene pack wrote a folder: one file per scene, plus a reel. That
was the wrong shape. A scene pack is **one continuous video of one character**
that an editor cuts to music and puts their own transitions on. It now is
exactly that:

- **One video.** Type a length (`2:00`), get one file about that long. No
  folder, no clips.
- **Whole shots, cut where the show cuts.** Every piece starts on the show's
  own cut into a shot and ends on its cut out of one. Every join in the pack
  is a plain cut that looks like an edit the show made, and it's a clean place
  for an editor to cut. No dissolves; the transitions are yours to add.
- **Shots that are really theirs.** A shot counts only when the character is
  on screen for most of it. Back-to-back shots they hold stay together as one
  piece, so a conversation plays the way the show edited it.
- **Fewer, longer pieces.** Pieces of 3 seconds or more go in first, so the
  pack doesn't strobe through quick close-ups. Shorter ones only fill in when
  there isn't enough of the longer ones.
- **Their clearest shots, in story order,** as before.
- **About the length you asked,** within 2 seconds or 5%, whichever is more.
  A 2:00 pack lands between 1:54 and 2:06. It stops on a whole shot rather
  than cutting the last piece mid-shot to hit the exact second.

## Using it

Same as before: choose someone, type a length into **Scene pack length** in
the Export panel, press Enter. The panel says how many pieces it took and how
long the pack will be. **Edit cuts…** lists every piece it could have used.
The ones not taken are listed as dropped, so you can swap one in or out.
**Export scene pack** writes one video, named by the file name box.

For a video scanned before this version, the first pack takes a little longer.
FluxCutter reads the video once to find where the camera cuts (about 14 seconds
for a 7-minute episode), shows "Finding where the camera cuts…" with a progress
bar and a Cancel button, and keeps the result. Videos scanned from now on have
their cuts found during the scan itself.

## Measured

| | 22-minute live-action test video | 7-minute animation test video |
|---|---|---|
| Asked for | 2:00 | 1:00 |
| Came out | **1:57.6** | **0:50.1**: all there is of the lead in whole shots, and the panel says so |
| Pieces | 17, from 3.0 to 21.5 s, none under 3 s | 10 |
| Joins that are clean cuts | 16 of 16 | 8 of 9; one between two similar shots |
| The show's own cuts inside pieces | about 22 | about 7 |

How "clean" was checked: for every join, the last frame before it and the first
frame after it were compared, taken straight from the source video. On the live-action
pack the smallest change at a join was 12.4 and the median 37.3, on the same scale
the cutter uses to spot a camera cut (12). So every join is a real change of
shot, not a jump within one. Separately, the finished video was scanned for any
two picture changes within half a second of each other, which would mean a
flash of another shot; there were none.

**Why prefer longer pieces.** Taking strictly the clearest shots made the
same 2:00 pack 28 pieces, 12 of them 3 seconds or less. Taking 3-second-plus
pieces first made it 17 pieces with a median of 5 seconds, for 6% less
clarity.

## Where it gives less than you'd expect

- **Very fast-cut footage.** A trailer or a montage where most shots last
  about a second gives little, because a piece needs at least 1.5 seconds of
  their own shots. On the 23-second test trailer, a 14-second reel of one
  person became a 1.5-second pack. Episodes are fine: the 22-minute test
  episode gave 2 minutes easily.
- **A camera cut FluxCutter missed.** If two people's shots are read as one
  shot, it's mostly neither of them, so neither of them gets it. The trailer
  above has one of these. It errs on the side of leaving footage out rather
  than putting someone else in your pack.

## What it doesn't change

- With the length box empty, exporting a reel is exactly as before.
- Scans give the same people as before. A scan now also records where the
  camera cuts, from the frames it already reads: 3.3 seconds on a 22-minute
  episode's 75-second scan, about 4%. It's kept beside your scans.
- Kept scans, names and corrections are untouched.

## How to undo

Revert the commit. The camera cuts kept beside your scans are only read by
scene packs, so they can stay or be deleted.
