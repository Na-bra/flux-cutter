# Four fixes from the review of the faces code

**Status:** on branch `fix/faces-review-1-4`, waiting for you to merge.

## What changed

1. **Animation: characters close together are no longer mistaken for one face
   found twice.** The step that removes duplicate boxes was given each box in
   the wrong form, so boxes looked bigger than they were, more so towards the
   bottom right of the frame.
2. **Someone on screen with a person can't be added to that person's card.**
   Profile faces and other groups that fail the "is this a face?" test can be
   rescued onto a neighbouring card. That rescue now refuses a card the group
   shares a frame with, because two faces in one frame are two people.
3. **Rescued faces can't lift a card over the screen-time minimum.** The minimum
   is now checked first, so a tiny card can't survive by gaining non-faces.
4. **Merging a card from an older scan no longer risks losing faces.** Older
   scans didn't record tracks. Merging one with a newer card left the track list
   short, and a later split dropped the faces it didn't cover. Cards already
   saved in that state are safe too: the uncovered faces now count as a track.

## What it means for your videos

**Your results don't change on any of the five test videos.** These are fixes for
cases that are rare or absent in them, not an accuracy gain.

| | Before | After |
|---|---|---|
| test: contamination, cards | 0.00%, 2 | same |
| test_2: contamination, cards | 0.00%, 2 | same |
| animation: faces found, contamination, cards | 344, 1.62%, 8 | same |
| animation2: faces found, contamination, cards | 966, 6.15%, 13 | same |
| test_3: faces found, contamination, cards | 9,813, 1.03%, 40 | same |

Every other number per video and per person is the same too. `s01e01` isn't on
this Mac, so it wasn't measured.

Why nothing moved:

- **Fix 1** changes the outcome in only 2 of animation2's 1,049 frames with
  detections, and in none of animation's 263. It keeps 2 more boxes, and neither
  reaches the final results.
  **The review suggested this bug might explain animation2 missing faces. It
  doesn't.** Those misses have another cause.
- **Fixes 2 and 3** only act when a rejected group is rescued, and on these videos
  none is. test_3 has 13 rejected groups, and none has a card it clearly
  continues, with or without this change.
- **Fix 4** only matters when you merge a card from an older scan.

## Kept scans

Kept scans stay valid, so your names, merges and splits are safe. A scan kept
before this keeps its answer until the video is scanned again.

## How to undo it

Revert the commit. Nothing is stored differently.
