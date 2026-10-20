# Exporting and finding repeats do less pointless work

**Status:** on branch `refactor/expensive-functions`, waiting for you to merge.

## What changed

I profiled every command on real footage to find where FluxCutter's own code
spends its time, as opposed to the face models, the video decoder and the video
encoder, which do most of the work and were already tuned. Two pieces of
FluxCutter's own code were doing work and then throwing the result away. Both are fixed. **Nothing you get out changes:**
the reels and the repeats found are exactly the same as before.

1. **Exporting a reel.** To avoid a flash of the next shot at the start or end
   of each cut, the exporter compares small thumbnails of the frames at each
   cut's edges. It was making a thumbnail of *every* frame, then only looking
   at the ones at the edges. It now makes them only for the frames it compares.
2. **Leaving repeated footage out of season reels** (recaps, openings). Matching
   two episodes tried about 50,000 possible time offsets and counted each one
   in full, even when it obviously couldn't win. Those are now skipped.

## Numbers

| | Before | After |
|---|---|---|
| Thumbnails made, exporting one person from the 22-minute test video (79 cuts) | 9,137 | 2,133 |
| Time spent making them | 14.8 s | 1.9 s |
| Matching every pair of the 14 videos you've opened as a folder (91 pairs) | 2.22 s | 0.43 s |
| Slowest pair (two 22-minute episodes) | 0.25 s | 0.04 s |

How "nothing changes" was checked: both reels were decoded and every frame and
every sound sample hashed. The hashes match. All 91 pairs of repeats came out
identical. All 136 tests that cover the exporter, the repeats code, folders and
modes pass.

A whole export isn't quoted as "X% faster". The same export took anywhere from
93 s to 195 s depending on what else the Mac was doing, and that swing is bigger
than the 13 s saved. On a 79-cut reel, expect about 13 s less.

## What wasn't worth changing

- **Scanning a video**: grouping people is 0.36 s of a 52 s scan. The rest is
  decoding the video and running the face models, both already tuned.
- **Shot detection, the season report and the evaluation suite** also spend
  their time decoding and in the models.

## Worth doing next, separately

Finding repeats decodes each video a second time to fingerprint it, about 13 s
per 22-minute episode, the first time a folder is opened. The scan has already
decoded the same file. Taking the fingerprints during the scan would remove that.
It's left for its own change, because it ties two kinds of kept data together.

## Your scans and videos

Kept scans, names and corrections aren't touched. Nothing to redo.

## How to undo

Revert the commit. Nothing is stored differently, so there's nothing else to undo.
