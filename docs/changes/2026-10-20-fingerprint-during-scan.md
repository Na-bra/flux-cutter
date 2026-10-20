# Opening a folder no longer reads every video twice

**Status:** on branch `refactor/fingerprint-during-scan`.

## What changed

When you open a folder, FluxCutter scans each video for people, then looks
for footage that repeats between videos (recaps, openings) so a season reel
doesn't show it twice. To find repeats it was reading every video a second
time, from start to finish, right after the scan had just read it.

Now it fingerprints each video *while the scan reads it*. The second read is
gone. **What it finds is exactly the same.** The fingerprints are identical,
so the repeats are too.

This applies to a folder of two or more videos in the window, and to
`batch --combine` on the command line. A single video isn't affected: it never
looks for repeats, so it doesn't pay for fingerprints.

## Numbers

What the second read cost, against what fingerprinting during the scan costs:

| Video | Second read (before) | During the scan (after) | Same fingerprint |
|---|---|---|---|
| test_3 (22 min) | 22.4 s | 2.8 s | yes |
| animation2 (22 min) | 18.6 s | 3.5 s | yes |
| test | 2.1 s | 0.3 s | yes |
| test_2 | 1.2 s | 0.2 s | yes |

So for a folder of 22-minute episodes, expect about 16–19 s less per episode
the first time you open it. Opening the folder a second time was already
instant and stays that way.

I haven't quoted a before/after for opening a whole folder. The one pair I ran
(176.7 s against 107.1 s) differs by far more than the decoding removed, so
most of that gap was the Mac, not this change.

## What to know

- **Only a complete fingerprint is kept.** If you cancel a scan halfway,
  nothing is saved, and the next scan starts the fingerprint again from the start.
- **Videos you've already scanned on their own** are reused without being read,
  so the first time they're opened as part of a folder they're still
  fingerprinted the old way. After that it's kept, as before.
- **Your kept scans and fingerprints** are untouched and still used.

## How to undo

Revert the commit. Nothing is stored differently, so there's nothing else to undo.
