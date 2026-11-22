# Add more than one video to the window

**Status:** on branch `claude/issue-52-several-videos`, waiting for you to
merge. Closes [#52](https://github.com/Na-bra/flux-cutter/issues/52).

## What changed

The source switch at the top of the window had **Video** and **Folder**. It
now has three:

    SOURCE VIDEO            Video  Videos  Folder

**Videos** is a set of files you pick yourself, from as many folders as you
like. Press **Add…** and choose one video, or several at once; press it again
for a file somewhere else. Episodes 3, 7 and 12 of a show kept in three
places are one set.

The set is listed under the toolbar, with the folder each video came from:

    IN THE SET   [ animation5.mp4  show-a  × ]  [ animation6.mp4  show-b  × ]

Press **Scan for people** and it behaves exactly as folder view does today:
one card per person across every video, a face from each video under the
card, the **Same person?** question where a link is not clear, a name given
once naming them in every video, recaps left out of the reel, and one reel
or scene pack cut from the lot. **Report** works the same way too.

Two things the folder view cannot do:

- **Add a video to a set that is already scanned.** Only the new video is
  scanned — the others keep the scans they are holding — and its people join
  the people already on screen. Whoever you had chosen stays chosen, and
  their reel grows to take in the new video: the cuts, the reel length and
  the longest scene pack in the rail are all read again.
- **Take a video out again.** The **×** beside it removes it and its cards
  from the cast. Where the cast had joined one person across that video and
  another, they become two people again — which is right; it is only ever
  shown what is in the set.

Removing the last video in a scanned set is refused ("This is the only
video left. Add another first, or switch to Video"), the same rule as "not a
person" never emptying a gallery.

**Folder view and single-video view are untouched.** There is one gallery of
people, though, so whichever of **Videos** and **Folder** you scanned last
is the one holding it; switching to the other shows "Press Scan for people"
rather than a cast the app no longer has.

## What it does to your videos and your kept scans

Nothing is written to your videos, and nothing about kept scans changed.

- **A video with a kept scan is not scanned again.** Scanning a set reuses
  every kept scan it can, exactly as the folder view does; adding a video to
  a set skips the videos already in it entirely, and the one being added
  reuses its own kept scan if it has one. In the measurement below, opening
  the same three videos as a folder after picking them took **0.4 seconds**,
  because all three scans were reused.
- **Kept scans are shared with the folder view and the command line.** A
  video scanned as part of a picked set is scanned under the same settings
  and keyed the same way, so `python -m app report`, `batch`, folder view
  and single-video view all get that work for free afterwards.
- **Your named people are untouched** except in the ways naming already
  works: naming someone in a picked set writes the name into each of those
  videos' kept scans, as it does in folder view.
- **Your answers to "Same person?" are kept** as before, beside the scans,
  and an answer you gave in this window survives a video joining the set.

## The numbers

Nothing in `evaluation/baseline.json` can move: no detection, embedding,
tracking, grouping, timeline or export code was touched. The change is in
the window's own layer (`app/ui/web.py`, `app/ui/window.html`) and in how a
set of videos is assembled (`app/ui/folder.py`), which the evaluation suite
does not use. So the suite was not re-run.

What was measured instead is the thing the feature promises: that picking
files finds what putting them in one folder finds. Three consecutive
28-minute animation episodes (`animation5`, `animation6`, `animation7`),
symlinked into three separate folders and picked one at a time:

| | Before (one folder) | After (three folders, picked) |
|---|---|---|
| People across the three | 34 | **34** |
| In all three videos | 5 | **5** |
| Person-for-person the same cast | — | **yes** |
| Picking files from different folders | not possible | works |
| Scanning the first two | — | 476.2 s |
| Adding the third to the scanned set | not possible | **162.4 s** (one scan, not three) |
| Opening the same three as a folder afterwards | — | 0.4 s, all three scans reused |
| The lead across all three | — | 57 cuts, reel 3:32, 0:06 of recap left out |
| A 0:30 scene pack of the lead | — | 28.9 s, 10.3 MB, written in 2.8 s |
| After removing the third video | — | 32 people across 2 videos |

That measurement is a test, so it can be taken again:

    .venv/bin/python -m pytest tests/test_multi_video_episode.py --measure -s

The whole test suite passes: **931 tests, 3 skipped** (895 passed and 2
skipped before this change; 37 new tests, one of them the measurement
above, which an ordinary run skips).

## Anything that got worse

Three things, all small, and none of them affects a video you already have:

1. **One gallery, two views that can fill it.** Scan a folder and the picked
   set's cast is gone from the screen (and the reverse). The window says so
   rather than showing you people it can no longer act on, but if you work
   in both views you will rescan — which, with kept scans, is the 0.4
   seconds above rather than minutes.
2. **Two videos with the same file name are told apart only by order.** Pick
   `e3.mp4` from two different shows and the face strip under a card, the
   cut list and the filmstrip all just say `e3.mp4`; the filmstrip numbers
   them V1, V2 in the order you added them. The set under the toolbar does
   show each video's folder.
3. **The suggested file name and the report's title still come from the
   first video's folder.** A reel of a set picked out of `show-a`,
   `show-b` and `show-c` is suggested as `show-a-person-1.mp4`.

**One thing is not done.** This has not been driven in the real window — the
nightly run cannot open it. Everything on the Python side of the bridge is
covered by tests (36 new ones in `tests/test_folder.py` and
`tests/test_ui_web.py`), the page's JavaScript is now checked for syntax
errors by a test, and a test holds every call the page makes to a method
that exists. What wants your eye: that the three-way switch reads well, that
the list of videos under the toolbar sits comfortably when there are six or
seven of them, that **Add…** opens a dialog that takes more than one file,
and that the **×** buttons are big enough to hit.

## How to undo

Revert the commit. Nothing is stored differently, so there is nothing to
clean up: kept scans, fingerprints, remembered answers and named people are
all written in the same formats and places as before, and folder view keeps
reading them.

To keep the feature but hide the view, remove the one line in
`app/ui/window.html` that adds the third button
(`<button type="button" data-view="videos" …>`); nothing else in either view
depends on it.
