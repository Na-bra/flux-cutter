# The evaluation suite measures the new test videos

**Status:** on branch `feature/new-evaluation-labels`, waiting for you to merge.

## What changed

You replaced two test videos with new ones under the same names:

- `test_3.mp4` is now a 44-minute live-action episode (a Sam & Cat and
  iCarly crossover).
- `animation2.mp4` is now a 28-minute animated episode.

The evaluation suite's labels described the old videos, so it skipped both
("not the file its truth was labelled on"). Both now have new labels, and the
suite measures them again.

The labels are drafts, made the same way as before. FluxCutter scanned each
video, then every card was looked over as sheets of its tracks (one row per
run of the same face), saying who each card is and which tracks are someone
else. Faces that made no card were labelled too, and 12 frames per video were
checked for faces the scan missed. As before, live-action people are numbered
by card (Person 1, Person 2…). Cartoon characters are named by how they look,
because the show wasn't recognised: "Red marked boy", "Old cyborg".

## What the suite now says

| | test_3 (44 min, live action) | animation2 (28 min, animation) |
|---|---|---|
| Faces labelled | 9,813 | 966 |
| People | 142 (most are audience members with no card) | 18 |
| Found faces that really are faces | 98.9% | 98.0% |
| Faces found, in the checked frames | 90.9% | 33.3% (4 missed of 6) |
| Faces on someone else's card | **1.03%** | **6.15%** |
| Cards holding someone else | 5 of 40 | 4 of 13 |
| Extra cards (one person split) | 7 | 4 |

**Live action is mostly clean.** The worst card is a split-off card of the
curly-haired boy in a hat (Person 5). It also holds 32 faces of Person 3 and 12
of Person 6, mostly seen in profile.

**Animation is where the trouble is, as before.** Most of its 6.15% is one
card: the old cyborg's card holds 27 faces of a younger man in goggles from a
greyscale flashback. The labeller thought he might be the old cyborg when
young. If he is, those 27 faces are not a mistake, and contamination drops to
about 3%. Only you can settle that.

These numbers can't be compared with the old ones for `test_3` and
`animation2`, because they're different videos. `evaluation/baseline.json`
now holds the new numbers for both.

## What's uncertain

These are drafts, as your earlier decision allows. The labellers noted the
calls they were least sure of in each label file's notes. The main ones:

- Whether the man in goggles is the old cyborg when young (above).
- On test_3, a few cards that might be one person (for example cards 21 and
  25, two boys in the same scene), and audience members matched across
  different reaction shots.
- Faces seen only from behind are labelled "unknown". A labeller had named some
  of them from where they stood in the scene; that was changed back to
  "unknown", since nobody could tell those faces apart.

## How to undo

Revert the commit. The old labels can't be used with these videos anyway.
