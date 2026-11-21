# The evaluation suite measures the eight new test videos

**Status:** on branch `feature/eight-new-evaluation-videos`, waiting for you to merge.

## What changed

You added eight videos to `assets/test-videos`:

- **Live action:** `test_4` (46 min) and `test_5` (43 min), two episodes of one dark TV drama.
- **Animation:** `animation4` to `animation9` (26 to 30 min each), six more episodes of the same series as `animation2`.

All eight now have labels, and the evaluation suite measures them. It now covers 13 videos on this Mac; `s01e01` still isn't here.

The labels are drafts, made the same way as before. FluxCutter scanned each video, then every card was looked over as sheets of its tracks. The faces that made no card were labelled too, and 12 frames per video were checked for faces the scan missed. Fifteen labelling agents did it in parallel, and their answers were merged with no clashes.

**Who's who:**
- Live-action people are numbered by card (Person 1, Person 2…).
- Cartoon characters keep animation2's names when they appear again.
- Characters new in these episodes got descriptive names. These were checked against each other by face, so a character has the same name in every episode. For example, the woman with a cracked face and blue lips had four names before matching.

## What the suite says

| Video | Faces | Faces found that are faces | Faces found, checked frames | Faces on someone else's card | Cards holding someone else | Extra cards |
|---|---|---|---|---|---|---|
| test_4 | 4,037 | 99.0% | 81.8% | **1.87%** | 3 of 14 | 1 |
| test_5 | 4,960 | 99.6% | 90.0% | **1.46%** | 4 of 12 | 3 |
| animation4 | 1,256 | 97.7% | 75.0% | **4.76%** | 1 of 11 | 4 |
| animation5 | 1,756 | 99.5% | 80.0% | **7.79%** | 4 of 17 | 8 |
| animation6 | 1,670 | 98.9% | 100.0% | **5.41%** | 3 of 21 | 8 |
| animation7 | 773 | 98.4% | 50.0% | **3.40%** | 2 of 11 | 1 |
| animation8 | 1,168 | 98.9% | 66.7% | **3.41%** | 2 of 14 | 5 |
| animation9 | 1,054 | 99.5% | 71.4% | **4.09%** | 4 of 16 | 7 |

**Live action is mostly clean.** In both episodes, the worst card is a split-off card of the man with short spiky hair (Person 2). It collects dark shots, backs of heads and pieces of other people. In test_5 it holds 26 faces of Person 1.

**Animation is where the trouble is, as with animation2:**
- **animation5, the worst at 7.79%:** the blond boy's card also holds 32 faces of the red-marked boy and 29 of the freckled goggles girl.
- **animation6:** a card of greyscale visor guards also holds 31 faces of the smoking man.
- **animation4:** the red-marked boy's card swallows a short retro-cartoon segment drawn in a different style.
- **Splits:** the same character often lands on several cards. animation5 and animation6 each have 8 extra cards.

**Missed-face figures rest on very few faces,** between 4 and 11 per video, so a single face moves them by 10 to 25 points.

## What it means for your videos and kept scans

Nothing in FluxCutter changes. These are measuring files only, and kept scans are untouched.

## How to undo it

Revert the merge. The eight label files and their suite entries go away.
