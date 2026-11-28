# Animation cards hold fewer of the wrong characters

**Status:** on branch `claude/issue-65-animation-review`, waiting for you to
merge. Issue [#65](https://github.com/Na-bra/flux-cutter/issues/65).

## The short version

Animation was the weakest part of FluxCutter: between 3.4% and 7.8% of the
faces on a card belonged to somebody else. **Across the eight animated
episodes that is now 3.6% instead of 5.1%** (each episode weighted by how many
faces it contributes) — about three in ten of the wrong faces are gone, and
the worst card in the suite went from 76 faces of other characters to 21.

The reason it was that bad turned out to be a single number in the wrong
units. The rest of this change is a review of the animation code and the
fixes it found.

## What was wrong, and what changed

### The one piece of hard evidence was switched off

Everything in identity matching is a threshold fitted to some footage, with
one exception: **two faces in the same frame are two different people.** That
is a fact about the video, not an estimate, and it is the only thing in the
pipeline that can refuse a merge outright.

It comes with a let-out, because one person *can* be on screen twice — a
split screen, a photograph, a mirror. So a shared frame only counts as
evidence while the two faces look as different as two people actually do.
"As different as two people" was set to 0.50, which is right for filmed
faces. Animation uses a different model, on which two *different* characters
already score about 0.57 and two faces that could be merged score 0.75 or
more. So nothing in an animated episode was ever different enough to count:
the rule was consulted and always said "could be one person showing twice".
It never blocked anything.

Reading it on the animation model's own scale (0.90) is the single largest
accuracy gain available here. It is not a new idea — the October note on
animation card merges tried it on two episodes and found it bought almost
nothing. On the eight episodes there are now, it buys a great deal.

### Four fixes from reading the code

**A face the model could not measure took a whole shot apart.** When a run of
frames is followed as one face, FluxCutter checks whether it quietly turns
into a different character halfway through and cuts it if so. That check
divided by the length of each face's description — and a description of length
zero made every comparison in the shot meaningless. Meaningless comparisons
lose every test, so instead of "no cut here" the answer came back "cut here",
at every single frame: one unmeasurable face shattered the whole shot into
single frames. Faces that cannot be compared now simply have no say.

**A refusal outlived its reason.** When two cards are nearly as good a match
for each other as a third card is for one of them, FluxCutter refuses to join
them rather than toss a coin. But it then never looked at that pair again,
even after the third card had been folded into something else and stopped
being a near-tie. A character could stay on two cards because of a card that
no longer existed.

**The rescue that puts profile shots back counted too generously.** A group
of faces the "is this really a face?" test throws out can be given back to the
card it sits beside, if at least five of its faces do. Each face has a frame
on either side of it, so counting every overlap let three faces cast five
votes. It is five faces now. (This never fires in animation — the animation
detector produces no eye positions, so the test it rescues from does not run
— and it changes nothing on any of the five live-action test videos.)

**Card pictures favoured big faces over sharp ones.** Focus is measured by how
much fine detail a crop has, which depends on how many pixels the crop has:
the same face measured large scores many times higher than measured small.
Live action has always measured a fixed-size crop for exactly this reason;
animation measured the raw one, which ranged from 24 pixels to most of the
frame. So the focus score mostly repeated how big the face was, and every
reasonably large face maxed it out — which defeats the point of having it.
It is measured at a fixed size now, as live action does.

### Tidying

- Finding where a shot turns into someone else now works from one table of
  comparisons instead of rebuilding one for every piece it cuts, and no
  longer calls itself once per cut (a long shot used to risk running out of
  stack).
- The overlap that decides "same face, next frame" was written out twice, in
  two files, as a bare `0.3`. It lives in one place now.
- Embedding a frame's faces no longer builds every crop at model size before
  starting, which is memory it never needed: the model takes one at a time.

### What the review found and deliberately left

- **A card's centroid still drifts the first time you merge or split it.**
  When profile faces are rescued onto a card, the card deliberately keeps the
  description it had, because that description is what matches it to the same
  person in another video. Merging or splitting recomputes it from every face,
  profile shots included — so the first edit moves it. Fixing it means
  recording which faces were rescued inside every kept scan, which would
  throw away every scan you have. It cannot affect animation at all, so it is
  left for a change of its own.
- **Turning the near-tie rule off.** Now that shared frames are real evidence,
  it was worth asking whether the near-tie rule is still needed. It is:
  without it contamination goes from 3.6% to 6.7% (animation9 from 4.5% to
  12.0%), in exchange for 21 fewer extra cards. It stays.
- **Treating a shared frame as settling a near-tie.** If the footage proves a
  third card is not the one in question, arguably there is no coin to toss.
  Measured, that reading is worse: animation 1.0% → 2.3%, animation6 2.7% →
  5.0%, for three fewer extra cards. The existing reading stays.
- **A lower ceiling (0.85).** Keeps less of the gain (4.2% rather than 3.6%)
  for three fewer extra cards. 0.90 was chosen because it is much better on
  animation5, the worst episode in the suite.
- **Faces missed in the checked frames (33% to 50% on some episodes).** Not
  touched. Each episode has only four to eight frames checked by eye, so the
  figure moves in big steps and cannot guide a change yet. The likeliest cause
  found by reading is that a character whose face runs off the edge of the
  frame is dropped for being too small after being cut to the frame; that
  needs its own measurement.
- **The short segment in another art style in animation4** (2:34 to 2:43) still
  lands on the lead's card. Nothing in the review reaches it.

## What it does to your videos

Animated videos only. **Live action is untouched** — every number on `test`,
`test_2`, `test_3`, `test_4` and `test_5` is exactly what it was.

| | Before | After |
|---|---|---|
| animation: contamination, cards, extra cards | 1.62%, 8, 2 | **0.98%**, 9, 3 |
| animation2 | 6.15%, 13, 4 | 6.15%, 13, 4 |
| animation4 | 4.76%, 11, 4 | **2.56%**, 11, **3** |
| animation5 | 7.79%, 17, 8 | **3.67%**, 20, 11 |
| animation6 | 5.41%, 21, 8 | **2.68%**, 24, 10 |
| animation7 | 3.40%, 11, 1 | *4.30%*, 12, 2 |
| animation8 | 3.41%, 14, 5 | *3.45%*, 14, 5 |
| animation9 | 4.09%, 16, 7 | *4.51%*, 16, 7 |
| **all eight, by face** | **5.13%** | **3.62%** |

Detection is unchanged: the same 8,987 faces are found, with the same
precision and the same recall, on all eight.

**The trade.** Five episodes get cleaner cards, two by more than half. Three
get very slightly dirtier:

- **animation8, 3.41% → 3.45%.** Not one wrong face moved: the same 34 are on
  the same two cards. What changed is what they are a share of — 13 of the
  pink-haired girl's faces no longer sit on any card, so there are fewer
  right faces to divide by.
- **animation9, 4.09% → 4.51%.** Three faces of the blond boy joined the
  pink-haired girl's card. 28 wrong faces on cards became 31.
- **animation7, 3.40% → 4.30%.** Four faces of the red-haired masked cyborg
  joined a 29-face card whose majority is the worried tufted man. 14 wrong
  faces became 18. That card used to hold 70 faces; his reel now shows 0.2
  seconds of other named people instead of 5.7, though a smaller share of it
  is him (10% against 19%).

The price paid for all of it is **six more cards across the eight episodes**
(39 extra cards to 45). That is the cheap error in FluxCutter and always has
been: a character on two cards is one merge away from being on one, while a
stranger's face on someone's card ends up in their reel. It is also not the
leads being cut up further — on animation5 the blond boy, the red-marked boy
and the goggles girl have *exactly* the share of their faces on their main
card that they had before. What the extra cards are is the wrong faces moving
off a card they did not belong on and onto one of their own:

    animation5                                     before    after
    the blond boy's crowded card: faces on it         201        74
      of them, the red-marked boy's                    32         2
      the freckled goggles girl's                      29         8
      the old cyborg's                                 12         9
      the green-haired cap girl's                       3         2
    wrong faces on any card in the episode            107        50

One person comes out better off in a way worth naming: animation6's visor
guard keeps 36% of his faces on his main card instead of 80%, but that card is
now his rather than a pool of six characters, so his reel went from 18% him to
97% him, and from 34 seconds of other people to half a second.

## Your kept scans

**Animated videos are scanned again, once.** The ceiling is part of what a
kept scan is filed under, so an animation scan kept before this will not be
found and the video is rescanned the first time you open it. Any names, merges
and splits you made stay attached to the old scan: if you go back to this
version of FluxCutter they are there again.

Live-action scans are untouched and keep everything.

Two things change for animated videos that are not about who is who: the
picture chosen for a card, and which clips a scene pack picks, since both
weigh how sharp a face is and that is now measured properly. Nothing is
stored differently.

## How to undo it

Revert the commits on this branch. Animated videos are then scanned again
once more, back to their old answers; live action never moved.

If you want only part of it: the ceiling is one number,
`cooccurrence_similarity_ceiling` in the Animation entry of `app/modes.py`.
Putting it back to 0.50 restores the old animation numbers and leaves the five
code fixes in place.

## How this was measured

`.venv/bin/python -m app evaluate` over every suite video on this Mac
(`s01e01` is not here), before and after, at the labels in
`evaluation/truth/`. Those labels are drafts, not reviewed by you, so the
numbers are evidence rather than proof. `evaluation/baseline.json` now holds
this run.

One thing to flag honestly: in `test_3`'s saved numbers, two groups of 26
leftover faces that make no card swapped places in the file. Every number is
identical; it is a tie in a sort, and nothing in this change can reach it (the
near-tie rule, the only grouping behaviour that moved, is off in live action).
Two runs today agree with each other, so it looks like the kept baseline was
recorded under slightly different conditions rather than anything here.

`.venv/bin/python -m pytest tests -q`: 1,006 passed, 3 skipped.
