# Why two animation cards still hold two characters

**Status:** on branch `fix/mode-owned-margin-and-ceiling`. You chose 0.02
for animation's margin (see "Every setting between 0 and 0.05" below), and it's
now the default. The nightly run did the investigation; it's been moved here
without the measuring script it committed but never ran.

This is issue #42. On `animation2`, two cards each hold two characters:

- **card 4** — a split-off card of Gwen's that also carries 11 faces of the
  black-haired boy with glasses, 4 of Ben and 4 of Grandpa Max;
- **card 5** — the dark-haired teen, carrying all 11 faces of the boy in a
  blue shirt.

The issue asked for the merge to be found and stopped. It was found. It
cannot be stopped with any of the settings the pipeline has, and the rest of
this page is the evidence, because the fix that *would* work charges more
than the issue allowed.

## What is actually happening

Both cards are joined during grouping, when FluxCutter decides two runs of
faces are the same character. It merges them when they are at least 75%
alike on average. These pairs are **80% or more alike** — two genuinely
different characters, drawn in the same style, that the animation face model
cannot tell apart.

Three things were expected to catch it. None does:

- **"Two faces in one frame are two people"** — FluxCutter's only piece of
  hard evidence. It does nothing here. The tracks involved never appear in
  the same frame as each other, so there is nothing to check. (It is also
  switched off in animation generally, for a separate reason written up in
  `Instructions.md` §55; turning it back on made `animation` slightly worse
  and cut the bull creature's card in two.)
- **Folding duplicate cards together more readily**, so Gwen's split-off
  card rejoins her main one and the boy's faces have nowhere wrong to go.
  Tried, and it is much worse: two different characters' whole cards are
  themselves 80% alike on this footage, so loosening it fused Grandpa Max
  with the teen *and* the boy in blue onto one 418-face card. Contamination
  went from 1.83% to 4.45%.
- **Demanding more resemblance before merging** (75% → 80%). Both cards come
  out exactly as before. The merges are already above 80%.

## The one thing that works, and its price

There is a rule that refuses a merge when a *second*, different character is
almost as good a match — a coin flip rather than a decision. It is off by
default. Turning it on does fix card 5 and does bring contamination down
past the bar the issue set:

| | contamination now | with the rule at 0.05 |
|---|---|---|
| animation | 2.48% | **0.67%** |
| animation2 | 1.83% | **0.97%** |

But it pays for that by breaking your main characters across several cards:

| | now | with the rule at 0.05 |
|---|---|---|
| Ben on `animation` | 98% of his faces, 1 card | **56%, spread over 4 cards** |
| Ben on `animation2` | 98%, 1 card | 72%, 2 cards |
| Grandpa Max on `animation2` | 98%, 1 card | 86%, 3 cards |
| Gwen on `animation2` | 90%, 3 cards | 79%, 4 cards |
| cards in the gallery | 7 and 14 | 10 and 18 |

And card 4 is *still* not clean even then — Gwen's split-off card keeps the
boy's 11 faces.

So the trade is: cleaner cards, but your gallery shows Ben four times and
Grandpa Max three, and each of their reels is shorter by a third. That is a
decision about your videos, not a bug fix, so it was left to you.

A gentler setting (0.02) is a middle point — `animation` 1.62%,
`animation2` 1.39%, which still passes the issue's bar — and it costs Ben
and Max about 11% of their faces each, and one extra card apiece.

## Every setting between 0 and 0.05, measured

The pages above tried 0.02 and 0.05. Every margin in between, on both videos,
with what each costs your main characters:

| Margin | animation | animation2 | Meets #42's bar? | Ben on animation | Ben / Max on animation2 |
|---|---|---|---|---|---|
| 0 (now) | 2.48% | 1.83% | — | 1 card | 1 card / 1 card |
| 0.01 | 2.49% | 1.37% | no, by one face | 1 card | 1 card / 2 cards |
| 0.015 | 2.52% | 1.39% | no | 2 cards | 2 / 3 |
| **0.02** | **1.62%** | **1.39%** | **yes** | **2 cards** | **2 / 3** |
| 0.03 | 0.67% | 1.11% | yes | 3 cards | 3 / 3 |
| 0.05 | 0.67% | 0.97% | yes | 4 cards | 2 / 3 |

**0.02 is the gentlest setting that meets the bar.** At 0.02 each main
character's reel loses about a tenth of its length to a second card of the same
person: Ben on animation goes from 57 s of him to 50 s, and from 263 s to 236 s
on animation2; Grandpa Max goes from 178 s to 158 s. Each reel is also slightly
purer. Merging the two cards puts the missing part back.

At 0.03 Ben keeps only 65% of his faces on his main card on animation, which is
the 0.05 problem starting early.

**You chose 0.02, and it's now animation's default.**

## What this does to your videos and your kept scans

- **Animation videos are scanned again** the first time you open each one,
  because its kept scan was made with the old setting. Corrections and names
  you made on the old scan stay with that old scan.
  Names you've saved as people are matched against the new cards as usual.
- Expect Ben, Grandpa Max and other main characters to sometimes show up on
  **two cards**. Merging them puts their whole reel back together.
- **Live-action videos are untouched**: same setting, same kept scans.

## Going back for one scan

To scan one animation video the old way, without the margin:

    python -m app group your-video.mp4 --mode animation --margin-threshold 0

A run with a different `--margin-threshold` is kept as a separate scan, so
you can compare the two and switch between them.

## The numbers, before and after

The whole suite, before this work and after it:

| video | precision | recall | contamination | cards holding someone else | extra cards |
|---|---|---|---|---|---|
| test | 100.0% | 71.4% | 0.00% | 0 of 2 | 0 |
| test_2 | 100.0% | 90.9% | 0.00% | 0 of 2 | 0 |
| animation | 97.7% | 71.4% | 2.48% | 3 of 7 | 1 |
| animation2 | 96.0% | 50.0% | 1.83% | 4 of 14 | 2 |
| test_3 | 99.7% | 74.2% | 0.22% | 5 of 42 | 2 |

Identical in both columns, which is why there is only one table.
`s01e01` is not on this Mac and was skipped. The live-action videos were
checked because grouping is shared code: `test`, `test_2` and `test_3` are
all unchanged, down to each person's individual numbers.

## What did change in the code

One real but invisible fix. `--margin-threshold` and
`--cooccurrence-ceiling` were the last two settings that still defaulted to
a live-action number baked into the program, instead of belonging to the
mode you chose. An animation scan silently used live action's values for
both. They now come from the mode, like every other threshold, which is
what makes `--margin-threshold 0.05` above a real choice rather than a
number that gets overwritten.

Both modes keep the values they were already effectively using, so this
changes no result anywhere — that is what the identical table above shows.

## How to undo it

Revert the commit that made 0.02 the default, or set `margin_threshold` back
to `0.0` for animation in `app/modes.py`. Your old animation scans are kept
unless the scan cache has since dropped them for space (it removes the oldest
first), so going back usually needs no rescanning.
