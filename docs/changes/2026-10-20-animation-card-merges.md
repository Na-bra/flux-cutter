# Why two animation cards still hold two characters

**Status:** investigation, no change to how your videos are scanned. Every
number in the evaluation is exactly what it was. There is a choice for you
to make at the bottom.

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
  `Instructions.md` §53; turning it back on made `animation` slightly worse
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

## What this does to your videos and your kept scans

**Nothing.** No threshold changed. Your kept scans stay valid, keep their
answers, and keep every name and correction you have made on them. Nothing
needs rescanning.

## Try it yourself on one scan

Both of these are per-run flags, so they change nothing permanently:

    python -m app group your-video.mp4 --mode animation --margin-threshold 0.05

A run with a different `--margin-threshold` is kept as a separate scan, so
your existing one is untouched and you can compare them and go back.

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

Revert the one commit on `claude/issue-42-animation-card-merges`, or just
delete the branch: nothing on `main` depends on it.
