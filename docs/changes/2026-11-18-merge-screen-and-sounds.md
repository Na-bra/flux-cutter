# Merging gets its own screen, and choosing a face makes a sound

**Status:** merged into `main`.

## What changed

**Merge and split animations.** Merging or splitting cards now plays a short
animation before the gallery updates.

**Merging has its own screen.** While cards merge, the gallery steps aside, as
it does for the split picker. The cards being merged line up, with the one
being kept on the right. They slide together to a meeting point, a "Merged"
badge appears, and the gallery returns with the single card. It takes about a
second.

**Choosing a face makes a sound.** Selecting a card, or a track in the split
picker, plays a short pencil scratch alongside the ring drawn round it.
Deselecting is silent.

If your Mac is set to reduce motion, the cards don't travel and there is no
sound. The merge screen shows for a quarter of a second instead.

## What it means for your videos

Nothing changes in what a merge does, or in scans, kept scans and exports. This
only changes what you see and hear in the window.

## How it was checked

The merge screen ran in a headless browser with a stand-in for the app behind
it. Merges of 2, 3 and 5 cards, repeated merges and reduced motion all finished
with the gallery back and the merge applied once, with no page errors. The sound
code ran without errors, but **nobody has listened to it in this check**, and
the window wasn't opened with a real scan.

## How to undo it

Revert the merge of this change. Nothing is stored differently.
