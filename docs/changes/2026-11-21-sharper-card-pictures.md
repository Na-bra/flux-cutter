# Card pictures are made at the size the window shows them

**Status:** merged into `main`.

## What changed

Each person's card picture is now made at 384 pixels instead of 192.

On a Retina screen, a card in the window is about 300 pixels wide. A
192-pixel picture was stretched to nearly twice its size to fill it, which
made faces look soft even when the frame behind them was sharp.

## What it means for your videos

- **Close-ups get sharper.** Most live-action leads are filmed close up, so the
  original face is bigger than 192 pixels and there's real detail to keep. On
  test_4, 37 of the 132 cards in your scan gain detail.
- **Small faces look the same.** A face that was small in the frame has no
  extra detail to show, so it's as soft as before.
- **Some cards stay blurry.** A card whose chosen picture is dark or blurry in
  the video itself keeps it. That needs a better way of choosing the picture,
  which isn't part of this change.
- **The window loads slightly more.** For test_4's 132 cards, the pictures go
  from 0.5 MB to 1.2 MB in total.

## Kept scans

A kept scan stores each card's face at full size, so scans you kept before
this get the sharper picture the next time you open them. Nothing needs
scanning again, and names, merges and splits are untouched.

## How it was checked

- 896 tests passed and 2 were skipped. A new test checks the picture size for
  a fresh scan, a kept scan and a card after an edit.
- Pictures from your own test_4 scan were compared side by side at the size a
  card shows them, old and new. The close-ups were visibly sharper.

## How to undo it

Revert the merge of this change. Nothing is stored differently.
