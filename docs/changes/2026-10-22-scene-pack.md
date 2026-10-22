# Scene packs: someone's clearest scenes, as clips, at a length you choose

**Status:** on branch `feature/scene-pack`, waiting for you to merge.

## What it does

A reel is everything someone is in. A **scene pack** is a set length of
their *best* scenes: "a minute of Ben". You get a folder with one video file
per scene (what editors work from) and one reel of all of them.

In the window, after choosing someone:

1. Type a length in **Scene pack length** in the Export panel: `90`, `1:30`
   or `2m`. Press Enter.
2. FluxCutter picks their clearest scenes until the pack is that long. The
   panel says how many scenes it picked. The button now reads **Export scene
   pack**.
3. Optional: open **Edit cuts…** to see the picked scenes. The rest are
   listed as dropped, so you can drop a scene or bring one back by hand, as
   with any reel.
4. **Export scene pack** writes a folder, named after the file name box, with:
   - `ben.mp4`, the reel of every scene in the pack, in story order;
   - `ben 001 (0m14s).mp4`, `ben 002 (4m07s).mp4`, …, one clip per scene,
     numbered in story order and named after where the scene starts. In a
     folder of episodes the episode is in the name too:
     `ben 003 (S01E02 4m07s).mp4`.

Empty the length box to go back to exporting a whole reel. The length stays
set when you choose someone else, so you get their pack at the same length.

## How "clearest" is decided

Each scene the reel would have had is scored by:

- **how good their faces in it are**, using the same measure that picks the
  picture on their card: sharp, big enough, confidently detected, and looking
  like them; and
- **how much of the scene they're actually in**, so a scene that's mostly the
  other side of a conversation ranks below one that's mostly them.

Scenes are taken best first, whole, while they fit. The first one that
doesn't fit is trimmed to what's left, around its clearest moment, so the pack
comes out the length you asked for. If what's left is under 2 seconds, it
stops there instead of adding a flash. A scene they're never actually seen in
is never picked. If there isn't enough of them, you get everything, and the
panel tells you how much there was.

## Measured

| | 22-minute live-action test video | 7-minute animation test video |
|---|---|---|
| Asked for | 1:00 | 0:30 |
| Scenes picked | 14 of 91 | 4 of 13 |
| Written: reel, and the clips added up | **60.02 s** | **30.40 s** |
| Time to write the pack | 31 s | 23 s |

The reel and the clips always add up to the same length, because they're the
same scenes.

**Why the planned scenes are a little longer than the pack.** Cutting takes a
fraction of a second off the ends of every scene. It drops any flash of the
neighbouring shot and ends on a whole frame, exactly as it does for a normal
reel. On the first try the 60-second pack came out 52.9 s: every scene lost
0.17–0.87 s, 0.55 s on average. The picker now allows for that, which is why
the panel says "about 1:00". The 0:30 pack on the animation video, which the
allowance wasn't measured on, came out 30.4 s.

**Is it really their best moments?** In the middle frame of each of the 14 clips
in the one-minute pack, the lead is on screen, mostly close up, in 13. The
fourteenth catches the other side of his conversation at that moment. A
normal reel does that too, because scenes are padded and short gaps bridged.

## What it doesn't change

- Exporting a whole reel is exactly as before when the length box is empty.
- Scans, kept scans, names and corrections aren't touched.
- If a pack's folder already has files in it, the next pack goes into
  `ben (2)` rather than mixing with the first one.
- Cancelling a pack removes what it had written, as cancelling a reel does.

## How to undo

Revert the commit. Nothing is stored differently.
