# The window looks like a proof sheet

**Status:** on branch `feature/proof-sheet-window`, waiting for you to merge.
These changes were made on 17 November, outside the Claude session that
committed them. This page describes them as found.

## What changed

The window has a new look, modelled on a photographer's proof sheet:

- **Red and black.** A dark background with red for actions and selections.
  It is dark in every case; the window no longer switches between a light and
  a dark theme.
- **Serif italic for words, mono for numbers.** Newsreader and DM Mono replace
  Archivo and IBM Plex Mono.
- **Faces in film frames.** Every face sits in a cream frame with sprocket
  holes along the top and bottom. The person you choose gets a red ring drawn
  round them, loose and slightly off, like a grease-pencil mark, and it draws
  itself in. If your Mac is set to reduce motion, the ring appears without the
  animation.
- **Quieter controls.** Fields are underlined rather than boxed. Switches such
  as Video / Folder and People are words with an underline, not pills.
- **No black bars on faces.** Thumbnails used to be padded into squares with
  black bars, which showed as empty black inside the new frames. The bars are
  trimmed off. Only rows and columns that are entirely black are removed, so a
  dark face is never cut, and a picture that is mostly bars is left as it is.

Nothing about what FluxCutter does has changed: scanning, the people it finds,
scene packs and exports are the same.

## Checked

- All 860 tests pass, including a new one for trimming the bars (a dark face
  keeps its edges).
- Opened in the real window on the test clip: the scan, the film frames, the
  ring on the chosen face and the Export panel (scene pack field included) all
  show correctly, and nothing scrolls sideways.

## How to undo

Revert the commit. Nothing is stored differently.
