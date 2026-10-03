"""How well FluxCutter finds and keeps apart the people in a video.

The thresholds were fitted to one 22-minute episode, and a reel that holds
another character is the failure that matters most here. This measures it
against ground truth -- every face the pipeline produced on a video, labelled
with who it is -- rather than by looking at galleries:

- **detection**: of the faces found, how many are faces (precision); of the
  faces on screen in checked frames, how many were found (recall);
- **identity**: how often one person is split across cards, and how often a
  card holds someone else (contamination -- the one to watch);
- **timing**: how closely a person's card's appearances match theirs;
- **export**: how much of the reel cut from their card shows them, and how
  much shows somebody else instead.

`truth` is the format, `metrics` the measuring, `run` the pipeline pass and
the suite. See Instructions.md for the suite and its results.
"""
