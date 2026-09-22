# FluxCutter

FluxCutter finds the people in a video and cuts everything one of them is in
into a single reel. Point it at an episode, or a whole season, click a face,
and export.

It runs on macOS and Windows as a desktop app, or from source with Python.

## Download

Get the latest build from [Releases](https://github.com/Na-bra/flux-cutter/releases).
The builds are not signed, so both systems warn the first time:

- **macOS** calls the app damaged. It is not. Run
  `xattr -dr com.apple.quarantine /path/to/FluxCutter.app`, or open
  **System Settings → Privacy & Security → Open Anyway**.
- **Windows** shows "Windows protected your PC". Click **More info → Run anyway**.

The face models (about 174 MB) download on first use, with a progress bar,
and are checked against a pinned checksum.

## Using it

**One video.** Choose a video, pick the content type (live action or
animation), and press **Scan for people**. You get a card for everyone who is
on screen for more than a few seconds. Click a card to see how long their
reel is and frames from it, then **Export reel**. Click more cards to put
several people in one reel.

**A whole season.** Switch the source to **Folder** and scan. You get one card
per person across every episode, with their face from each. When two cards
look alike but not clearly enough to join on their own, FluxCutter asks
**Same person?** and shows both faces. Export gives one reel across the
folder, with recaps and repeated openings shown once.

**Fixing the cards.** Grouping gets most of a video right, not all of it.
Select cards and use **Merge** (the same person on two cards), **Split…**
(two people on one card), or **Not a person** (a logo or the back of a
head). Corrections are kept, so they are still there next time.

**Names.** Select a card and press **Name…**. Names are remembered across
every video you open: after a scan, a card that looks like someone you named
before shows their name with a question mark, and you answer **That's them**
or **Not them**.

**Report.** In folder view, **Report** writes who is in which episode and for
how long, as a page and a spreadsheet, beside your reels.

**Advanced.** The button beside Sampling holds the three settings that
change most what a scan finds — how alike two faces must be, the least
screen time worth a card, and the smallest face to look at — each described
by what moving it does.

A scan of a 22-minute episode takes about a minute. It is kept, so opening
the same video again is instant.

## From source

Python 3.12 on macOS, Windows or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m app ui                 # the window
```

The window draws through the system's web view (WebKit on macOS, WebView2 on
Windows), so there is nothing else to install. The models download the first
time they are needed; `python -m app models fetch` gets them ahead of time.

## Command line

Everything the window does is also a command. The main ones:

```bash
# Who is in a video: a montage of person cards, numbered from 0
python -m app group episode.mp4 --interval 0.5

# Cut one person's reel, by card number, by name, or from a photo
python -m app export episode.mp4 --select-index 0 --output reel.mp4
python -m app export episode.mp4 --select-name "Jamie Lee" --output reel.mp4
python -m app export episode.mp4 --reference photo.jpg --output reel.mp4

# One person across a folder, as one reel or one per episode
python -m app batch season-1/ --person "Jamie Lee" --combine jamie.mp4
python -m app batch season-1/ --person "Jamie Lee" --output-dir reels/

# Who is in which episode, and for how long
python -m app report season-1/

# When one person is on screen
python -m app timestamps episode.mp4 --select-index 0
```

And for housekeeping: `people` lists everyone you have named (`people forget
NAME` removes one), `scans` shows what is kept (`scans clear` deletes it),
and `models` shows where the face models are. Add `--rescan` to ignore a
kept scan, and `--help` to any command for its options.

Some options worth knowing:

- `--mode animation` for drawn footage.
- `--encoder h264_videotoolbox` is several times faster on a Mac with Apple
  silicon; the window uses it there by default.
- `--select-index 0 2` puts two people in one reel: every scene either is in.
- `--keep-repeats` keeps recaps in a combined reel.

Videos can be `.mp4` or `.mov`.

## Building the app

```bash
./packaging/build.sh             # dist/FluxCutter.app and a zip to send
```

On Windows the same script builds `dist/FluxCutter/`. A Windows build has
to be made on Windows, so pushing a `v*` tag builds both on GitHub Actions
and attaches them to a draft release, with notes taken from the tag's
message. Tag releases with `git tag -a --cleanup=whitespace -F notes.md
v1.2.3` so markdown headings survive.

## Tests

```bash
pytest -q
```

The suite needs no display and mostly no footage: the sound-and-picture sync
tests generate their own. Tests that need real video look for it at
`assets/test-videos/test.mp4` (and the long episode at `test_3.mp4`), which
are not in the repository, and skip without them. Tests run on every pull
request.

## Troubleshooting

- **The window will not open on Windows.** The WebView2 runtime is missing.
  It comes with Windows 11 and with Edge on Windows 10; otherwise install
  Microsoft's Evergreen runtime. Every command except `ui` works without it.
- **Import errors or a broken environment.** Delete `.venv` and set it up
  again as above.
- **A warning about duplicate `libavdevice` symbols on macOS.** An older
  environment has `opencv-python` instead of `opencv-python-headless`.
  Reinstall from `requirements.txt`.

## More

[Instructions.md](Instructions.md) is the project's working log: how each
stage was built, the measurements behind every default, and what was tried
and dropped.
