# FluxCutter

FluxCutter is a Python prototype for identifying a person in a video and compiling all of that person's appearances into a single exportable clip. The goal is to turn a raw video into a focused "character reel" by detecting faces, grouping them by identity, and selecting the moments where the target person appears.

This project is currently in an early prototype stage focused on validating the workflow rather than polishing a production-ready application.

## What the project does

The intended workflow is:

1. Load a video file
2. Extract frames for inspection
3. Detect human faces in each frame
4. Link detections across consecutive frames into face tracks
5. Group tracks that belong to the same person
6. Show the detected people in a face gallery
7. Let the user choose a target person
8. Find every matching appearance in the timeline
9. Merge nearby clips and export the final result

The codebase is organized around this stage-by-stage workflow, with separate modules for video handling, face detection, grouping, and UI interactions.

## Current status

This repository is a prototype and the core functionality is still being validated. The focus is on demonstrating that the video-to-face-detection-to-selection flow works on real footage, not on shipping a complete consumer-facing app.

The whole pipeline now runs end to end — a 22-minute episode goes in, a single reel of one person's appearances comes out — through either the CLI or [the desktop window](#the-desktop-window). What is still unsettled is accuracy on hard footage rather than whether the stages connect.

The project structure is intentionally simple and lightweight so experimentation remains easy.

## Project structure

```text
flux-cutter/
├── app/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── main.py              # pipeline stages, shared by every front end
│   ├── faces/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── embedder.py
│   │   ├── anime.py       # animation-mode detection + embedding
│   │   ├── reference.py   # picking a person with a photo
│   │   ├── edits.py       # merging, splitting, discarding identities
│   │   ├── quality.py
│   │   ├── tracker.py
│   │   └── grouper.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── __main__.py      # python -m app.ui
│   │   ├── web.py           # the window: a thin bridge over worker.py
│   │   ├── window.html      # what the web view draws
│   │   ├── worker.py        # the window's background work, toolkit-free
│   │   └── gallery.py       # thumbnail/montage rendering
│   └── video/
│       ├── __init__.py
│       ├── frames.py
│       ├── loader.py
│       ├── source.py        # keeps a video readable after its file moves
│       ├── timeline.py      # detections -> appearance intervals
│       ├── cutter.py        # segments -> one reel, in-process via PyAV
│       └── export.py        # intervals -> cut segments -> one reel
├── modes.py             # live action / animation, chosen by the user
├── scans.py             # keeping a scan, so a video is not scanned twice
├── assets/
│   ├── models/
│   │   ├── face_detection_yunet_2026may.onnx
│   │   └── face_recognition_arcface_w600k_r50.onnx
│   └── test-videos/
│       └── test.mp4
├── tests/
├── Instructions.md
├── README.md
├── .gitignore
└── .venv/
```

## Setup

This project is currently validated on Python 3.12 and uses OpenCV YuNet running on CPU for face detection instead of MediaPipe.

### 1. Create a virtual environment

```bash
cd flux-cutter
python3.12 -m venv .venv
source .venv/bin/activate
```

If you are using a different Python version, make sure it is compatible with the installed OpenCV package. Python 3.12 is the version used in this repo during validation.

### 2. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins `opencv-python-headless` rather than `opencv-python`: the headless build skips OpenCV's camera/GUI backend, which is what caused the duplicate-`libavdevice`-symbol warning below when paired with PyAV. The project has no GUI/capture-device usage through OpenCV, so this is a safe swap and not a downgrade in capability. The desktop window draws through a web view and never through OpenCV, so the two do not conflict.

**The window needs nothing installed alongside Python.** It draws through the system's own web view — WebKit on macOS, WebView2 on Windows — so `pip install -r requirements.txt` is the whole setup. This used to be Tk, whose `_tkinter` C extension ships separately from Python and is absent from Homebrew's `python@3.12`, which meant `brew install python-tk@3.12` before the window would open at all.

Every CLI command works without a web view too — `app/__main__.py` imports the UI lazily for exactly that reason — so a headless machine can still run the whole pipeline.

### 3. The models fetch themselves

Nothing to do here. The detector uses the official OpenCV Zoo YuNet model (230 KB) and identity grouping uses ArcFace `w600k_r50` (174 MB, a ResNet50 trained on WebFace600K). **Both download automatically the first time something needs them** — the moment you run a command that detects or embeds a face, not at install and not at import.

They land in a per-user cache (`~/Library/Application Support/FluxCutter/models` on macOS, `%LOCALAPPDATA%` on Windows, `$XDG_DATA_HOME` on Linux), so a frozen app on a read-only volume still works. Set `FLUXCUTTER_MODEL_DIR` to put them elsewhere.

```bash
python -m app models          # where they are, and whether they are present
python -m app models fetch    # download now rather than mid-scan
python -m app models clear    # delete the downloaded copies
```

Each file's SHA-256 is pinned, verified after download, and the file is only moved into place once it matches — so a truncated download, a proxy that mangles it, or a mirror that changes what it serves is rejected rather than loaded. An interrupted download leaves nothing behind and can simply be retried. This is not theoretical: an earlier model in this project arrived corrupted through a text-mode round trip at 70 MB instead of 38 MB and surfaced as five confusing test failures rather than as a download error.

<details>
<summary>Fetching them by hand instead</summary>

A copy in `assets/models/` always wins over the cache, so a checkout with the models already in it keeps working untouched and never re-downloads:

```bash
mkdir -p assets/models
curl -L --fail -o assets/models/face_detection_yunet_2026may.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx

curl -L --fail -o assets/models/face_recognition_arcface_w600k_r50.onnx \
	https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx
```

ArcFace is fetched standalone rather than from InsightFace's `buffalo_l` bundle, which is 275MB to extract the same 166MB file — the rest of it is detection/landmark/attribute models this project doesn't use. Verify with `shasum -a 256`; the expected hashes are pinned in [app/models.py](app/models.py).

</details>

Both models load through OpenCV's DNN module (`cv2.FaceDetectorYN` for detection, `cv2.dnn.readNetFromONNX` for ArcFace), so no extra Python package is needed beyond what's already installed -- no torch, no onnxruntime.

### 4. Validate the loader

Run the project’s real sample video through the loader to confirm the environment is healthy:

```bash
python -c "from app.video.loader import load_video, get_video_info; c = load_video('assets/test-videos/test.mp4'); print(get_video_info(c)); c.close()"
```

You should see metadata similar to:

```python
{'width': 2160, 'height': 2160, 'codec': 'h264', 'frames': 701, 'duration': 23.366666666666667, 'fps': 30.0}
```

### 5. Generate a face gallery

The gallery command samples real frames, detects faces once per frame, crops each detected face with a small padding, and saves a simple thumbnail grid:

```bash
python -m app gallery assets/test-videos/test.mp4 --interval 1.0 --output-dir output/face-gallery
```

You can also select an item while generating the gallery:

```bash
python -m app gallery assets/test-videos/test.mp4 --interval 1.0 --select-index 0
```

### 6. Group detections into identities

The group command runs the same sampling and detection pass, embeds each face with ArcFace, links detections into face tracks, and groups those tracks into per-identity clusters using agglomerative average-linkage clustering with a similarity floor, then runs a consolidation pass that folds together groups whose centroids agree (which is what catches one actor split across two clusters, e.g. in and out of costume). It saves one representative thumbnail per identified person:

```bash
python -m app group assets/test-videos/test.mp4 --interval 0.25 --output-dir output/face-groups
```

Grouping quality depends heavily on the sampling interval, because tracking is what makes it work. Denser sampling gives the tracker consecutive frames it can actually link, and a track's averaged embedding is far more reliable than any single frame's. On the test video:

| `--interval` | detections | tracks | identity groups |
| ------------ | ---------- | ------ | --------------- |
| 1.0s         | 23         | 17     | 3               |
| 0.5s         | 49         | 33     | 4               |
| 0.25s        | 88         | 38     | 2               |

At 1.0s, shots cut and faces jump between samples, so tracks mostly degenerate to length 1 and the stage does nothing. Prefer 0.5s or denser for real grouping work.

The group counts stay in the same small range across sampling rates because the minimum-screen-time cutoff scales with the interval; the detection counts underneath it do not. Compare montages rather than totals when judging a change — the total moves for reasons unrelated to accuracy.

Three filters run after clustering, and each exists because of a failure seen on the 22-minute test footage rather than in principle. A **consolidation pass** folds together groups whose centroids agree, which is what reunites one actor split across two clusters (on that footage, the same character in and out of a costume mask). A **non-face filter** drops whole groups whose landmark geometry says they are not people — YuNet reports confident "faces" for backs of heads and for the show's logo, and because those detections fail in the same way they cluster into a convincing phantom identity. A **minimum-screen-time cutoff** sets aside identities too brief to be worth selecting. Together these took that video from 412 person cards to 165, then to roughly 40 once the screen-time cutoff applies.

Every filter routes its rejects to the unassigned count rather than deleting them, so the report always accounts for every detection.

### Memory

`extract_frames` streams: it yields one decoded frame at a time rather than returning a list, so memory is flat in the length of the video instead of scaling with it. A 22.6-minute 720p episode at a 1.0s interval peaks around 1.0 GB; the previous list-returning version peaked at 4.9 GB on the same run, and would have needed roughly 15 GB at a 0.25s interval.

Two things follow for anyone calling it directly:

- The result is a one-shot iterator with no length. Count as you go rather than calling `len()`, and wrap it in `list(...)` only when you genuinely need random access to a short video.
- Decoding happens during iteration, so iterate **inside** the `with load_video(...)` block. Consuming it afterwards reads from a closed container.

Identities with very little screen time are set aside rather than shown as person cards. The cutoff is derived from the video's runtime and the sampling interval — roughly 0.5% of runtime, and never less than 3 seconds of screen time — so it means the same thing whether you sample at 1.0s or 0.25s. On the 23s clip that is 3 detections at a 1.0s interval; on a 22-minute episode it is 7. Override it with `--min-detections N`, or pass `--min-detections 1` to keep everything.

Grouping behavior is tunable:

```bash
python -m app group assets/test-videos/test.mp4 \
	--similarity-threshold 0.35 \
	--margin-threshold 0.0 \
	--consolidation-threshold 0.5 \
	--min-group-eye-span 0.15 \
	--min-confidence 0.7 \
	--min-face-size 40 \
	--min-detections 5
```

See [Instructions.md](Instructions.md#7-stage-02-accuracy-notes-identity-grouping) for how these defaults were chosen against the real test footage.

### 8. Export a person's reel

Once you know which person card you want, `export` cuts every one of their appearances out of the source and joins them into a single file:

```bash
python -m app export assets/test-videos/test.mp4 --select-index 0 --output output/reel.mp4
```

`--select-index` uses the same numbering as the `group` montage. On Apple silicon, add `--encoder h264_videotoolbox --quality 55` — it runs roughly 3.6x faster than the portable `libx264` default and writes a smaller file.

**Cutting happens in-process, through PyAV — `ffmpeg` is no longer needed.** It used to shell out to the binary, which broke the moment the app was packaged: a Finder-launched `.app` inherits `/usr/bin:/bin:/usr/sbin:/sbin`, so a Homebrew ffmpeg is invisible to it. PyAV was already a dependency, already ships FFmpeg in its wheel, and already carries every encoder used here, so this removed a dependency rather than adding one. Verified frame-for-frame against the old subprocess path: 360 video frames from the same three segments, either way.

**Segments are re-encoded, not stream-copied.** This is deliberate and not an oversight. A stream copy can only begin at a keyframe, and keyframe spacing on real footage is coarse: on the 22-minute test video keyframes sit a median 2.67s apart (up to 7.84s) while the median appearance is 3.0s long, so a copied cut would routinely open several seconds early — on somebody else's face. The 23-second test clip is nearly all-intra (keyframes 0.03s apart) and hides this completely, so anything validated only there will look perfect and fail on real video. Joining the finished segments *is* a stream copy, safely, because they were all just written with identical codec parameters.

The cut list is not the appearance list. `timestamps` answers "when was this person on screen", which is a detection question; a watchable reel is an editorial one, and they disagree. On the 22-minute footage the lead's appearances come back as 173 intervals with a median duration of 3.0s. Cut literally that is a strobe, so `merge_for_export` widens each interval, bridges gaps too short to cut across, and grows anything still under a minimum length:

| `--bridge-gap` / `--min-segment` | segments | total | median |
| --- | --- | --- | --- |
| raw appearance intervals | 153 | 517.9s | 3.00s |
| 1.5 / 2.0 (default) | 97 | 652.9s | 4.51s |
| 2.0 / 3.0 | 77 | 704.3s | 7.51s |
| 3.0 / 3.0 | 52 | 757.8s | 10.51s |

More bridging means fewer, longer cuts but more footage where the person is briefly absent. `--export-padding` adds headroom on each side, on top of whatever padding the appearance intervals already carry.

A real run on the 22-minute episode: 1057 detections for the lead, 173 appearance intervals (527.0s on screen), 107 segments cut (662.2s), encoded in 180s at 3.67x realtime with `h264_videotoolbox`, peak memory 1.24 GB. The wall-clock cost is dominated by the detect/embed pass (~7 minutes), not by the cutting (~3 minutes).

Export needs no external binary. `ffmpeg` on PATH is still useful for inspecting output with `ffprobe`, and the test suite skips its verification steps without it, but nothing in the pipeline requires it.

### 7. Compute appearance timestamps for one person

Once you know which person card you want (from the `group` command's output or montage), the `timestamps` command runs the same grouping pass and converts that one person's detections into appearance intervals — contiguous spans of time they're on screen, gap-split, padded, and clamped to the video's duration:

```bash
python -m app timestamps assets/test-videos/test.mp4 --interval 0.5 --select-index 0
```

`--select-index` refers to the same ordering shown by `group` (largest identity group first). Gap tolerance and padding default to sampling-derived values (`2x` and `0.5x` the `--interval`, respectively) but can be overridden:

```bash
python -m app timestamps assets/test-videos/test.mp4 --interval 0.5 --select-index 0 \
	--gap-tolerance 1.5 \
	--appearance-padding 0.3
```

See [Instructions.md](Instructions.md#8-appearance-timestamp-notes-appvideotimelinepy) for why those defaults are tied to the sampling interval rather than fixed constants.

### 9. Pick the person with a photo instead of an index

`--select-index` means running `group` first, looking at a montage, and
counting cards. When you already know who you want and have a picture of
them, name them directly:

```bash
python -m app export assets/test-videos/test_3.mp4 \
	--reference photos/lead.jpg --output output/reel.mp4
```

The photo is detected and embedded once, before the scan starts, and the
scan's identities are then scored against it:

```text
lead.jpg matched Person #1 at 0.81 (next best 0.14).
```

It works on `timestamps` too, and `--select-index` and `--reference` are
two ways of saying the same thing — pass one, not both.

The match is scored against each identity's centroid (the mean over every
observation in the group), not its best single frame, for the same reason
track averaging exists: one frame's embedding on hard footage is noise.

Two refusals are deliberate, because a confident wrong answer here costs
minutes of encoding somebody else's face:

- **Nobody matches.** Anything under the mode's own grouping threshold
  (0.35 for live action) is reported as absent rather than rounded up to
  the nearest face. Lower it with `--reference-threshold` for a hard photo.
- **Two people match about equally.** Within 0.05 of each other, the run
  stops and asks you to pick, because that means the scan probably split
  one person across two cards.

A photo embedded in one mode cannot select an identity grouped in the
other — an ArcFace vector and a CCIP vector have a cosine similarity, and
it means nothing. Mixing them is refused, not silently scored.

On the 22-minute episode, with a reference photo written from each of the
38 identities and read back through the full decode-detect-embed path, 37
of 38 selected their own identity; the right one never scored below 0.543
and no wrong one reached 0.30. See [Instructions.md](Instructions.md#19-choosing-a-person-with-a-photograph-appfacesreferencepy)
for what that does and does not prove.

### 10. Run a whole folder

Name the person once, in the window, on any episode where they appear. Then
cut them out of the whole season into one reel:

```bash
python -m app batch ~/footage/season-1 \
	--person "Jamie Lee" --combine output/jamie-season-1.mp4
```

No picture of them is needed from anywhere but the footage. The name is
looked up in the kept scans of those videos, and every card carrying it is
averaged into one face. Episodes where you named them use those cards;
every other episode is searched for that face, with the same floor and
margin as a photo, so an episode where the match is unclear is skipped and
said so rather than guessed at. Naming them in two episodes gives a steadier
face than naming them in one.

Measured on the 22-minute episode split into two files, with four
characters named in the first: all four matched the right card in the
second, at 0.92-0.97, with no other card above 0.30. A reel joining both
files kept picture and sound within 1ms over two minutes.

Without `--combine`, each video gets its own reel, named after it
(`S01E03-reel.mp4`) in `--output-dir`. Nobody named yet? `--reference
photos/lead.jpg` matches a photograph instead. Folders, files, or a mix;
`--recursive` searches at any depth.

**One reel from many videos.** Every video is scanned and planned before
anything is encoded, so a video that cannot contribute is found out before
minutes of encoding depend on it. Videos of a different shape are fitted
with black bars rather than stretched, and a different sample rate or a
missing sound track is handled. A video at a different frame rate from the
rest is left out by name — joining 25fps footage into a 24fps reel would
play it at the wrong speed — and the reel is cut from the others.

**Nothing aborts the run.** An episode that is unreadable, holds nobody who
matches, or has nothing worth cutting is recorded and skipped, and the run
carries on:

```text
--- Batch summary ---
  episode-slice.mp4 -> episode-slice-reel.mp4 (26.7s)
  test.mp4 -- No identity in this video matches lead.jpg. The closest scored 0.08,
              under the 0.35 floor.
  test_2.MOV -- No identity in this video matches lead.jpg. The closest scored 0.04,
                under the 0.35 floor.

1/3 produced a reel, 26.7s of footage in total.
```

Losing nineteen good scans because episode three was a different mode is
the one failure that would make this unusable.

`batch` takes the same scan and encoding options as `export`, so
`--interval`, `--mode`, `--encoder` and the editorial knobs all apply to
every video in the run.

### 11. Scans are kept, not thrown away

A scan is the expensive part — minutes of detecting and embedding on a
full-length video — and it is now kept, so the same video under the same
settings comes back instantly:

```text
Reusing the scan of test_3.mp4 from 4 minutes ago (55s of work skipped).
Pass --rescan to run it again.
```

On the 22-minute test episode a scan takes 55.3s and comes back in 0.05s,
from a 6.3 MB file. The `group` → `export` workflow above no longer scans
the footage twice.

There is no staleness check, because there is nothing to check: the key
covers the video's identity (path, size, modification time) **and** every
setting that can change what a scan produces, so a hit means the same scan
would have given the same answer. A scan-format number is part of it too,
so a release that changes detection or grouping invalidates kept scans
rather than serving an answer the current code would not give — while a
release that changes cutting or the window leaves them, and the names and
corrections stored with them, alone.

Scans kept by 1.9.0 through 1.9.3 are re-filed automatically the first
time each video is opened, so upgrading loses nothing.

```bash
python -m app scans          # what is kept, and how much space it uses
python -m app scans clear    # delete all of it
python -m app export ... --rescan   # ignore what is kept, this once
```

What is stored is the identities — embeddings, boxes, landmarks,
timestamps — plus one representative crop per person for the gallery to
draw, not the footage. A corrupt or half-written entry costs a rescan
rather than a failed scan.

### 12. Fixing what the grouping got wrong

Grouping is tuned against real footage and gets most of a video right. It
still sometimes puts one actor on two cards, or keeps a logo as a
convincing phantom person. In the window, select cards and correct it:

- **Merge** — two or more cards that are the same person.
- **Split…** — a card that holds two people. It shows one shot per *track*
  and you pick the ones that are somebody else.
- **Not a person** — for the phantom identities.

Splitting happens along track boundaries and nowhere else, because a track
is the only unit here that is provably one person: spatial continuity
across consecutive sampled frames proves it, and nothing else does.

Corrections are written back over the kept scan, so a fix survives closing
the window. `--rescan` is the way back to what the clustering actually
said.

### 13. A reel of several people

Click a second card and it joins the first rather than replacing it; on the
command line, pass several indexes:

```bash
python -m app export episode.mp4 --select-index 0 2 --output output/leads.mp4
```

That is every scene *any* of them is in, on one combined timeline — which
is not the same as gluing two reels together. When one lead leaves a scene
and the other arrives a second later, the combined timeline sees one
continuous appearance rather than cutting it in two.

On the test episode, person #1 alone is 152 appearance intervals and person
#2 is 100; the two together are 124, because the scenes they share merge.

### 14. Seeing the reel before encoding it

Selecting a card used to tell you "14 cuts, about 4:31" and then ask you to
commit minutes of encoding on faith. The rail now shows six frames from the
reel itself, spread across its length. It takes 0.30s on a 22-minute
episode.

### 15. Naming the people

Cards start as "Person #2", which is a position rather than a person — and
the position moves every time the gallery is corrected. Select a card,
press **Name…**, and it keeps that name instead:

- The card shows it, and so does the rail.
- It names the exported file: `episode-jamie-lee.mp4` rather than
  `episode-person-2.mp4`.
- It survives merging, splitting and discarding. Merging keeps the name of
  whichever card contributed most, so folding a stray half of an actor into
  the named one keeps the name. Splitting leaves it with whoever stays
  behind, because a split says "those shots are somebody else".
- It is stored with the kept scan, so it is still there when the video is
  reopened.

On the command line a name selects a person, case-insensitively:

```bash
python -m app export episode.mp4 --select-name "Jamie Lee" --output reel.mp4
```

That is more durable than `--select-index 0`, which refers to a position
that any correction can change. Names are limited to 60 characters and
cannot contain the characters a path cannot carry, since they become part
of a filename.

## The desktop window

Everything above is also available as one screen, which is the shorter route if you just want a reel out of a video:

```bash
python -m app ui                                  # or: python -m app.ui
python -m app ui assets/test-videos/test_3.mp4    # with a video preloaded
```

Pick a video, press **Scan for people**, click a face, press **Export reel**. Clicking a second face adds it to the reel rather than replacing the first, and the buttons above the grid name people or merge, split and discard cards the grouping got wrong. Reference photos are command-line only for now, and a name you give a card here is what `batch --person` looks for across a folder. The card grid is the same identity gallery the `group` command writes to a montage, and selecting a card tells you what you are about to get — `Person #2 selected - 14 cuts, about 4:31 of footage` — before you commit to an encode that runs for minutes.

**A whole folder.** Switch the source to **Folder**, pick a season, and scan: each episode is scanned (or its kept scan reused) and the window shows one card per person across all of them, with a strip of their face from each episode and how many they are in. Where two cards look alike but not clearly enough to join on their own, the window asks **Same person?** with the two faces side by side — nothing is guessed. Click a person to see their reel across the folder, name them once to name them in every episode, and export one reel. No photo of anyone is needed.

**Folder** and **File name** are separate fields because they change on different rhythms. A folder is chosen once for a session's worth of reels (**Choose...** opens a directory picker, and a folder that does not exist yet is created on export). The file name follows whichever face is selected — `test_3-person-2.mp4` — but only while it is still the name the app suggested; type your own and it survives clicking through the whole gallery. A missing `.mp4` extension is added for you.

Both long operations run on a worker thread, so the window keeps drawing, the progress bar tracks real work (frames sampled while scanning, cuts encoded while exporting), and the action button turns into **Cancel**. Cancelling a scan stops at the next sampled frame; cancelling an export stops after the current cut and leaves no partial file behind.

The window and the CLI drive the same `run_identity_pipeline`, and `ScanSettings` defaults to the same constants the CLI parser does, so the same video and interval give the same people either way. A test asserts that, because a UI that quietly grouped differently from the command line would be a genuinely confusing thing to debug.

Two defaults differ from the CLI, both deliberately:

- **Sampling defaults to 0.5s**, matching `export` rather than `group`. Someone who opened a window wants a reel.
- **The encoder defaults to `h264_videotoolbox` on Apple silicon**, where it is ~5x faster than `libx264` (4.7s versus 23.9s on the same 12-second reel). It is a visible dropdown rather than a hidden default, and `ExportSettings` still defaults to portable `libx264` for any other caller.

**Quality** is a named level rather than a number, because the two encoders' scales run in opposite directions — `-crf` is 0–51 and lower is better, `-q:v` is 0–100 and higher is better. `Standard`/`High`/`Maximum` translate per encoder. The CLI still takes `--quality` as a raw number, so it wants the right scale for the encoder you picked.

## Building a desktop app

```bash
./packaging/build.sh
```

Out comes `dist/FluxCutter.app` (242 MB) and `dist/FluxCutter-macos.zip` (97 MB, which is what you send people). Most of that is cv2 (89 MB), onnxruntime (71 MB) and PyAV (43 MB); the window itself costs about 2 MB, because the web view belongs to the operating system rather than to the app. On Windows the same script produces `dist/FluxCutter/`. It installs PyInstaller if missing, checks Tk is present before wasting your time, and zips with `ditto` rather than `zip` because a `.app` is a directory and `zip` loses its executable bits.

`--no-zip` builds only; `--run` opens the result. The underlying command is `pyinstaller packaging/FluxCutter.spec --noconfirm` if you would rather drive it yourself.

The models are not bundled — the app downloads and verifies them on first use, which keeps the download to a third of what shipping them would cost.

### Why Windows needs something else

**PyInstaller freezes the interpreter it is running on.** There is no cross-compilation: a Windows `.exe` must be built on Windows. On an Apple Silicon Mac this is not a matter of installing the right tool —

- Wine would have to emulate x86 on ARM *and* host a full PyInstaller run over OpenCV and PyAV. Impractical.
- A Windows-on-ARM VM (UTM) runs natively and quickly, but produces an **ARM64** `.exe` that most people's x64 PCs cannot run.
- An x64 Windows VM on M1 means whole-machine emulation. Hours, if it works.

So the Windows build needs a real x64 Windows machine. Pushing a `v*` tag runs [.github/workflows/build.yml](.github/workflows/build.yml), which rents one from GitHub, builds both platforms, and attaches the zips to a draft release. If you have a Windows PC to hand, `./packaging/build.sh` on it does the same thing without any of that.

### Neither build is signed

Both platforms will warn on first launch. That is expected and it is what a certificate would buy away.

**macOS** quarantines anything downloaded and Gatekeeper reports an unsigned quarantined app as *damaged* — which is a lie, but it is what your users see. Either clear the flag:

```bash
xattr -dr com.apple.quarantine /path/to/FluxCutter.app
```

or open **System Settings → Privacy & Security → Open Anyway**. Right-click → Open no longer works; Apple removed that in Sequoia.

**Windows** shows SmartScreen's "Windows protected your PC" — **More info** → **Run anyway**, once.

Signing costs $99/yr (Apple, and notarization is required for distribution *outside* the App Store, not just inside it) or $200–400/yr (Windows). Worth it for a public download page; not worth it for sending a build to people who can be told about the warning.

## Troubleshooting

### The desktop window will not start

On Windows this means the WebView2 runtime is missing. It ships with Windows 11 and arrives with Edge on Windows 10, so it is present on most machines; where it is not, Microsoft's Evergreen Runtime installer adds it. Everything except `python -m app ui` works without it.

### The virtual environment is broken or packages are missing

If you see import errors such as `ModuleNotFoundError`, stale package state, or a mismatched interpreter, recreate the environment completely:

```bash
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### YuNet model is missing

If you see this error:

```python
FileNotFoundError: YuNet model not found
```

download the model with the command in the setup section and keep it at `assets/models/face_detection_yunet_2026may.onnx`.

### ArcFace model is missing

If you see this error:

```python
FileNotFoundError: ArcFace model not found
```

download it with the command in the setup section and keep it at `assets/models/face_recognition_arcface_w600k_r50.onnx`. This affects the `group` and `timestamps` commands; `info`, `extract`, `detect`, and `gallery` don't need it.

### Face tracking notes

Between detection and grouping, `app/faces/tracker.py` links each frame's detections to the previous frame's by bounding-box overlap (IoU >= 0.3). Two faces at nearly the same place in consecutive sampled frames are the same person by continuity, which is a much stronger signal than comparing two independent single-frame embeddings. Each resulting track is then grouped as one unit using its *averaged* embedding.

This matters because single-frame embeddings are noisy on hard footage — profile angles, motion blur, harsh key light. Averaging over a track lifts that signal without changing any grouping threshold. The effect was pronounced with the previous SFace embedder, where same-person pairs across different shots routinely scored 0.24-0.43 cosine similarity against a 0.45 floor; ArcFace separates those cases better, but track averaging still measurably steadies the estimate.

Two safeguards keep tracks honest:

- **Gap tolerance** (`max_frame_gap`, default 1): a track survives a single missed detection (a blink, brief occlusion, or a frame the detector dropped) before closing.
- **Contradiction veto** (`contradiction_floor`, default 0.25): box overlap alone cannot distinguish a continuing face from a hard cut that places a different person in the same part of the screen. A candidate whose embedding is too dissimilar to the track's *previous frame* breaks the track even when the boxes overlap. The comparison is against the previous frame rather than a running average on purpose — an average is dominated by history once a track has run for a second or two, which lets cut-throughs slip past.

When the veto misfires it errs the cheap way: the grouping stage can re-merge two split tracks of one person, whereas a track that merges two people contaminates a group centroid irreversibly.

### Detector settings

The validated detector configuration is:

- model: `assets/models/face_detection_yunet_2026may.onnx`
- input size: `640x640`
- confidence threshold: `0.6`
- NMS threshold: `0.3`
- top_k: `5000`

This was the best precision and throughput balance we observed on the real 2160 by 2160 test video.

Note that frames are decoded as RGB (`extract_frames` uses `rgb24`) but YuNet is an OpenCV DNN model trained on BGR, so `FaceDetector.detect` swaps the channels before inference. This is not cosmetic: feeding RGB straight through still finds most faces, but on the test video it cost 7 of 23 detections outright and degraded the landmark precision that face alignment depends on, dropping the best same-person similarity from 0.71 to 0.52. That measurement predates the ArcFace swap, but alignment is if anything more sensitive now: ArcFace is fed a crop warped onto a canonical 112x112 pose derived entirely from those same five landmarks.

### Gallery notes

Gallery thumbnails use a default padding ratio of `0.08` around each detection and a `192x192` thumbnail canvas. The gallery also applies a simple representative-sampling pass so it does not flood the grid with near-duplicate detections from adjacent sampled frames.

### MediaPipe crashes or fails to import

This project no longer depends on MediaPipe. Earlier iterations hit environment-specific native crashes and version mismatches on macOS. The current implementation uses OpenCV YuNet, which is the supported path in this repo.

If you still have MediaPipe packages installed from a previous setup, remove them before continuing:

```bash
python -m pip uninstall -y mediapipe
```

### Duplicate FFmpeg/av libraries on macOS

Older setups (installing `opencv-python` instead of `opencv-python-headless`) show warnings about duplicate `libavdevice` symbols when both OpenCV and PyAV are installed, because both packages bundle their own copy of `libavdevice` (OpenCV's camera/GUI backend). This project doesn't use OpenCV's GUI or capture-device features, so `requirements.txt` installs `opencv-python-headless` instead, which doesn't bundle `libavdevice` at all — the warning shouldn't appear if you installed from `requirements.txt`.

If you still see it, confirm `pip show opencv-python-headless` (not `opencv-python`) is installed, or recreate the environment and reinstall cleanly as noted above.

## Usage

The project includes a command-line utility for interacting with videos.

### Getting Video Info

To print metadata for a video file:

```bash
python -m app info assets/test-videos/test.mp4
```

Expected output is a dictionary containing fields such as width, height, codec, frame count, duration, and FPS.

## Example output

```python
{'width': 2160, 'height': 2160, 'codec': 'h264', 'frames': 701, 'duration': 23.366666666666667, 'fps': 30.0}
```

## Testing

```bash
cd flux-cutter
source .venv/bin/activate
pytest tests -q
```

540 tests covering the loader, frame sampling, the detector, the embedder, the tracker, identity grouping, identity corrections and naming, appearance timelines, export segmentation, sound-to-picture sync across joins and across videos, reference-photo and named-person matching, linking a folder's cards into one cast, batch runs, the kept-scan cache, the model downloader, mode selection, and the window's bridge and worker layers. They validate against the real sample video in `assets/test-videos/test.mp4` rather than synthetic frames wherever the stage is about real footage — the detector test confirms it finds a face in actual video while ignoring blank frames.

The tests need no display. `app/ui/worker.py` deliberately imports no toolkit at all, and `app/ui/web.py` keeps every decision on the Python side, so the window's behaviour — which filename to suggest, when to refuse a click, what to do about footage that moved — is tested without opening anything.

The pure decision logic — what counts as one appearance, which segments are worth cutting, when two groups are the same person — is tested without encoding a frame, which is why the suite runs in under a minute despite the pipeline taking minutes on real video.

The current detector configuration is YuNet on CPU with a `640x640` detection canvas and a `0.6` confidence threshold, which was the best precision/throughput tradeoff observed on the test footage.

The gallery stage uses the same real sample video and the already-validated detector output; it only adds cropping, thumbnail generation, representative sampling, and grid rendering.

## Notes

- Supported video formats are currently limited to `.mp4` and `.mov`.
- The loader raises a `VideoLoadError` when the file is missing, not a regular file, wrongly formatted, or unreadable.
- This project is intended as a working prototype for validating the core concept before expanding into a more polished application.

## Roadmap

The next logical prototype milestones are:

- ~~frame extraction and preview validation~~
- ~~face detection accuracy checks on real footage~~
- ~~face tracking across consecutive frames~~
- ~~person grouping and identity clustering~~ (working; tuned against one 22-minute video, so treat the thresholds as fitted to that footage until a second one confirms them)
- ~~clip selection and merge logic~~
- ~~final exported video composition~~
- ~~a desktop window over the whole flow~~
- ~~sampling faster than a full sequential decode~~ — measured, and the lever this roadmap named is not there: **seeking is 3.5–5x slower**, even done once per GOP rather than once per sample, because each seek flushes the decoder and gives up the multi-core decoding that made sequential reading fast. Skipping non-reference frames pays instead: ~13% off decode, ~5% off a whole scan, with the provable-false-merge count unchanged. See [Instructions.md](Instructions.md#18-sampling-faster-what-the-roadmap-expected-and-what-measured).
- ~~choose a person with a photo instead of a gallery click~~
- ~~run a whole folder of videos in one command~~
- ~~keep a scan, so re-opening a video does not re-run the detect/embed pass~~ (55.3s becomes 0.05s on the test episode)
- ~~merge, split and discard person cards, so a grouping mistake can be corrected without re-tuning thresholds~~
- ~~cut a reel of several people at once~~
- ~~see frames from the reel before committing to an encode~~
- ~~name the people, so a card is a person rather than a position~~
- ~~cut one person out of a folder of videos into one reel, found by a name given in the window~~
- ~~the same in the window: scan a folder, pick someone from one cast of face cards, answer "same person?" when a link is unclear~~
- **next:** leave repeated footage (intros, recaps) out of a season reel, then snap cuts to shot boundaries so a reel holds the character's shots rather than both sides of every conversation

This roadmap may evolve as the prototype proves which stages need refinement.
