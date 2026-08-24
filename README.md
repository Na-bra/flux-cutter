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

The project structure is intentionally simple and lightweight so experimentation remains easy.

## Project structure

```text
flux-cutter/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── faces/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── embedder.py
│   │   ├── tracker.py
│   │   └── grouper.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── gallery.py
│   └── video/
│       ├── __init__.py
│       ├── frames.py
│       └── loader.py
├── assets/
│   ├── models/
│   │   ├── face_detection_yunet_2026may.onnx
│   │   └── face_recognition_arcface_w600k_r50.onnx
│   └── test-videos/
│       └── test.mp4
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_video.py
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

`requirements.txt` pins `opencv-python-headless` rather than `opencv-python`: the headless build skips OpenCV's camera/GUI backend, which is what caused the duplicate-`libavdevice`-symbol warning below when paired with PyAV. The project has no GUI/capture-device usage, so this is a safe swap and not a downgrade in capability.

### 3. Download the YuNet and ArcFace models

The detector uses the official OpenCV Zoo YuNet model. Identity grouping uses ArcFace (`w600k_r50`, a ResNet50 trained on WebFace600K), shipped inside InsightFace's `buffalo_l` bundle. Download both once and keep them in the repository-local model directory:

```bash
mkdir -p assets/models
curl -L --fail -o assets/models/face_detection_yunet_2026may.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx

# ArcFace ships in a bundle; extract just the recognition model and rename it.
curl -L --fail -o /tmp/buffalo_l.zip \
	https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o -j /tmp/buffalo_l.zip 'w600k_r50.onnx' -d assets/models
mv assets/models/w600k_r50.onnx assets/models/face_recognition_arcface_w600k_r50.onnx
```

The `buffalo_l` download is ~290MB but only `w600k_r50.onnx` (~174MB) is kept; the other models in the bundle are detection/landmark/attribute models this project doesn't use.

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

The group command runs the same sampling and detection pass, embeds each face with ArcFace, links detections into face tracks, and groups those tracks into per-identity clusters using nearest-centroid matching with a similarity floor and a margin over the runner-up group (to avoid forcing an ambiguous match). It saves one representative thumbnail per identified person:

```bash
python -m app group assets/test-videos/test.mp4 --interval 0.25 --output-dir output/face-groups
```

Grouping quality depends heavily on the sampling interval, because tracking is what makes it work. Denser sampling gives the tracker consecutive frames it can actually link, and a track's averaged embedding is far more reliable than any single frame's. On the test video:

| `--interval` | detections | tracks | identity groups |
| ------------ | ---------- | ------ | --------------- |
| 1.0s         | 23         | 17     | 10              |
| 0.5s         | 49         | 33     | 19              |
| 0.25s        | 88         | 38     | 24              |

At 1.0s, shots cut and faces jump between samples, so tracks mostly degenerate to length 1 and the stage does nothing. Prefer 0.5s or denser for real grouping work.

These counts are from the current ArcFace embedder with agglomerative grouping, and they are **not** comparable to counts recorded before either change — the group total moves for reasons that have nothing to do with accuracy, so compare montages rather than totals.

Grouping errors on this footage are now overwhelmingly one-sided: the pipeline splits one person into several cards far more often than it merges two people into one. Most of the surplus cards are single-detection groups from genuinely hard frames — heavy motion blur, extreme profile, near-darkness — where the embedding is unreliable and correctly matches nothing. Reducing them is a crop-quality problem (gating blurry detections), not a threshold problem; lowering the floor to absorb them re-introduces real false merges.

Grouping behavior is tunable:

```bash
python -m app group assets/test-videos/test.mp4 \
	--similarity-threshold 0.35 \
	--margin-threshold 0.05 \
	--min-confidence 0.7 \
	--min-face-size 40
```

See [Instructions.md](Instructions.md#7-stage-02-accuracy-notes-identity-grouping) for how these defaults were chosen against the real test footage.

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

## Troubleshooting

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

The repository includes validation for both the video loader and the detector:

```bash
cd flux-cutter
source .venv/bin/activate
pytest tests -q
```

This project is currently expected to pass all checks in the included suite. The test coverage validates the actual sample video in `assets/test-videos/test.mp4` and confirms that the face detector finds a face in real footage while ignoring blank frames.

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
- person grouping and identity clustering (working, still over-splits on hard footage)
- clip selection and merge logic
- final exported video composition

This roadmap may evolve as the prototype proves which stages need refinement.
