# FluxCutter

FluxCutter is a Python prototype for identifying a person in a video and compiling all of that person's appearances into a single exportable clip. The goal is to turn a raw video into a focused "character reel" by detecting faces, grouping them by identity, and selecting the moments where the target person appears.

This project is currently in an early prototype stage focused on validating the workflow rather than polishing a production-ready application.

## What the project does

The intended workflow is:

1. Load a video file
2. Extract frames for inspection
3. Detect human faces in each frame
4. Group detections that belong to the same person
5. Show the detected people in a face gallery
6. Let the user choose a target person
7. Find every matching appearance in the timeline
8. Merge nearby clips and export the final result

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
│   │   └── face_detection_yunet_2026may.onnx
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
python -m pip install av numpy pytest opencv-python pillow
```

### 3. Download the YuNet and SFace models

The detector uses the official OpenCV Zoo YuNet model, and identity grouping uses the OpenCV Zoo SFace model. Download both once and keep them in the repository-local model directory:

```bash
mkdir -p assets/models
curl -L --fail -o assets/models/face_detection_yunet_2026may.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx
curl -L --fail -o assets/models/face_recognition_sface_2021dec.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Both models load through `opencv-python`'s DNN module (`cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`), so no extra Python package is needed beyond what's already installed.

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
python app/main.py gallery assets/test-videos/test.mp4 --interval 1.0 --output-dir output/face-gallery
```

You can also select an item while generating the gallery:

```bash
python app/main.py gallery assets/test-videos/test.mp4 --interval 1.0 --select-index 0
```

### 6. Group detections into identities

The group command runs the same sampling and detection pass, embeds each face with SFace, and groups them into per-identity clusters using nearest-centroid matching with a similarity floor and a margin over the runner-up group (to avoid forcing an ambiguous match). It saves one representative thumbnail per identified person:

```bash
python app/main.py group assets/test-videos/test.mp4 --interval 1.0 --output-dir output/face-groups
```

Grouping behavior is tunable:

```bash
python app/main.py group assets/test-videos/test.mp4 \
	--similarity-threshold 0.45 \
	--margin-threshold 0.05 \
	--min-confidence 0.7 \
	--min-face-size 40
```

See [Instructions.md](Instructions.md#7-stage-02-accuracy-notes-identity-grouping) for how these defaults were chosen against the real test footage.

### 7. Compute appearance timestamps for one person

Once you know which person card you want (from the `group` command's output or montage), the `timestamps` command runs the same grouping pass and converts that one person's detections into appearance intervals — contiguous spans of time they're on screen, gap-split, padded, and clamped to the video's duration:

```bash
python app/main.py timestamps assets/test-videos/test.mp4 --interval 0.5 --select-index 0
```

`--select-index` refers to the same ordering shown by `group` (largest identity group first). Gap tolerance and padding default to sampling-derived values (`2x` and `0.5x` the `--interval`, respectively) but can be overridden:

```bash
python app/main.py timestamps assets/test-videos/test.mp4 --interval 0.5 --select-index 0 \
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
python -m pip install av numpy pytest opencv-python pillow
```

### YuNet model is missing

If you see this error:

```python
FileNotFoundError: YuNet model not found
```

download the model with the command in the setup section and keep it at `assets/models/face_detection_yunet_2026may.onnx`.

### SFace model is missing

If you see this error:

```python
FileNotFoundError: SFace model not found
```

download it with the command in the setup section and keep it at `assets/models/face_recognition_sface_2021dec.onnx`. This only affects the `group` command; `info`, `extract`, `detect`, and `gallery` don't need it.

### Detector settings

The validated detector configuration is:

- model: `assets/models/face_detection_yunet_2026may.onnx`
- input size: `640x640`
- confidence threshold: `0.6`
- NMS threshold: `0.3`
- top_k: `5000`

This was the best precision and throughput balance we observed on the real 2160 by 2160 test video.

### Gallery notes

Gallery thumbnails use a default padding ratio of `0.08` around each detection and a `192x192` thumbnail canvas. The gallery also applies a simple representative-sampling pass so it does not flood the grid with near-duplicate detections from adjacent sampled frames.

### MediaPipe crashes or fails to import

This project no longer depends on MediaPipe. Earlier iterations hit environment-specific native crashes and version mismatches on macOS. The current implementation uses OpenCV YuNet, which is the supported path in this repo.

If you still have MediaPipe packages installed from a previous setup, remove them before continuing:

```bash
python -m pip uninstall -y mediapipe
```

### Duplicate FFmpeg/av libraries on macOS

You may see warnings about duplicate `libavdevice` symbols when both OpenCV and PyAV are installed. This is usually noisy but not fatal if the app still runs. The project is validated with the current combination of `opencv-python` and `av`.

If the app becomes unstable, recreate the environment and reinstall cleanly as noted above.

## Usage

The project includes a command-line utility for interacting with videos.

### Getting Video Info

To print metadata for a video file:

```bash
python app/main.py info assets/test-videos/test.mp4
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

- frame extraction and preview validation
- face detection accuracy checks on real footage
- person grouping and identity clustering
- clip selection and merge logic
- final exported video composition

This roadmap may evolve as the prototype proves which stages need refinement.
