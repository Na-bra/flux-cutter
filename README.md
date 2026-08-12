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
│   └── test-videos/
│       └── test.mp4
├── tests/
│   ├── __init__.py
│   └── test_video.py
├── Instructions.md
├── README.md
└── .gitignore
```

## Setup

Use a virtual environment for local development:

```bash
cd flux-cutter
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install av pytest
```

## Running the video loader

The project includes a small video validation utility that opens a supported MP4 and prints basic stream metadata:

```bash
cd flux-cutter
source .venv/bin/activate
python -c "from app.video.loader import load_video, get_video_info; c=load_video('assets/test-videos/test.mp4'); print(get_video_info(c)); c.close()"
```

Expected output is a dictionary containing fields such as width, height, codec, frame count, duration, and FPS.

## Example output

```python
{'width': 2160, 'height': 2160, 'codec': 'h264', 'frames': 701, 'duration': 23.366666666666667, 'fps': 30.0}
```

## Testing

The repository includes a basic video test file for validating the loader and metadata extraction logic:

```bash
cd flux-cutter
source .venv/bin/activate
pytest tests/test_video.py -q
```

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
