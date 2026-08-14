# FluxCutter — Development Instructions

## 1. Project Overview

FluxCutter ("Face Cutter") is a Python desktop application prototype for video editors. It lets a user select a specific person appearing in a video and automatically compiles all of that person's on-screen appearances into a single exported clip.

**Core workflow:**

1. Import a video.
2. Let the user verify the imported video (preview / metadata check).
3. Detect faces across the video's frames.
4. Group detections belonging to the same person.
5. Display the detected people as a face gallery.
6. Let the user select a person from the gallery.
7. Find that person's appearances throughout the video.
8. Record timestamps for those appearances.
9. Extract the relevant clips and merge nearby ones.
10. Export the resulting compiled video.

**Current goal:** prototype validation, not production readiness. Every design decision below should be judged against one question — *does this help prove the workflow works?*

---

## 2. Current Development Goal

The immediate goal is to prove the core technical workflow — detection, grouping, tracking, extraction — is feasible with acceptable accuracy and performance. Do not attempt to build the full application in one pass. Development proceeds through small, demonstrable prototype iterations, each validated before the next begins.

### Prototype sequence

| Stage | Input → Output | Proves |
|:-----:|-----------------|--------|
| **0.1** | Video → frame extraction → face detection → face gallery | Faces can be reliably detected, cropped, and displayed |
| **0.2** | Face gallery → selected person → grouping → timestamps | Detections can be clustered per-identity and located in time |
| **0.3** | Timestamps → clip extraction → nearby-clip merging | Clip boundaries and merge logic behave sensibly |
| **0.4** | Merged clips → final video export | End-to-end pipeline produces a usable output file |

**Exit criteria per stage:** a prototype is "done" only when it runs end-to-end on at least one real test video and its output can be manually inspected (gallery images shown, timestamps printed, clips playable, final video watchable). Advance to the next stage only after that check passes — don't stack unvalidated stages on top of each other.

---

## 3. Development Philosophy

### Keep the prototype simple
Do not introduce architecture, abstractions, libraries, or infrastructure unless they solve a problem that has actually shown up in the code. Avoid premature production architecture.

Do not create folders such as `controllers/`, `repositories/`, `services/`, `factories/`, `managers/`, or `utils/` unless the codebase has demonstrably outgrown flat modules. Prefer simple, readable, single-purpose Python modules.

### Do not over-engineer
A working prototype beats an elaborate architecture. If two simple modules solve a problem, don't build a framework around it. If a requirement isn't needed for the current prototype stage, defer it — note it, don't build it.

### Validate assumptions early
When a technical assumption is uncertain (detection accuracy on low-res footage, clustering quality across lighting changes, FFmpeg behavior on odd codecs, etc.), write a small script to test it against real sample footage rather than designing around an unverified assumption. A five-minute test script beats an hour of speculative design.

### Out of scope for now
To keep iterations focused, the following are explicitly deferred until 0.1–0.4 are validated:

- Multi-person simultaneous tracking/export
- GUI polish, theming, or packaging as a standalone executable
- Batch processing of multiple videos
- Performance optimization beyond "runs in reasonable time on a test clip"
- Error handling beyond what's needed to keep the prototype from crashing outright

---

## 4. Project Structure

The initial structure should remain approximately:

```text
FluxCutter/
│
├── README.md
├── INSTRUCTIONS.md
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── video/
│   │   ├── __init__.py
│   │   ├── loader.py       # import + verify
│   │   └── frames.py       # frame extraction
│   │
│   ├── faces/
│   │   ├── __init__.py
│   │   ├── detector.py     # per-frame face detection
│   │   └── grouper.py      # clustering detections into identities
│   │
│   └── ui/
│       ├── __init__.py
│       └── gallery.py      # face gallery + selection
│
├── tests/
│   ├── __init__.py
│   └── test_video.py
│
└── assets/
    └── test_videos/
```

This structure should grow only in response to a stage actually needing it — e.g. `app/faces/tracker.py` and `app/video/export.py` are natural additions once stages 0.2 and 0.3/0.4 begin, but shouldn't be scaffolded in advance.

---

## 5. Working Agreement

- One prototype stage at a time. Don't start 0.2 code while 0.1 is unvalidated.
- Prefer a script that proves a point over a polished module.
- If a library choice (face detection, clustering, video I/O) is still undecided, treat picking it as part of stage 0.1 — try the simplest viable option first, swap later only if it fails on real footage.
- Keep this document updated as decisions are made; it should reflect the actual state of the project, not just the plan.

---

## 6. Open Decisions

Track unresolved technical choices here as they come up, and resolve them in the stage that needs them rather than up front. Suggested starting points:

| Decision | Needed by | Status |
|---|---|---|
| Face detection library (e.g. `face_recognition`, `mediapipe`, `insightface`) | 0.1 | Resolved: OpenCV DNN + YuNet 2026may on CPU. Best practical precision/throughput tradeoff on the test footage. |
| Face embedding model for identity grouping | 0.2 | Resolved: OpenCV Zoo SFace (`face_recognition_sface_2021dec.onnx`) via `cv2.FaceRecognizerSF`. Same ecosystem as YuNet (consumes its landmarks directly for alignment via `alignCrop`), zero new dependencies, CPU-only, cross-platform. |
| Clustering approach for grouping detections into identities | 0.2 | Resolved: incremental nearest-centroid matching (not a clustering framework) with an absolute cosine-similarity floor *and* a margin over the second-best group, so ties fall through to a new group instead of forcing a merge. See accuracy notes below. |
| Video I/O / frame extraction tooling (e.g. OpenCV, PyAV) | 0.1 | Resolved: PyAV for loader/frame extraction. It already works reliably with the sample video. |
| GUI toolkit for the face gallery view | 0.1 | Resolved for prototype 0.1: simple saved gallery montage via OpenCV grid rendering. |
| Appearance-interval strategy (detections -> timestamps) | 0.2/0.3 boundary | Resolved: real per-detection PTS timestamps (already carried on `FaceObservation.source_timestamp`) grouped into contiguous spans by a sampling-derived gap tolerance, then padded by a sampling-derived amount and clamped to video bounds. No new frame decoding or tracking added. See accuracy notes below. |
| Clip extraction / merge tooling (e.g. FFmpeg via subprocess vs. a Python wrapper) | 0.3 | Not yet decided |

Update the Status column as each is resolved, and note *why* — a one-line rationale is enough to save re-litigating it later.

---

## 7. Stage 0.2 Accuracy Notes (Identity Grouping)

OpenCV Zoo documents 0.363 cosine similarity as SFace's verification
threshold on standard benchmarks. Running the actual test video through
the pipeline (`python app/main.py group ...`) at that value produced one
clearly wrong merge: 7 detections spanning nearly the full clip got
chained into a single group via centroid drift, even though the pairwise
cosine similarity between most of those 7 crops was only 0.15-0.48 (well
below threshold) — only two adjacent-timestamp pairs were genuinely the
same shot (0.75 and 0.48). Visually, the merged group mixed at least two
different men who happened to share an open-mouth/teeth-baring
expression.

Raising `DEFAULT_SIMILARITY_THRESHOLD` to 0.45 (`app/faces/grouper.py`)
broke that cluster apart into its constituent identities while still
correctly merging genuine same-shot repeats a few seconds apart (verified
by re-cropping and visually inspecting each multi-detection group). Above
~0.55, even same-shot repeats 0.5s apart stopped merging — recall dropped
without a corresponding gain in precision. 0.45 with a 0.05 margin was
the best point found on this footage: no observed false merges, some
missed matches (an actor's later scenes sometimes seeding a new group
instead of rejoining an earlier one), which is the intended tradeoff per
the "false merge is worse than a missed match" requirement for this
stage.

This was validated on one ~23s test clip; treat 0.45 as a starting point
to re-check once more/longer footage is available, not a universal
constant.

---

## 8. Appearance Timestamp Notes (`app/video/timeline.py`)

Detections already carry the real decoded-frame PTS in seconds
(`app/video/frames.py` sets `timestamp = float(frame.time)`, not a
reconstructed `sample_index * interval`), so `build_appearance_intervals`
didn't need to touch timestamp derivation — only decide which nearby
detections belong to one contiguous appearance.

Both the gap tolerance (how far apart two detections can be before
they're treated as separate appearances) and the padding (how much
buffer to add around each appearance) are derived from the actual
`--interval` used for that run, not fixed constants:

- gap tolerance defaults to `2 x sample_interval` — one missed sample is
  tolerated as noise before splitting into a new appearance.
- padding defaults to `0.5 x sample_interval` — a detection only proves
  the person was on screen within about half a sampling step of it.

Validated against the real test clip at `--interval 0.5`: a person
correctly grouped as one identity across 4 detections at
10.5s/12.0s/12.5s/17.0s produced 3 separate appearance intervals rather
than one 10.5s-17.0s span, because pulling the actual frames at those
timestamps shows two distinct shots (a close-up, then a wider lab-coat
shot) with a real cut between them — confirming the gap-based split was
correct, not a bug. A second, closely-spaced pair (11.0s/11.5s, the same
shot) correctly merged into one appearance. See the stage-2/3 boundary
final report in conversation history for the full validation transcript.

Known limitation: none of this has frame-level precision — a boundary is
only known to within about half a sampling interval, since frames
between samples were never inspected. At the current default
`--interval 1.0`, that's a ~0.5s fuzz band on every boundary. Tightening
`--interval` narrows it at the cost of more detection/embedding work per
run.