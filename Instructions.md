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

What has actually been added since, and why: `app/faces/tracker.py` and `app/faces/embedder.py` (0.2), `app/video/timeline.py` (0.2), `app/video/export.py` (0.3/0.4), `app/__main__.py` (the CLI, split out of `main.py` once it outgrew it), and `app/ui/app.py` + `app/ui/worker.py` (the desktop window, section 7k).

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
| Face embedding model for identity grouping | 0.2 | Resolved: ArcFace `w600k_r50` (InsightFace `buffalo_l`) via `cv2.dnn.readNetFromONNX`, 512-d. Superseded OpenCV Zoo SFace, which was over-splitting the same actor across shots (49 detections -> 23 groups at 0.5s sampling; ArcFace gives 10). Still zero new dependencies (no torch/onnxruntime) and CPU-only, but alignment had to be reimplemented since `alignCrop` is SFace-specific, and it costs ~4.6x more per face. |
| Clustering approach for grouping detections into identities | 0.2 | Resolved: agglomerative average-linkage over units (a unit = one observation, or a whole track). Replaced incremental nearest-centroid, which was order-dependent enough to produce 11-17 groups from the same 33 tracks depending only on arrival order. Keeps the similarity floor and the margin rule, the latter re-derived for linkage semantics. See 7b. |
| Video I/O / frame extraction tooling (e.g. OpenCV, PyAV) | 0.1 | Resolved: PyAV for loader/frame extraction. It already works reliably with the sample video. |
| GUI toolkit for the face gallery view | 0.1 | Resolved for prototype 0.1: simple saved gallery montage via OpenCV grid rendering. |
| Appearance-interval strategy (detections -> timestamps) | 0.2/0.3 boundary | Resolved: real per-detection PTS timestamps (already carried on `FaceObservation.source_timestamp`) grouped into contiguous spans by a sampling-derived gap tolerance, then padded by a sampling-derived amount and clamped to video bounds. No new frame decoding or tracking added. See accuracy notes below. |
| Clip extraction / merge tooling (e.g. FFmpeg via subprocess vs. a Python wrapper) | 0.3 | Resolved: FFmpeg via subprocess. PyAV is already a dependency, but cutting needs timestamp rebasing and A/V sync across concatenated segments, which ffmpeg's `-ss`/`-t` and concat demuxer already solve correctly and PyAV would mean hand-writing. Cost, recorded honestly: this is the project's one external *binary* dependency, a real departure from the zero-new-dependencies habit elsewhere. See 7h. |

Update the Status column as each is resolved, and note *why* — a one-line rationale is enough to save re-litigating it later.

---

## 7. Stage 0.2 Accuracy Notes (Identity Grouping)

OpenCV Zoo documents 0.363 cosine similarity as SFace's verification
threshold on standard benchmarks. Running the actual test video through
the pipeline (`python -m app group ...`) at that value produced one
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

### 7a. Re-tune after ArcFace replaced SFace

Everything above describes SFace. When ArcFace (`w600k_r50`) replaced it
as the embedder, 0.45 stopped being meaningful: a threshold is a property
of the model's similarity distribution, not a portable constant. On the
same footage, ArcFace pushes different-person pairs much closer to zero
(median pairwise similarity 0.077, vs SFace's 0.154) while holding the
best same-person pair slightly higher (0.771 vs 0.706) — i.e. better
separation, but a different operating range.

Re-running the same experiment (build tracks once, then re-group at each
candidate threshold, watching for the wide-span/low-internal-similarity
signature of a false merge) showed a sharp cliff rather than a gentle
curve:

| threshold | groups | worst merge |
| --------- | ------ | ----------- |
| <= 0.33   | 7      | 9 detections spanning 19.0s, weakest internal pair +0.070 |
| 0.34      | 10     | 3 detections spanning 2.0s, weakest pair +0.274 |
| 0.35-0.37 | 11     | 3 detections spanning 2.0s, weakest pair +0.274 |
| 0.42-0.45 | 13     | 2 detections spanning 1.0s, weakest pair +0.274 |

At 0.33 and below the same failure mode as the original SFace experiment
persists — centroid drift chaining clearly different people (a pair at
+0.070 similarity is not one person). It disappears at 0.34.

`DEFAULT_SIMILARITY_THRESHOLD` is now **0.35**: the lowest value on the
stable 0.35-0.37 plateau, chosen over the literal cliff-edge value of
0.34 because sitting exactly on a discontinuity is fragile against
footage variation. `DEFAULT_MARGIN_THRESHOLD` stays 0.05 — re-validated,
not merely inherited: grouping is byte-identical anywhere in 0.00-0.08,
so 0.05 sits comfortably inside that flat region.

Known remaining weakness: visual inspection of the 1.0s montage shows one
blonde actor still split across several person cards. That is the
intended direction of error ("a false merge is worse than a missed
match"), but it means recall across shots is still the weak axis. If that
becomes the priority, lowering toward 0.34 or adding a deliberate
cross-shot merge pass are the levers — not raising the threshold.

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

---

## 7b. Grouping algorithm change (nearest-centroid -> agglomerative)

The incremental nearest-centroid grouper assigned each unit to the best
group that existed *at the moment it arrived*, then folded it straight
into that group's centroid. Two consequences, both measured on the test
footage rather than assumed:

- **Order dependence.** Feeding the same 33 tracks in 12 different
  shuffled orders, with identical data and identical thresholds, produced
  group counts of 11, 12, 13, 14, 15 and 17. Any measurement of an
  accuracy change smaller than that swing was noise.
- **Centroid pollution.** One wrong early assignment permanently moved a
  centroid, which then attracted further wrong matches. This is the
  "centroid drift" 7 already described; it was a property of the
  algorithm, not of the embedding.

`IdentityGrouper` now buffers units and clusters them with agglomerative
average linkage: repeatedly merge the globally most-similar pair of
clusters until the best remaining pair falls below the similarity floor.
Linkage similarity is the mean cosine similarity over all cross-cluster
observation pairs, updated after each merge by the Lance-Williams rule
(exact, and keeps the run O(n^2) in units rather than recomputing every
pair). Re-running the shuffle test now yields byte-identical grouping
across every order, at every threshold tried.

Because clustering needs every unit up front, `add`/`add_track` only
buffer; grouping runs on first access to `groups` (or `finish()`). They
return a *unit index* rather than a group id — group ids do not exist
until clustering runs. The pipeline already collected all tracks before
grouping, so nothing streaming was given up.

**The margin rule had to be re-derived, not ported.** Under
nearest-centroid, "the runner-up is nearly as similar" meant one unit
matched two rival identities about equally well. Under average linkage it
usually means the opposite: three clips of one person are all mutually
similar, so every pair scores high, and a naive runner-up test blocks the
very merges it should allow — a literal port split three identical faces
into three groups. A near-tie now counts as ambiguity only when the
competitor is a *different* identity: close to one endpoint, yet too far
from the other to merge with it.

`DEFAULT_SIMILARITY_THRESHOLD` stays 0.35. Re-sweeping under the new
linkage put the knee in the same place: at 0.32 and below a group spanning
20.5s survives whose weakest internal pair is only +0.172, and at 0.35 the
worst surviving merge is a 2.5s same-shot group at +0.274.

### Remaining error profile

Errors are now strongly one-sided: over-splitting, not false merging. At
0.5s sampling, 8 of 19 groups are single detections, nearly all from
heavily blurred, extreme-profile or near-black crops. Some genuine
cross-shot merges that previously failed now succeed (one actor's card
spans 1.5s-22.0s).

The next lever is therefore **crop quality, not thresholds**: a blur/
quality gate that stops unreliable crops from seeding their own identity
card, alongside the existing confidence and face-size floors, which do not
measure blur at all. Lowering the similarity floor to absorb those
singletons is the wrong fix — the sweep shows it re-admits real false
merges well before it rescues them.


---

## 7c. First full-length run (`test_3.mp4`, 22.6 min)

Everything before this was tuned on one 23-second clip. The first run on a
full episode (1280x720, 1355s, ~26 recurring characters) changed one
conclusion outright.

Baseline at 1.0s sampling: 3111 detections -> 1936 tracks -> **412 identity
groups**, 222s wall (94s embedding), peak RSS 4.87 GB.

**What worked.** The main cast came out clean and stable across the whole
episode: the largest group held 348 detections spanning 18s-1295s, the next
230 spanning 15s-1291s. Cross-shot identity matching -- the thing ArcFace
and agglomerative linkage were adopted for -- does its job on clear footage.

**What did not.** 294 of the 412 groups held <= 2 detections. Splitting
that tail by each group's best similarity to any main character:

| best sim to a main character | groups | share | median blur |
| --- | --- | --- | --- |
| >= 0.35 (already above the floor) | 159 | 54.1% | 86 |
| 0.25 - 0.35 (near miss) | 51 | 17.3% | 51 |
| 0.15 - 0.25 (weak) | 52 | 17.7% | 45 |
| < 0.15 (not a usable face) | 32 | 10.9% | 21 |

The majority were **already above the similarity floor and refused
anyway**, which ruled out the threshold as the cause and pointed at the
margin rule. Confirmed by sweeping it (see `DEFAULT_MARGIN_THRESHOLD` for
the numbers): disabling it alone took 412 groups -> 221 and lifted
main-character coverage from 64.5% to 78.8%, without pushing group
cohesion anywhere near the different-person range. `DEFAULT_MARGIN_THRESHOLD`
is now 0.0.

The lesson worth keeping: the 23s clip could not distinguish "this
parameter is harmless" from "this clip has too few faces to exercise it".
Parameters validated only on `test.mp4` should be treated as unvalidated
until a full-length video exercises them.

### Still open after this run

- **Crop quality gate.** 23.7% of detections have Laplacian variance < 40
  on the aligned crop, and the tail is full of backs of heads, extreme
  profiles and motion blur that YuNet scores 0.72-0.89 -- above the 0.7
  confidence floor, so nothing currently filters them. A blur gate at 20
  plus the margin change gives 175 groups / 80.6% coverage, versus 221 /
  78.8% for the margin change alone. Blurred detections should land in
  `unassigned` rather than being dropped silently.
- **Frame extraction does not stream.** `extract_frames` returns a list of
  every sampled frame, so memory scales with video length x sampling rate:
  ~4.9 GB at 1.0s here, ~7.5 GB at 0.5s, ~15 GB at 0.25s. The README
  recommends 0.5s or denser for real grouping work, which on a
  feature-length video will not fit in memory. Making it a generator
  removes the ceiling and is likely worth more on real footage than
  further accuracy tuning.
- **Representative thumbnails can misrepresent a group.** `_observation_quality`
  is confidence x box area, so a large blurry crop outranks a small sharp
  one; several montage cards show a hair or neck crop for a group whose
  members are mostly clean faces. Folding sharpness into that score would
  fix the montage without touching grouping.


---

## 7d. Streaming frame extraction

`extract_frames` returned a list of every sampled frame, so peak memory was
roughly `width x height x 3 x (duration / sample_interval)`. That put the
sampling density identity grouping actually wants out of reach on real
footage: on the 22.6-minute 720p episode it needed ~4.9 GB at a 1.0s
interval and would have needed ~15 GB at 0.25s, against 17 GB of machine.
The README recommends 0.5s or denser for grouping work, so the recommended
setting was also the unaffordable one.

It now yields one decoded frame at a time. Measured on `test_3.mp4`,
1.0s interval, same machine:

| | list | streaming |
| --- | --- | --- |
| peak RSS | 4.87 GB | **1.04 GB** |
| wall clock | 221.9s | **216.3s** |
| identity groups | 221 | 221 |

Memory is now flat in video length rather than proportional to it, output
is unchanged, and there is no throughput cost -- interleaving decode with
detection turned out to be free here.

Worth noting for anyone reading the run report: "Total processing time"
went from 108s to 267s across this change **without anything getting
slower**. Decoding used to happen eagerly before the timer started and now
happens inside it, so the number covers strictly more work. Wall clock is
the only figure comparable across the change.

### Consequences for callers

Validation stays eager. A plain generator function defers its entire body
to first iteration, which would have meant a bad `sample_interval` raising
somewhere far from the call that caused it, so `extract_frames` validates
its arguments and *returns* an inner generator.

Two lazy-evaluation hazards, both of which the existing tests walked into:

- **The result is a one-shot iterator with no length.** Callers that
  reported `len(frames)` now tally as they go; `_run_identity_pipeline`
  returns a `_PipelineResult` carrying `frame_count` and `last_timestamp`
  because its callers used to read those off the materialized list.
- **Iteration must finish inside the `with load_video(...)` block.**
  Decoding happens while iterating, so consuming the iterator after the
  container closes reads from a closed file. `test_embedder`'s fixture did
  exactly this and passed only because a list had already been built.

`run_appearance_timestamps` needed a small restructure for the same
reason: its duration fallback read `frames[-1][0]`, and the last timestamp
of a stream is not knowable until the stream is spent, so that fallback now
resolves after the pipeline rather than before it.


---

## 7e. Consolidation pass (duplicate identities among large groups)

After the margin fix, `test_3.mp4` still produced visibly duplicated people
-- the same actor appearing as several separate person cards. Two theories
were tested and both were wrong, which is worth recording so they are not
retried:

- **Orphan fragments failing to attach.** A two-stage absorber that pulled
  small groups into large ones on strong nearest-frame evidence moved the
  count only 305 -> 274. The leftover groups are genuinely marginal, not
  near-misses.
- **Pose manifolds needing nearest-neighbour linkage.** Single linkage
  chained badly: at threshold 0.55 the largest cluster swelled to 1155
  observations and its 5th-percentile cohesion collapsed to 0.139, joining
  different people through one lucky frame each.

The duplicates were not in the tail at all -- they were **among the largest
groups**. Comparing the 30 biggest:

| pair | sizes | centroid sim | average linkage |
| --- | --- | --- | --- |
| #1 vs #24 | 447 / 31 | 0.626 | 0.315 |
| #6 vs #12 | 100 / 50 | 0.550 | 0.275 |
| #3 vs #25 | 186 / 29 | 0.549 | 0.245 |
| next closest | 178 / 34 | 0.280 | 0.152 |

Each of the top three is one actor split in two, and average linkage puts
all three below the 0.35 floor. Rendering the pairs showed why: #1 vs #24
is the blonde lead **with and without a costume mask**. One person can
occupy two distant regions of embedding space, and average linkage asks
whether a candidate resembles *every* frame of a cluster -- the wrong
question once a cluster is large and varied.

`IdentityGrouper._consolidate` runs after clustering and folds together
whole groups whose centroids agree, greedily on the best pair, so it is
order-independent like the clustering phase. `DEFAULT_CONSOLIDATION_THRESHOLD`
is 0.50, picked from the gap above (true duplicates 0.549-0.626, next
candidate 0.280); anything in 0.30-0.54 separates them.

It is a second phase rather than the linkage rule itself because centroid
linkage from the start is far looser while clusters are one or two frames
wide, where a single noisy embedding *is* the prototype.

Measured on `test_3.mp4` at 1.0s, cumulative with the earlier margin change:

| | groups |
| --- | --- |
| original (margin 0.05, no consolidation) | 412 |
| margin 0.0 | 221 |
| margin 0.0 + consolidation | **186** |

Person #1 grew 348 -> 446 -> 516 detections and now spans 18.0s-1325.0s.

**The margin rule had to be extended to cover this pass.** Consolidation
initially re-merged pairs clustering had refused as a coin flip between two
identities -- the two mechanisms contradicted each other, and two existing
tests caught it. `_merge_is_unambiguous` now takes the floor it should judge
"different identity" against, and consolidation applies it with a blocked-pair
set so a refused pair is not reselected forever.

### Non-face clusters

Person #4 in the consolidated montage is 185 detections of the show's
spinning logo. YuNet scores those graphics confidently, nothing downstream
asks whether a detection is a face, and they are self-similar enough to form
a large stable cluster. This is not a grouping bug and no threshold fixes it
-- it is the same crop-quality gap as the back-of-head detections, and it
argues for the quality gate rather than against consolidation.


---

## 7f. Non-face group filter

`test_3.mp4` produced a 185-detection "Person #4" spanning the whole episode
that was not a person: a mixture of backs of heads, hard profiles, and the
show's spinning logo. YuNet scores those confidently (0.72-0.95), nothing
downstream asks whether a detection is really a face, and -- the part that
makes it a *large* group rather than scattered noise -- they resemble each
other, because their landmarks fail in the same way. Degenerate detections
cluster into a stable phantom identity.

**Blur does not catch it.** That was the obvious first guess and it is wrong:
the phantom group's median Laplacian variance is 169, against 119-284 for
real people. The logo is perfectly sharp. What separates them is landmark
geometry, specifically eye separation as a fraction of box width:

| group | detections | median blur | eye span / width |
| --- | --- | --- | --- |
| #1 (real) | 516 | 284 | 0.350 |
| #2 (real) | 284 | 119 | 0.351 |
| #3 (real) | 191 | 172 | 0.306 |
| **#4 (phantom)** | **185** | **169** | **0.139** |
| #5-#20 (real) | 33-163 | 9-577 | 0.317-0.430 |

`eye_span_ratio` computes it from landmarks that already exist, so the check
is free.

**The test is per group, not per detection.** At the detection level the
signal does not separate: real people are also filmed in profile, and cutoffs
that removed most degenerate frames also removed 8-14% of genuine ones. Those
frames are harmless individually because tracking attaches them to a track
containing better frames. Only a whole cluster of them, with no good frame
anywhere, is not a person.

`DEFAULT_MIN_GROUP_EYE_SPAN` is 0.15. At that value, on test_3.mp4, the
phantom is the **only** group of 5+ detections removed; the costumed
character (whose mask distorts the landmarks) and a legitimate
profile-heavy cluster both survive. Raising it to 0.22 starts taking those
too, which is why it is set low rather than in the middle of the gap.

Rejected observations are reported as `unassigned`, not dropped. They were
detected and simply could not be attributed to anyone; silently discarding
them would misreport how much of the video the pipeline accounted for.
`IdentityGrouper.unassigned` is now a property combining the up-front
unreliable buffer with groups this filter rejected, since the second set only
exists once clustering has run.


---

## 7g. Minimum screen time per identity

The gallery still listed a long tail of identities holding one or two
detections. They are not wrong -- a face really was there -- but nobody
would pick them out of a gallery, and on `test_3.mp4` they were more than
half of all cards.

The setting could not simply be "a minimum number of detections", because a
detection count is not portable. The same 23s clip gives its largest
identity 5 detections at a 1.0s interval and 26 at 0.25s, so a fixed count
means four different things at four sampling rates. The sampling-invariant
quantity is screen time: detections x interval.

Screen time alone is not enough either, because significance is relative to
runtime. Three seconds is an eighth of a 23s clip and a rounding error in a
feature film. So `auto_min_detections()` requires the larger of an absolute
floor and a share of the runtime:

    required_seconds = max(3.0, 0.005 * duration)
    min_detections   = round(required_seconds / sample_interval)

| video | interval | required | min detections |
| --- | --- | --- | --- |
| test.mp4 (23s) | 1.0s | 3.0s | 3 |
| test.mp4 (23s) | 0.25s | 3.0s | 12 |
| test_3.mp4 (22.6min) | 1.0s | 6.8s | 7 |
| 90-minute film | 1.0s | 27.0s | 27 |

The 3-second floor is the anchor: on the 23s clip at a 1.0s interval it
reproduces exactly the "drop anything under 3 detections" rule that
prompted this, taking that video from 10 groups to 3.

`--min-detections N` overrides the derivation entirely, and `1` disables
the filter. The resolved value and its reasoning are printed on every run,
because a filter that silently hides identities is one people should be
told about rather than left to infer from a short gallery.

Rejected observations go to `unassigned`, consistent with the non-face
filter: they were detected, they are just not an identity worth offering.

The derivation lives in `app/main.py`, not in `IdentityGrouper`, which is
given a plain integer. The grouper deliberately knows nothing about video
duration or sampling rate, and this keeps it that way.


---

## 7h. Stage 0.3 notes (clip extraction)

### Re-encoding is mandatory, and only real footage showed it

The obvious implementation is `ffmpeg -ss ... -c copy`, which is near-instant.
Measuring keyframe spacing first (per section 4's rule about testing uncertain
assumptions) ruled it out:

| video | keyframe gap, median | max |
| --- | --- | --- |
| test.mp4 | 0.03s | 0.40s |
| test_3.mp4 | 2.67s | 7.84s |

A stream copy can only start at a keyframe. On `test_3.mp4` the median gap
(2.67s) is nearly the median appearance length (3.0s) and the worst gap
(7.84s) exceeds most segments entirely, so copied cuts would routinely open
seconds early on the wrong person -- fatal for a tool whose whole promise is
"only this person".

`test.mp4` is nearly all-intra and would have hidden this completely. An
approach validated only on the short clip would have looked flawless and
failed on the first real video. Segments are therefore re-encoded; only the
final join is a stream copy, which is safe because every segment was just
written with identical codec parameters.

Cut accuracy was verified rather than assumed: the reel's first frame differs
from the source at the requested timestamp by 0.494 mean absolute pixel value
(re-encode noise), against 61.5 for a deliberately wrong reference frame half
a second away.

### The cut list is not the appearance list

`build_appearance_intervals` answers a detection question. A watchable reel is
an editorial one, and on real footage they disagree sharply: the lead's
appearances come back as 153 intervals (at 1.0s sampling) with a median
duration of 3.0s and 38 of them under 2s. Cut verbatim that is a strobe, which
fails the stage-0.4 "final video watchable" criterion no matter how accurate
each individual cut is.

`merge_for_export` is therefore a separate, pure pass: pad for headroom, bridge
gaps too short to cut across, grow anything still under a minimum length, then
merge again -- growing can close a gap that was wide enough to keep a moment
earlier, and skipping that second pass yields overlapping segments and
duplicated footage in the join. It is deliberately free of I/O so the judgement
about what makes a watchable segment is testable without encoding a frame.

Defaults (bridge 1.5s, minimum 2.0s) take those 153 intervals to 97 segments
with nothing under 2s, for 26% more footage. The added footage is not purely
waste: a one-second cutaway held through preserves conversational context that
a hard cut destroys.

Note the ratio is much worse on sparse short clips -- on `test.mp4` five
intervals totalling 5.0s become three segments totalling 12.0s, because
growing one-second appearances to the two-second minimum dominates.

### Measured end-to-end

A full export of the lead from the 22-minute episode at 0.5s sampling: 1057
detections, 173 appearance intervals (527.0s on screen), 107 segments (662.2s),
encoded in 180.3s at 3.67x realtime with `h264_videotoolbox`, peak RSS 1.24 GB,
599s total. Output decoded end to end with no errors and drifted 0.15s across
107 joins. Of 16 frames sampled across the reel, 14 clearly showed the target,
including one in costume -- the consolidation pass from 7e visible in the
finished product. The other two were title/credit cards, i.e. detections on
faces printed in graphics.

Wall time is dominated by the detect/embed pass (~7 min), not the cutting
(~3 min); ArcFace is the target for any future speed work, not ffmpeg.


---

## 7i. Batched embedding, and where the time actually goes

Embedding was the pipeline's largest compute cost, so `FaceEmbedder.embed_batch`
now runs one forward pass per frame instead of one per face. The batched result
is bit-identical to the per-face path -- checked with an exact comparison, not a
tolerance, because these vectors feed cosine-similarity clustering and any drift
would move real cluster boundaries silently.

**The gain is smaller than a microbenchmark suggests, and the reason is worth
recording.** Throughput against synthetic batches:

| batch | ms/face | speedup |
| --- | --- | --- |
| 1 | 28.8 | 1.00x |
| 2 | 24.7 | 1.17x |
| 4 | 21.9 | 1.32x |
| 8 | 21.5 | 1.34x |
| 32 | 22.1 | 1.30x |

The batch size is not a free parameter: it is however many faces are in the
frame, and that is a property of the footage rather than of the sampling rate.
Sampling twice as often produces twice as many frames, not fuller ones --
measured over the same first 240s of `test_3.mp4`, a 1.0s interval gives 2.17
faces per frame and a 0.5s interval gives 2.16. So the operating point sits at
the shallow end of that curve at every sampling density. Measured A/B on the
same 3111 detections, embedding fell from 93.88s to 82.75s -- **1.13x**,
matching the batch-2 row rather than the batch-8 one. Quoting the plateau figure
as the expected gain would have overstated it by about 20%.

An earlier note here claimed 4.6 faces per frame at 0.5s and predicted a larger
gain at denser sampling. That number came from a scan script dividing one run's
detections by another run's frame count; the prediction was wrong for a reason
that is obvious once stated, which is why the corrected measurement is recorded
rather than quietly dropped. Re-running the whole pipeline at 0.5s confirms
there is no denser-sampling dividend: 6221 detections embedded in 164.91s is
26.5 ms each, against 26.6 ms for the 3111 detections at 1.0s. **1.13x is the
ceiling for this footage**, not a floor to improve on by sampling harder.

Batching is per frame rather than across frames deliberately. Frames arrive from
a generator holding one at a time (7d), and buffering several to fill a larger
batch would trade back the memory streaming was introduced to reclaim.

### The larger inefficiency is decode, not embedding

Measured separately on the 22-minute video at 0.5s sampling:

| phase | time |
| --- | --- |
| decode + sample | ~104s |
| detect + embed | ~315s |

Decoding produces 2712 sampled frames from 32508 source frames -- **12 frames
decoded for every one kept**. That is a structurally larger waste than the ~11s
batching recovers, and seeking rather than decode-and-discard is the obvious
next lever. It is not obviously free: PyAV seek lands on keyframes, whose
spacing on this footage is a median 2.67s (7h), so the accuracy of sampled
timestamps would need the same kind of measurement that ruled out stream-copy
cutting.

## 7j. One merge rule, two callers

`timeline.py` and `export.py` had both grown a routine for joining spans that
sit close together. They are asking different questions -- one re-merges after
padding has pushed neighbouring appearances into each other, the other bridges
gaps too short to cut across -- but it is the same operation at different
thresholds, and timeline's version was exactly the zero-gap case of export's.

It now lives once, as `timeline.merge_spans(spans, gap_seconds=0.0)`.
`export.py` already imported from `timeline.py`, so the dependency direction is
unchanged and no cycle is introduced.

## 7k. The desktop window (app/ui/app.py, app/ui/worker.py)

The UI is two modules on purpose, split along the line that matters:

- `app/ui/worker.py` runs the work. It imports no Tkinter, so it can be
  tested on a machine with no display -- and it is (tests/test_ui_worker.py).
- `app/ui/app.py` is windows and buttons. It decides nothing about faces.

Neither one re-implements any pipeline stage. `worker.scan` calls the same
`run_identity_pipeline` the CLI calls, and `ScanSettings` defaults to the same
constants `app/__main__.py` passes. That is asserted by a test, because the
failure it prevents is nasty: a UI that grouped a video differently from the
command line would look like a face-recognition bug rather than a defaults bug.
Making the pipeline entry point public (it was `_run_identity_pipeline`) was
the whole of the change needed on the pipeline side.

### Threading, and why the queue is not optional

Tk is not thread-safe, and the failure mode when you touch a widget from a
worker thread is a hang or a hard crash rather than an exception with a stack
trace pointing at the mistake. So the rule is absolute: the worker thread only
ever calls `queue.put`. The main thread drains that queue on an 80ms timer and
is the only thing that touches a widget.

The drain loop empties the whole queue per tick rather than taking one message,
so a burst of progress updates cannot fall behind the work producing them.

Scanning a 22-minute video takes minutes and encoding takes minutes more, so
neither could run inline without freezing the window for the duration.

### Cancellation without a cancellation feature

Neither the pipeline nor the exporter knows what cancellation is, and neither
needed to learn:

- **Scanning:** the UI wraps the frame iterator (`_tracked_frames`) and raises
  `Cancelled` from inside it. The exception unwinds through
  `run_identity_pipeline`'s existing `finally`, which closes the detector and
  embedder on the way out.
- **Exporting:** the UI raises `Cancelled` from the `on_segment` callback. That
  unwinds through `export_segments`' `TemporaryDirectory` context manager, so
  the half-finished segments are deleted and no partial file is left behind.

Wrapping the iterator, rather than passing a progress callback down into the
pipeline, is what keeps `app/main.py` unaware that a UI exists at all: it
consumes an iterator either way. `on_segment` is the one small addition to
existing code -- `export_segments` could already print progress, but printing
is no use to a progress bar.

### What the selection preview is for

Clicking a face runs `plan_export`, which is pure and takes microseconds, and
reports the cut count and reel length before any encoding starts. The
alternative -- press Export and find out in four minutes -- was worse for the
same reason the CLI prints its segment count before encoding: the editorial
merge (7-export) changes the answer substantially from the raw appearance
list, and being shown 107 cuts when you expected 173 is information you want
before the wait, not after.

### Verification

A UI cannot be verified by unit tests alone, so it was driven end to end
without a human: construct the real window, set a video, invoke the scan
button, pump `update()` until the worker finishes, click a card, invoke
export, and assert on what the widgets then say. On test.mp4 that produced 4
people (identical to `python -m app group --interval 0.5`: 49 detections, 20
unassigned, 4 groups), and an export of Person #1 wrote a real 12.02s reel
from 3 cuts -- matching the 12.0s the selection preview had predicted.
Cancelling mid-export left no file.

Screenshots were not part of this: `screencapture` needs macOS Screen
Recording permission, which is not something to grant on a user's behalf.

### What driving it found that reading it did not

Three bugs, none of which a unit test on the pipeline would have reached,
all found by pushing bad input and awkward orderings through the real
window:

1. **The reel was named after the wrong video.** The suggested filename read
   the path box rather than the scan, so editing the box after a scan and
   then clicking a face produced `no_faces-person-2.mp4` for a reel cut
   entirely from `test.mp4`. Export was correct throughout; only the name
   lied.
2. **The gallery accepted clicks mid-export.** The running job holds its own
   person, so the encode was fine -- but the label and the filename field
   both changed to describe someone the encode was not cutting.
3. **A corrupt file crashed instead of explaining.** `loader.py` caught
   `av.AVError`, which PyAV 18 no longer defines, so the except clause
   itself raised and a non-video surfaced as `module 'av' has no attribute
   'AVError'`. This one was not a UI bug at all -- the CLI had it too, and
   nothing in the suite covered a malformed file.

The common shape: every one of them was a case of the display disagreeing
with the work. That is the failure mode a UI has and a pipeline does not,
and it is why `tests/test_ui_app.py` now exists alongside the worker tests,
skipping rather than failing where there is no display.

Settings are now frozen for the duration of a job for the same reason. A
control that still moves while the job that captured it runs is claiming an
influence it does not have.

## 9. Packaging research: shipping this as a desktop app

Not built, only investigated. A trial PyInstaller build was made and run so
these notes describe measured behaviour rather than expectations; the build
artifacts were thrown away afterwards.

### The trial build

`pyinstaller --windowed --add-data <both models>` against an entry point that
calls `app.ui.app.launch` produced a working `FluxCutter.app` on the first
serious attempt. What it does and does not do, run from the bundle:

| check | result |
| --- | --- |
| launches | yes |
| finds its bundled models | yes, unmodified |
| full detect -> embed -> track -> group pass | yes: 23 detections, 3 identities |
| identical to `python -m app group --interval 1.0` | yes, exactly |
| exports a reel | **no** |
| passes Gatekeeper | **no** |

Bundle size **342 MB**, dominated by things that are already there:

| component | size |
| --- | --- |
| cv2 | 89 MB |
| ArcFace model | 166 MB |
| av (with its bundled FFmpeg) | 42 MB |
| PIL | 11 MB |
| numpy | 6.5 MB |
| Python + Tcl/Tk | ~10 MB |

### Model loading survives freezing by luck, and should not rely on it

`Path(__file__).resolve().parents[2] / "assets" / "models"` resolves inside the
frozen bundle to `Contents/Frameworks/assets/models`, which is exactly where
`--add-data "...:assets/models"` puts them. It works today, unmodified. It
works by coincidence of two layouts agreeing, so anything that shipped for real
should resolve against `sys._MEIPASS` when `sys.frozen` is set rather than
count on that continuing to line up.

### The one real blocker: ffmpeg is not on PATH under Finder

Export shells out to the `ffmpeg` binary. A Finder-launched .app inherits
`/usr/bin:/bin:/usr/sbin:/sbin`, not the shell's PATH, so Homebrew's ffmpeg is
invisible to it. Reproduced by running the bundle under `env -i` with that PATH:
the scan completed normally and the export failed with our own error message,
telling a double-clicking user to `brew install ffmpeg`.

This is the only thing standing between the trial build and a working app.

**The fix is already installed.** PyAV is a dependency, is already in the
bundle at 42 MB, and its vendored FFmpeg carries every encoder the exporter
asks for -- verified by constructing each one:

| encoder | available in PyAV |
| --- | --- |
| h264_videotoolbox | yes |
| libx264 | yes |
| aac | yes |

So the cutting could run in-process instead of shelling out, and the external
binary dependency would disappear rather than needing to be bundled. That is a
real piece of work (segment encode plus concatenation, currently ~140 lines of
subprocess calls) and it changes a component that is currently correct and
tested, so it wants its own measured comparison against the existing output
before replacing it -- but it is the direction, and it removes a dependency
rather than adding one.

Bundling the Homebrew ffmpeg binary is the other option and is worse: 420 KB
of binary that links 37 dylibs, all of which would need relocating into the
bundle and re-signing.

### Licensing, which needs an answer before any distribution

Measured, not resolved:

- The installed `av==18.0.0` wheel from PyPI **ships libx264 and libx265** in
  `av/.dylibs/`.
- Its bundled libavutil, asked directly through ctypes, reports its license as
  **"LGPL version 3 or later"**, and its configuration string contains
  `--enable-libx264 --enable-libx265 --enable-version3` but **not**
  `--enable-gpl`.
- Upstream FFmpeg documents that combining with libx264 requires
  `--enable-gpl`, and that the result is GPL.

Those two things do not obviously agree, and the answer decides whether a
distributed FluxCutter can be closed-source. Nothing here should be treated as
a legal conclusion -- it is a flag that the question is real and currently
unanswered. Homebrew's ffmpeg, for comparison, is unambiguously
`--enable-gpl`.

Note the licensing exposure exists **today**, through PyAV, independently of
whether the exporter keeps shelling out to a separate binary.

### Gatekeeper

The trial bundle is ad-hoc signed with no Team ID, and `spctl` rejects it: a
downloader would be told the app is damaged. Shipping needs a paid Apple
Developer account, a Developer ID certificate, hardened runtime, and
notarization.

Of the three packaging tools, **Briefcase** handles signing and notarization
as a built-in step, while PyInstaller and py2app leave it to be scripted. That
is the main axis worth deciding on, since the trial shows PyInstaller can
already build the thing -- the hard part is not the freeze, it is everything
Apple requires afterwards. Hardened runtime is also documented to break some
native-extension imports, which with cv2 + numpy + PyAV in the bundle is worth
testing early rather than at the end.

### If this is picked up

Roughly in dependency order:

1. Move export in-process onto PyAV, removing the ffmpeg binary dependency.
2. Resolve the x264 licensing question.
3. Resolve model paths through `sys._MEIPASS` when frozen.
4. Choose the packaging tool on notarization support, not build capability.
5. Test hardened runtime early, against the native extensions specifically.
6. Consider whether the 166 MB model ships in the bundle or downloads on first
   run -- it is half the download either way, and a first-run fetch needs a
   progress UI and a failure path that the app does not currently have.

## 9b. Docker, and deferring the big downloads

Two proposals, considered separately because they pull in opposite
directions: one is about where the app runs, the other about what it
carries.

### Docker is the right answer to a different question

Containerising the **CLI** is straightforward and worth doing. Containerising
the **window** is not, and the reasons are specific rather than stylistic:

- **Hardware encoding disappears.** videotoolbox is an Apple framework and
  cannot exist inside a Linux container. Measured on the 12s test reel,
  same footage, same segments:

  | encoder | encode time | throughput |
  | --- | --- | --- |
  | h264_videotoolbox | 4.7s | 2.54x realtime |
  | libx264 | 23.9s | 0.50x realtime |

  About **5x**, consistent with the 3.67x realtime the 22-minute run managed
  with videotoolbox. Containerising means every export takes five times
  longer, permanently, on the machine where the app is most likely to run.

- **A Linux container cannot show a Mac window.** It would need XQuartz and
  X11 forwarding -- slow, ugly, and something no one installs to use a video
  tool.

- **Video I/O crosses a VM boundary.** Docker Desktop on macOS is a Linux VM,
  and the source footage would be bind-mounted across it.

- **Docker is a bigger ask than the app.** Someone who wants to cut a reel
  will not install a container runtime first.

Where Docker genuinely helps:

1. **As a build environment**, not a runtime -- a reproducible container that
   produces the Linux binary, with the host toolchain out of the picture.
2. **For the headless CLI**, if FluxCutter is ever run as a batch or server
   job. `python -m app export ...` is already the right shape for that, and
   there libx264 is the only option anyway, so nothing is lost.

So: container for the CLI and for builds, native bundle for the window. These
are two deployment targets, not two ways of doing one.

### Deferring the model download is right, with three conditions

Fetching models on first use rather than bundling them is a good instinct.
It takes the macOS bundle from **342 MB to about 176 MB** -- the ArcFace model
alone is 166 MB of it.

Three things have to be true first, and one of them is a trap:

**1. The download source needs fixing first.** ArcFace ships inside
InsightFace's `buffalo_l` bundle. Measured `Content-Length` of that bundle:
**288,621,354 bytes** -- to extract a **174,383,860 byte** file. A naive
first-run fetch would make the user download *more* than bundling costs
them. It only pays off if the extracted model is hosted directly. YuNet has
no such problem: 229,738 bytes, fetched directly.

**2. Checksums are mandatory, not optional.** This project has already lost
an afternoon to a silently corrupt model -- an SFace file that arrived as
70 MB instead of 38 MB with 15,998,341 replacement characters in it, from a
text-mode round trip, and presented as five failing tests rather than as a
download error. A first-run downloader turns that from a one-off into
something every user can hit. Verify a known SHA-256 and delete on mismatch.

**3. It needs the UI it does not have.** A 166 MB download needs a progress
indication, a cancel, a retry, a disk-full path and an offline path. The
window currently assumes models are simply present. This is most of the work
of the feature; the downloading itself is the easy part.

Where they go: `~/Library/Application Support/FluxCutter/models` on macOS,
not next to the app, which may be read-only or in `/Applications`.

### Auto-installing ffmpeg on first use: no

The same instinct applied to the ffmpeg binary should be resisted. Running a
package manager on someone's machine from inside an app needs admin rights,
assumes a specific package manager, and looks exactly like what security
software is built to stop.

It is also unnecessary. PyAV is already a dependency, already in the bundle,
and already carries h264_videotoolbox, libx264 and aac (9). Moving the cut
in-process removes the dependency instead of automating its installation --
strictly better than either bundling ffmpeg or fetching it.

The rule this suggests: **defer data, never defer executables.** Models are
inert files that a checksum can validate. Binaries are not.

## 9c. Self-distribution, and Windows

### Not shipping through the App Store does not mean not signing

Notarization is *not* an App Store requirement. It is the requirement for
distributing outside it: macOS attaches a quarantine flag to anything
downloaded, and Gatekeeper refuses to open an unsigned or un-notarized
quarantined app, reporting it as damaged rather than as unsigned. The trial
bundle here was ad-hoc signed and `spctl` rejected it (9).

So self-distribution on macOS needs a Developer ID certificate and
notarization anyway. Windows is genuinely different: an unsigned .exe raises
a SmartScreen warning the user can click past, so unsigned self-distribution
is viable there in a way it is not on macOS.

### Executables cannot be cross-compiled

PyInstaller freezes the interpreter it is running on. A Windows .exe has to
be built on Windows. For a project with no Windows machine, that means CI --
a GitHub Actions matrix over `macos-latest` and `windows-latest` is the
normal answer, and the repository already lives on GitHub.

### What Windows changes about the app

- **No videotoolbox.** Windows falls back to libx264 unless the machine has
  nvenc/qsv/amf, so exports run about 5x slower than they do on Apple
  silicon (9b). Nothing to fix; it is what the hardware offers.
- **ffmpeg is even less likely to be present.** The PATH problem that blocks
  the macOS bundle (9) is worse on Windows, where users are unlikely to have
  ffmpeg installed at all. This raises the priority of moving the cut
  in-process onto PyAV from "the cleanest fix" to "the only sane one".
- **The encoder list had to stop being hardcoded.** The dropdown offered
  `h264_videotoolbox` unconditionally, which on Windows is an encoder that
  does not exist -- selectable, and failing only at encode time.
  `available_encoders()` now asks PyAV what this machine can actually
  construct and offers only that, best first, with libx264 as an
  unconditional floor. Platform detection was the wrong tool: whether nvenc
  works is a question about the hardware, not about `sys.platform`.
