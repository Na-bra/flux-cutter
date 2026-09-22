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

## 10. Models fetched on first use (`app/models.py`)

The two ONNX files are 174 MB and 230 KB. Keeping them in the repository was
fine while this only ever ran from a checkout; it stops being fine the moment
anything is distributed, where they are half the download and never change.

They are now fetched the first time something needs them -- at the moment the
detector or embedder is constructed, not at import and not at install -- and
`app/models.py` is the only place that knows where they live or how to get
them.

### The download source had to be fixed before the feature was worth having

ArcFace ships inside InsightFace's `buffalo_l` bundle: **288,621,354 bytes to
extract a 174,383,860 byte file**. A first-run fetch of the bundle would make
the user download 275 MB to keep 166 MB -- more than bundling costs them, so
the feature would have been a regression. It is mirrored standalone on Hugging
Face, where `content-length` matches this project's own copy exactly.

### Verification is the point, not a nicety

Every spec pins a SHA-256, checked after download, and the file is moved into
place only if it matches. This is not defensive habit: an earlier model in
this project arrived through a text-mode round trip at 70 MB instead of 38 MB
with 16 million replacement characters in it, and presented as five confusing
test errors rather than as a download problem (7). A downloader turns that
from a one-off into something every user can hit.

Pinning also answers the mirror question. A third-party mirror is a
supply-chain exposure; the hash means it is *this project's* copy of the model
that is authorised, not whatever that URL serves later. If the mirror changes,
verification fails and nothing loads.

### Atomicity is what makes retrying safe

The download goes to a `.part` file beside the destination -- same filesystem,
so the rename is atomic -- and is renamed only after the hash matches. An
interrupted download therefore leaves nothing that a later run could mistake
for a finished one, which is what allows the answer to a failed fetch to be
simply "run it again". The cleanup catches `BaseException`, because the UI
cancels by raising through the progress callback and a `KeyboardInterrupt`
must not leave a half file either.

### Where they go

A per-user data directory, not next to the app: a frozen `.app` in
`/Applications` may sit on a read-only volume, and writing into an application
bundle is wrong even when it is permitted. `FLUXCUTTER_MODEL_DIR` overrides
it. A copy in `assets/models/` still wins over the cache, so an existing
checkout keeps working and no one is made to re-download what they have.

### Two callers, two kinds of progress

A 166 MB download on a slow connection takes minutes, and silence for minutes
is indistinguishable from a hang:

- The **CLI** gets `ensure_model_cli`, which says what it is fetching and how
  big before it starts, then redraws a bar in place.
- The **window** cannot use stdout at all, so `worker.fetch_models` pulls the
  download forward to before the pipeline starts and reports it through the
  same queue as everything else. Left to the detector and embedder to trigger
  on construction, it would have happened several frames deep with nowhere to
  report to.

Pulling it forward has a second benefit: a first run no longer decodes two
minutes of video before discovering it has no model to embed the faces with.

### What is not tested

The suite never touches the network. That the pinned URLs still serve the
pinned hashes is a fact about the outside world rather than about this code,
and belongs in `python -m app models fetch`, not in a test that fails on a
train. What is tested is the part that fails quietly: rejection of a file that
is not what was expected, and that neither a rejected nor an interrupted
download leaves anything behind.

### Verified against the live mirror

The pinned ArcFace URL was fetched end to end through `download_model`:
174,383,860 bytes, sha256 matching the pin, in **1161 seconds** on a domestic
connection. That number is the argument for the progress UI rather than an
aside -- a first run can be a nineteen-minute wait, and nineteen minutes of
silence is indistinguishable from a hang.

An earlier attempt at the same URL is worth recording because it is the case
this design exists for. `curl` returned **135,783,125 bytes of the
174,383,860 byte file and exited 0** -- a 38 MB shortfall reported as
success. The hash caught it. The size check now catches it first and says
"stopped early ... running the same command again will retry it", because
"not the expected file" would have sent someone to distrust the mirror when
the actual fix was to retry the transfer. The mirror was fine; the transfer
was not.

## 11. scikit-learn was considered and measured, then declined

Recorded so it is not re-litigated from priors. Measured on the cached
`test_3.mp4` track embeddings -- 1936 tracks, 3111 observations, 512-dim:

| method | clusters | noise | time |
| --- | --- | --- | --- |
| current grouper (after filters) | 40 | -- | 13.88s |
| sklearn agglomerative, d=0.65 | 277 | 0 | 4.44s |
| sklearn DBSCAN, eps=0.5 | 62 | 266 | 0.20s |
| sklearn HDBSCAN, min_cluster_size=5 | 46 | 545 | 11.72s |

**HDBSCAN was the candidate worth testing and it lost.** Its appeal was
native noise labelling, which looked like a principled replacement for the
hand-built eye-span filter and the phantom identity it removes (7f). In
practice it is no faster than what exists and discards **545 of 1936 tracks,
28% of them** -- roughly three times the ~185 degenerate detections the
eye-span filter targets. Replacing a working heuristic with a dependency is
only justified if the dependency does it better; this does it worse.

**Agglomerative clustering is what `grouper.py` already implements**, so it
buys maintenance rather than quality, and costs the margin rule (7b), which
has no sklearn equivalent. Its 277 clusters against the pipeline's 40 is the
gap filled by the consolidation pass, the non-face filter and the minimum
screen time -- four pieces of work sklearn does not replace, wrapped around
the one piece it does.

**KMeans and SpectralClustering are the wrong shape.** They need
`n_clusters`, and how many people are in a video is the question, not an
input.

DBSCAN is the only genuinely interesting result: 0.20s against 13.88s, a 70x
saving. It is still not worth taking today, because clustering is ~7% of a
run that spends ~104s in decode and ~83s embedding. It becomes worth
revisiting if sampling ever gets dense enough for O(n^2) to dominate -- at
0.25s on a feature-length film, 1936 tracks becomes roughly 8000 and 13.88s
becomes minutes. The shape to try then is DBSCAN as a cheap pre-pass that
splits obvious groups, with the existing logic run inside each, rather than
a wholesale swap that would discard the tuning.

### The measurement that made the question worth asking

The comparison began from "grouping costs 0.04s, so speed is not a reason to
touch it". That figure was wrong, and wrong in a way worth recording:
`add_track` only buffers, and `_cluster()` runs lazily on the first access to
`.groups` -- which happened in `build_identity_gallery`, **after**
`grouping_time` had already been finalised. The timer was measuring the
buffering alone: 0.10s, against the 13.88s the clustering actually costs.

A number that low is not merely inaccurate, it is an argument-stopper: it
says clustering is free and no one need look further. `run_identity_pipeline`
now calls `grouper.finish()` inside the timed section so the reported figure
means what it claims.

## 12. Cutting in-process (`app/video/cutter.py`)

Export no longer shells out to `ffmpeg`. The packaging trial (9) found that
a Finder-launched .app inherits `/usr/bin:/bin:/usr/sbin:/sbin`, so a
Homebrew ffmpeg is invisible to it: the frozen app scanned perfectly and
then could not cut. PyAV was already a dependency, already ships FFmpeg
inside its wheel, and already carries every encoder in use, so moving the
work in-process removed a dependency instead of bundling one.

It is also structurally simpler. The subprocess version wrote one temporary
file per segment and joined them with the concat demuxer, because that is
how a command-line tool must do it. Holding the output container open means
each segment's frames encode straight into the finished reel: no temporary
files, no second pass, no assumption that segments share codec parameters.

### Three bugs found by measuring rather than reading

None of these are visible in the code, and two are invisible in playback.

**Frames lost at every segment boundary.** `container.decode(video, audio)`
yields both streams interleaved, and audio runs ahead of video. Breaking the
loop on the first frame past the segment's end therefore ended on an *audio*
frame and discarded the video still to come: **7 frames per segment, 339 of
an expected 360** across a three-segment reel. The output looked fine. Each
stream is now finished independently and the loop stops only once both are
done. `tests/test_cutter.py` asserts the 360.

**Hand-computed timestamps do not mux.** Rewriting each frame's PTS against
a running clock -- the obvious way to make three disjoint spans into one
continuous timeline -- failed every `mux()` with EINVAL. An encoder's
`time_base` is not what the stream reports before it has been opened, and
arithmetic against the wrong base produces timestamps the muxer rejects.

**`pts = None` muxes happily and writes a broken file.** Letting the encoder
assign its own timestamps is the usual advice and it appears to work: the
file plays. Probed, its *video* stream has a duration of `0.033333` -- one
frame -- and an `avg_frame_rate` of `10800/1`. The audio stream carries a
sane duration, which is what hides it. Each stream now counts its own
output, video by frame number and audio by sample number, stamped against an
explicit 90kHz base. The result probes as `30/1` and exactly `12.000000`
seconds for three four-second segments, which is *better* than the
subprocess version managed (`27000/901`, `12.013334`).

### Verified

Frame-for-frame against the path it replaces: 360 video frames from the same
three segments, both ways, and slightly faster (3.9s against 4.6s). Then end
to end where it actually matters -- inside the frozen bundle, under
`env -i` with a Finder-like PATH and no ffmpeg anywhere: 4 people found, a
three-cut 11.9s reel written, decoding cleanly.

## 13. Surviving a video that moves (`app/video/source.py`)

A scan takes minutes and leaves a result that is only half self-contained:
the gallery is thumbnails in memory, but cutting the reel has to read the
footage a second time. Between those two moments the user is free to
rename the file, drag it to another folder, or unplug the drive it lives
on -- and doing any of that turned a finished scan into a dialog reading
*"Could not export: Could not open /the/old/path"*, with the only remedy
being to scan again.

The question that started this was whether to hold the video in RAM. The
answer is no, and the measurement is the reason.

### RAM buys nothing that a descriptor does not buy more cheaply

Measured on `test_3.mp4` (815 MB), seeking to 10:00 and decoding 60 frames:

| holding it as | decode | process RSS | survives a move |
| --- | --- | --- | --- |
| a path | 0.17s | 38 MB | no |
| an open descriptor | 0.17s | 65 MB | yes |
| bytes in RAM | 0.17s | 767 MB | yes |

Decode speed is identical three ways, because the OS page cache is already
doing the job a RAM copy would do. So the only thing a RAM copy buys is
detachment from the path -- and an open descriptor buys exactly the same
detachment for 27 MB instead of 767 MB. On a 4 GB feature film the RAM
version is not merely wasteful, it is a machine the app cannot run on.

### Two defences, because they fail in different situations

**A held descriptor.** On macOS and Linux a descriptor refers to the inode,
not the name. Verified by unlinking the file so that no path on the machine
reached it, then opening, seeking and decoding through the descriptor
anyway. `open()` hands out `os.dup()` copies rather than the descriptor
itself, so two readers cannot seek each other sideways.

**Relocation.** A descriptor dies with the process, so it does nothing for a
video moved while the app was closed, and -- see below -- it is not held at
all on Windows. `VideoSource.relocate` points the source at the file's new
home and the existing scan carries on. The UI checks reachability *before*
starting the encode, so a missing video costs a dialog rather than a
progress bar that runs to "Preparing cuts..." and then stops.

Relocation guards against picking the wrong file by comparing byte size.
That is a guard, not a proof: two different videos of exactly equal length
would pass it. It is worth having because the realistic mistake is choosing
a neighbouring clip out of the same folder, which the check catches
immediately.

### Windows deliberately gets only the second

CPython's `open()` on Windows does not request `FILE_SHARE_DELETE`, so a
held descriptor would stop the user renaming or deleting their own video
for as long as FluxCutter had it open. That trades a failed export for a
blocked file operation, which is the worse bargain in an app that sits open
all afternoon. `KEEPS_HANDLES` is therefore false there and Windows leans
on `relocate`.

This is reasoning from how the Windows CRT opens files, not from a test --
there is no Windows machine here. Opening with `FILE_SHARE_DELETE` through
`ctypes` and `msvcrt.open_osfhandle` would give Windows the descriptor too,
and is the obvious follow-up for whoever first runs this on Windows.

### Verified

Both paths, end to end, with a real scan in between:

- **With a descriptor:** scanned, then renamed the file *and* moved it to
  another directory *and* deleted the directory it came from. Export ran
  without noticing -- 3 cuts, 360 frames, `avg_frame_rate=30/1`,
  `duration=12.000000`.
- **Without one** (`KEEPS_HANDLES` forced false, standing in for Windows):
  the moved file was correctly reported unavailable, a same-folder decoy
  was refused on size, `relocate` to the real file succeeded, and the
  export produced the identical 360 frames and 12.000000s.

23 tests cover it, most of which need no footage: `VideoSource` validates
and holds a descriptor without decoding, so a file with the right extension
proves the file-system half in milliseconds.

## 14. The app icon (`packaging/make_icon.py`)

The bundle shipped PyInstaller's own logo, because the spec passed
`icon=None` and that is the fallback. Everything else about the packaging
said FluxCutter -- `CFBundleName`, the Dock label, the window title -- and
the icon said PyInstaller.

### Drawn in code, not committed as artwork

The shape is a dozen numbers, so `make_icon.py` holds them and writes all
three formats. That means the icon can be recoloured or re-cut at any size
without hunting for an original nobody can edit, and the reason for every
number is a comment rather than a memory. The generated `.png`, `.ico` and
`.icns` are committed anyway, because `.icns` can only be produced on macOS
(`iconutil`) and the Windows CI runner has to build without regenerating.

### The design constraint is 16x16

Focus brackets around a play triangle, split by the cut: brackets for
"find someone", the triangle for "video", the split for "cut". Interior
detail is pointless at Dock size, so what matters is the silhouette.

The first attempt failed exactly there. A bracket stroke of 38/1024 is 0.6
of a pixel at 16px, and the brackets dissolved into the gradient -- 4 white
pixels out of 256. Thickening the stroke to 64/1024 and shortening the arms
gives 16, and the corners read. `tests/test_icon.py` keeps that number
above 12, which sits between the two measurements: a guard against thinning
the artwork again, not a claim that 12 is where legibility begins.

### Two places the icon has to be installed

The bundle's icon (`CFBundleIconFile`, and the `.ico` compiled into the
Windows `.exe`) covers the Dock, the Finder and the taskbar. It does not
cover the window: Tk draws its own title-bar and taskbar-button icon, and
given nothing it draws its default feather. So `app.py` also calls
`iconphoto` with the bundled `.png`, resolved through `sys._MEIPASS` when
frozen and from the checkout otherwise. It is cosmetic, so a missing or
unreadable file is swallowed rather than allowed to fail a launch.

### Verified

Built and inspected: `CFBundleIconFile => "icon.icns"`, a 422 KB `.icns` in
`Contents/Resources` carrying every size from 16 to 1024, and `icon.png`
bundled where the frozen lookup finds it -- confirmed by pointing
`_icon_file` at the real `Contents/Frameworks` and watching it resolve
through to `Contents/Resources/icon.png`. The bundle then launched clean
under `env -i` with a Finder-like PATH.

## 15. Identity accuracy: what was actually wrong (`app/faces/grouper.py`)

The brief was to raise identity accuracy and cut false merges. The
investigation changed what "false merge" even meant here, so the measurement
comes first.

### Ground truth, without labelling anything by hand

Two facts about the footage label a large part of the problem for free:

- Two detections **in one track** are the same person -- spatial continuity
  proves it.
- Two detections **in one frame** are different people -- nobody is in two
  places at once.

That gave 2407 same-person and 4060 different-person pairs on `test_3.mp4`
with no human judgement, and it is what every number below rests on.

    same person        p5 0.381   median 0.704   p95 0.887
    different people   p95 0.165   p99 0.321     max 0.664

At the 0.35 floor, 0.74% of different-person pairs sit above it and 3.61% of
same-person pairs below. The embedding and its preprocessing are not the
limiting factor.

### The blur gate, which 7c predicted and the data refuted

7c proposed a Laplacian-variance gate as the next lever. The measurement
reproduces its headline figure exactly -- 23.7% of detections score under 40
-- and then disagrees with the conclusion:

| min sharpness of the pair | different-person median | same-person median |
| --- | --- | --- |
| < 10 | 0.030 | 0.701 |
| 10-20 | 0.029 | 0.704 |
| 20-40 | 0.027 | 0.658 |
| >= 100 | 0.020 | 0.719 |

Blur neither makes different people look alike nor stops same-person matches.
And the detections inside the actual false merges are not blurred at all:
median sharpness 90, median confidence 0.917, median box 74px, **none** below
the blur floor, against 13.4% of all detections. A blur gate would have
removed 13% of the footage and not one error. It was not implemented.

Visual inspection agreed: group 13 (47 detections, median sharpness 8.7) and
group 23 (26 detections, nearly all profile) are both clean single
identities. Gating on blur would have destroyed them.

### What was actually wrong: nothing enforced that two faces are two people

Counting groups that held two non-overlapping faces from a single frame
found **11** on the baseline. Those are not near-misses; they are proof.

So the rule is now enforced rather than hoped for. Units that share a frame
have their similarity struck to -inf before clustering starts, and a merged
cluster inherits both halves' conflicts, so the rule cannot be escaped by
merging through a third party. It costs 3 ms to build the matrix for 1627
units and 2.6 MB to hold it.

### The correction that mattered: a shared frame is not always two people

Four of those 11 were not errors. Pulling the frames up and *looking* at them
showed t=336s and t=368s are the title sequence, which tiles clips of the
**same** actor side by side, and t=490s is YuNet finding faces in photographs
stuck to a fridge. Enforcing the rule blindly broke correct groups: it moved
28 of group 1's 516 detections into a duplicate identity.

They are separable, because a split screen does not look like two people.
Only 6 of 4060 same-frame pairs reach 0.50 similarity, four of them the title
sequence. Above that ceiling the shared frame is read as one person shown
twice. The remaining 2 misreads are between tiny background extras.

### Multiple prototypes per identity: measured, and declined

Matching on the best of k=3 prototypes rather than one centroid does separate
the extremes better (best free pair 0.525 against 0.405). It also raises the
score of pairs that are provably different people: groups 3 and 8 -- a
curly-haired boy and a dark-haired teenager, confirmed different by eye --
went from 0.259 to 0.397, and 23 and 28 from 0.221 to 0.388. Best-of-k picks
the most flattering corner of each identity, which is how a false merge gets
in. Not implemented.

### Consolidation was set far above anything that fires

0.50 was fitted to three same-actor pairs visible among the 30 largest
groups. Measured across all 780 pairs it never fires at all: consolidation is
a no-op anywhere from 0.41 to 1.0. With the co-occurrence labels the picture
is unambiguous -- provably-different pairs top out at 0.334, and the two
confirmed missed merges sit at 0.400 and 0.405. **0.375** sits in that gap.

Lower is tempting and unevidenced. Below 0.334 there are labelled
different-person pairs, and co-occurrence only protects pairs that happen to
share a frame. Note also that the false-merge count *becomes circular* once
the constraint is enforced -- it is exactly what the rule forbids -- so it
cannot be used to justify going lower.

### Results on `test_3.mp4` (1356 frames, 3111 detections, 1.0s sampling)

| | before | after |
| --- | --- | --- |
| Identity groups | 40 | 39 |
| **False merges** (two people in one group) | **7** | **0** |
| Split-screen frames correctly kept together | 4 | 4 |
| **Unmerged pairs still scoring >= 0.375** | **2** | **0** |
| Detections assigned to an identity | 2326 | 2394 |
| Unassigned | 753 | 685 |

Every one of the top 6 groups was inspected as a contact sheet and holds one
person, including the hard cases the pipeline exists for: group 1 now spans
Henry unmasked, in the Kid Danger mask, and in red face paint.

### The other two clips

`test.mp4` improves for the second reason rather than the first. It has too
few simultaneous faces to exercise the co-occurrence rule, but the
consolidation change is decisive:

| | before | after |
| --- | --- | --- |
| Identity groups | 4 | 2 |
| Unassigned | 20 | 14 |

The clip has exactly two actors. Before, each was split in two -- a lit group
and a dark/profile group -- and 0.375 merges each pair back without merging
the two actors together. Inspected: group 1 is entirely one actor across both
lighting setups, group 2 entirely the other.

`test_2.MOV` is unchanged at 2 groups / 6 unassigned, having neither
simultaneous faces nor split identities to fix.

An earlier draft of this section claimed `test.mp4` was unchanged. That was
the harness bug below: it had been measured at the old threshold.

### Cost

Sharpness is 121 ms across all 3111 detections. The constraint adds 1.78s to
the grouping stage (4.08s to 5.85s), of which 3 ms is the matrix and the rest
is the extra merge attempts a blocked pair causes. About +0.9% on a 205s run.
Wall-clock figures from the later runs are not comparable -- an unrelated
video transcode was saturating the machine -- which is why the cost is quoted
from direct measurement rather than from run totals.

### A harness bug worth recording

The first three "after" runs measured nothing, because the measurement script
restated the pipeline's defaults and still said `consolidation_threshold=0.50`.
The threshold change appeared to do nothing, and a plausible story was
constructed for why. The script now reads the constants from the module. Any
harness that repeats a value the code already owns will eventually measure the
wrong build.

## 16. Why macOS called the app "Python" (`app/ui/macos.py`)

The window title always said FluxCutter; the menu bar and Dock said Python.
Packaging 14 assumed that was cosmetic and development-only, on the grounds
that the frozen `.app` registers correctly with LaunchServices. That was true
and beside the point -- the app is run from a checkout far more often than it
is run frozen, and it is the same app.

macOS takes the name from `CFBundleName` on `NSBundle.mainBundle`, and the
reason a checkout gets "Python" is not the obvious one. A framework build of
CPython -- Homebrew's is one -- re-execs GUI processes through a stub
application inside the framework, so the main bundle is:

    .../Python.framework/Versions/3.12/Resources/Python.app

whose CFBundleName is, reasonably enough, "Python". Every Tk program run from
a framework Python is called Python in the menu bar.

That bundle's info dictionary turns out to be an `__NSDictionaryM` -- mutable
-- so the name can be written straight into it before Tk starts. Tk reads it
once, while building the menu.

An earlier attempt at this was abandoned for the right reason and with the
wrong conclusion: it reported success without checking anything had changed,
and "it does not raise" is not evidence. `set_application_name` now reads the
key back and returns True only if the new value is there:

    infoDictionary class : __NSDictionaryM
    before               : 'Python'
    after                : 'FluxCutter'

### What is and is not verified

The value Tk reads is verified, through the real `FluxCutterApp()`
construction path. The menu bar itself is not: reading it needs macOS
Accessibility permission, which is not something to switch on for a test run
or on someone's behalf. So the claim here is "Tk is handed the right name",
one step short of "the pixels say FluxCutter".

Everything in the module is cosmetic and every failure path returns False, so
a different Objective-C runtime, an immutable dictionary on some other Python
build, or a future macOS that stops handing out the real dictionary all leave
the app called Python and working, which is where it started.

## 17. Animation mode (`app/modes.py`, `app/faces/anime.py`)

FluxCutter now has two pipelines, chosen by the user. The live-action one is
untouched: same YuNet, same ArcFace, same tracker, same thresholds, and the
regression run on `test.mp4` is identical to before (49 detections, 33
tracks, 2 identities, 14 unassigned).

### Why a second pipeline at all

Measured, not assumed. On `animation.mp4` (a 7-minute Ben 10 episode) the
live-action detector finds **0.26 faces per sampled frame**, and the montage
shows most of them are not characters -- it misses the plain cartoon face in
the second sampled frame entirely. The animation detector finds **0.45**, and
they are the actual characters.

### The models are not the ones the brief named, and that matters

The brief specified the `video-to-faces` stack: a Faster R-CNN anime detector
and ViT-B16 for embeddings. Only half of that is reachable.

- **ViT-B16 and ViT-L16 weights are gone.** Both are hosted on Google Drive,
  and both ids return a hard `404` with Google's own error page -- not a
  consent interstitial, checked with and without the confirm-token flow.
  `arkel23/animesion` publishes no GitHub releases and no mirror was found.
  A default that cannot be downloaded is not a default.
- **The Faster R-CNN detector is reachable** (158 MB, from a real GitHub
  release) but needs PyTorch, which is **529 MB installed** here.

So the substitutes were chosen on availability and weight, and both are ONNX:

| role | model | size | licence |
| --- | --- | --- | --- |
| detection | `deepghs/anime_face_detection` v1.1_s | 45 MB | MIT |
| embedding | `deepghs/ccip_onnx` caformer-24 | 150 MB | OpenRAIL-M |

CCIP is *Contrastive Character Image Pretraining* -- an anime **character**
embedding rather than a face embedding, which is why the crop it is given
includes some hair and costume (on this footage, hair colour is often what
separates two characters).

Neither loads in `cv2.dnn`: the detector's graph trips OpenCV's ONNX importer
on a Concat node. They run under **onnxruntime**, which is 80 MB installed
against PyTorch's 529 MB, and which is an *optional* dependency --
`requirements-animation.txt`, not `requirements.txt`. Nothing in live-action
mode imports it, and `app/modes.availability` reports the mode unusable with
the exact `pip install` line rather than failing at the first frame.

### The thresholds do not transfer, and this is the important part

CCIP similarity is not on ArcFace's scale, and not merely shifted -- the two
distributions have different shapes. Measured on 20 hand-labelled character
crops from `animation.mp4` (171 pairs):

| | same identity | different identity |
| --- | --- | --- |
| **CCIP** (animation) | min 0.685, p5 0.745 | median **0.567**, p95 0.763 |
| **ArcFace** (live) | p5 0.381 | p99 0.321, median 0.03 |

Two *different* anime characters sit at 0.567 where two different people sit
at 0.03. Running animated footage through the live-action floor of 0.35 would
put the entire cast in one group. So every mode owns its own detection
settings, grouping thresholds and tracker contradiction floor, and changing
one cannot affect the other.

The animation numbers are **provisional**. There is real overlap here that
live action does not have -- at 0.75, 6.6% of different-character pairs merge
and 8.2% of same-character pairs split -- and 20 crops is a far smaller
sample than the live-action figures rest on.

### No landmarks, and none invented

The animation detector returns boxes only. `landmarks` stays `None` rather
than being fabricated to keep the dataclass tidy: ArcFace's alignment is
built on those five points, and feeding it invented ones would produce
confident nonsense. It is also why the two embedders cannot be swapped
between modes.

### Embeddings cannot be mixed

Every embedding carries the id of the model that made it, and
`IdentityGrouper` raises `MixedEmbeddingSpaces` if two named spaces reach one
grouper. This is a hard error rather than a threshold problem because the
comparison *works*: the cosine similarity of an ArcFace vector and a CCIP
vector is a perfectly well-formed float, and nothing downstream would notice.

### What the real video produced

105 frames at a 4s interval, 41 detections, 28 tracks, **7 character groups**,
68.7s total. Inspected card by card:

- **Ben** (17 detections, 88s-332s), **Gwen** (14, 164s-352s) and
  **Grandpa Max** (6, 24s-308s) are each one character, correctly separated,
  across profiles, expressions and distances.
- One legitimate one-off villain face.
- **Two false positives**: a patch of smoke (confidence 0.44) and a
  **buffalo** (0.72). Animal and background faces are exactly the animation
  failure mode to expect, and neither confidence nor size separates them here.
- **One missed match**: a wide shot of Ben seeded its own card instead of
  joining his.

### Cost, measured on a quiet machine

| | live action | animation |
| --- | --- | --- |
| detector load | 63 ms | 69 ms |
| embedder load | 89 ms | 226 ms |
| detection | 22 ms/frame | 209 ms/frame |
| embedding | 117 ms/face | 412 ms/face |

Animation is roughly ten times slower to detect and three times slower to
embed. That is the model, not the plumbing, and it is reported rather than
hidden -- CPU is the only supported target for both modes, and nothing here
silently substitutes a smaller model to make the number look better.

## 18. Sampling faster: what the roadmap expected, and what measured

The roadmap named seeking as the next lever, on the reasoning that decode
throws away 12 of every 13 frames it touches. Measured, that reasoning does
not survive: **seeking is the slowest thing in this comparison, not the
fastest.**

Over the first 180s of `test_3.mp4` (720p, 24 fps) at a 0.5s interval:

| strategy | time | frames decoded | timestamp error (median / max) |
| --- | --- | --- | --- |
| sequential, threaded (the baseline) | 3.44s | 4316 | 0.02s / 0.04s |
| sequential, one thread | 14.74s | 4316 | 0.02s / 0.04s |
| **`skip_frame=NONREF`** | **2.97s** | **2287** | 0.04s / 0.08s |
| `skip_frame=BIDIR` | 6.11s | 1248 | 0.07s / 0.16s |
| seek per sample | 17.95s | 16223 | 0.02s / 0.04s |
| seek per GOP | 11.94s | 4073 | 0.02s / 0.04s |
| keyframes only | 0.87s | 61 | 79.89s / 146.93s |

### Why seeking loses, both ways round

A seek lands on the previous keyframe and decodes forward from it, so it
cannot read one frame without reading the GOP in front of it. Keyframes on
this footage sit a median 2.38s apart (mean 2.82s, max 10.43s) while
sampling wants a frame every 0.5s, so seeking per sample decodes *more*
than reading straight through -- 16223 frames against 4316, and 5.2x the
time.

Seeking once per GOP instead of once per sample fixes the redundancy (4073
frames decoded, fewer than the baseline) and is **still 3.5x slower**. That
is the result worth keeping: the cost is not the frames, it is the seek.
Each one flushes the decoder, and a flushed decoder gives up the threading
that made sequential decode 4.3x faster in the first place. Seeking and
multi-core decoding are in direct competition, and multi-core wins.

Seeking would only pay at intervals sparser than the GOP spacing, which is
the opposite of what identity grouping wants -- denser sampling is what
gives the tracker consecutive frames to link.

### What did pay

`skip_frame = NONREF` asks the decoder not to fully reconstruct frames that
nothing else is predicted from. Sampling reads one frame in 12 and discards
the rest, so those frames are decoded purely to be thrown away.

Consistent across footage, best of three runs:

| | baseline | NONREF | |
| --- | --- | --- | --- |
| `test_3.mp4` 720p @ 0.5s | 3.46s | 3.00s | -13% |
| `test_3.mp4` 720p @ 1.0s | 3.35s | 2.95s | -12% |
| `animation.mp4` 1080p @ 0.5s | 5.27s | 4.52s | -14% |
| `test_2.MOV` @ 0.5s | 1.28s | 1.21s | -5% |

On all-intra footage such as `test.mp4` (keyframes 0.03s apart) there are
no non-reference frames, so it is exactly a no-op -- the sampled timestamps
come back identical, which is asserted by a test.

### The cost, and why it was checked before shipping

A sample can land on the next reference frame rather than the exact one.
That moved sampled timestamps by a median 0.035s and at most 0.083s: two
frames at 24 fps, against a sampling interval 500 to 1000 times larger.

Cheap to say, and not enough on its own -- different frames mean different
detections, which means different tracks, which can mean different
identities. A 180s slice suggested it might matter (13 identities against
11). Over the whole episode it does not:

| `test_3.mp4`, 1.0s sampling | baseline | NONREF |
| --- | --- | --- |
| frames | 1356 | 1356 |
| detections | 3111 | 3110 |
| tracks | 1627 | 1625 |
| identities | 39 | 38 |
| **provable false merges** (15) | **2 groups / 4 pairs** | **2 groups / 4 pairs** |
| whole-scan wall clock | 55.3s | 52.4s |

The error metric is section 15's, and it needs no labelling: two
non-overlapping faces in one frame are two people, so a group holding both
is wrong. It is unchanged. One detection in 3111 differs.

So the saving is ~13% of decode and ~5% of a whole scan, for a change that
the accuracy measurement cannot distinguish from noise. It is on by
default, and `extract_frames(..., skip_nonreference=False)` samples the
exact frames on the schedule for anything that needs them.

The roadmap item is closed, in the sense that matters: the lever it named
was measured and is not there.

## 19. Choosing a person with a photograph (`app/faces/reference.py`)

The gallery asks a question that can only be answered after a scan: which
of these forty faces did you mean? When the user already knows who they
want, a photo answers it up front -- embed the picture once, score the
scan's identities against it, and the montage never has to be looked at.

Built on the pieces that already existed rather than a second matching
path: the mode's own detector and embedder, the grouper's cosine
similarity, and the grouper's own per-mode floor for "these are the same
person". A reference face is one more embedding in the space the scan is
already working in.

### Scored against the centroid, not the best frame

Each identity is compared on its `representative_embedding` -- the mean
over every observation in the group. That is the same reasoning that made
track averaging worth doing (7): a single frame's embedding on hard footage
is noisy enough that same-person pairs fall below the floor, and matching a
photo against one lucky frame is matching against noise.

### What it measures on the real footage

38 identities from `test_3.mp4` at 1.0s sampling, one reference photo
written per identity from its own representative crop and read back through
the full JPEG-decode-detect-embed path:

    correct                    37/38
    margin over runner-up      min 0.305   median 0.623
    wrong-identity scores      max 0.297   p95 0.147   median 0.024

The gap is wide: the right identity never scored below 0.543, and no wrong
one reached 0.30. The default floor is the mode's own grouping threshold
(0.35 for live action), which lands inside that gap rather than at a number
invented for this feature.

This is an in-domain floor, not a field test -- the crops come from frames
the scan itself saw, so a real photograph, shot on a different camera under
different light, will score lower. It establishes that the mechanism works
and what the score distribution looks like; it does not establish a
threshold for arbitrary photographs. `--reference-threshold` exists for
that reason.

The one failure is worth recording: identity 32's representative crop was
tight enough that YuNet found no face in it when read back as a photo. A
crop of a face is not always a photo of a face, and the error says so
rather than matching on nothing.

### Two things it refuses to do quietly

Both produce a confidently wrong reel, which is worse than a refusal after
minutes of encoding:

- **Matching across embedding spaces.** An ArcFace vector and a CCIP vector
  have a cosine similarity and it means nothing. A photo embedded in one
  mode cannot select an identity grouped in the other.
- **Choosing between two plausible people.** When the best and second-best
  identities sit within 0.05 of each other, that is reported as ambiguous.
  It is evidence the scan split one person in two, or that the photo is not
  clearly either of them, and both are worth stopping for.

The photograph is also read *before* the scan starts. Detecting and
embedding one face takes a second or two, and a photo with nobody in it is
much better discovered then than after seven minutes of scanning have
earned the right to fail. The same reasoning moved "you did not name a
person at all" to parse time.

## 20. A folder at a time (`run_batch`)

One photo, many videos, one reel each.

### Cross-video identity comes free, and that is the design

The obvious way to run a season is to scan episode one, take the chosen
person's centroid, and carry it forward. That centroid then has to survive a
change of lighting, camera, costume and hairstyle between episodes, and
every hop compounds the last one's error.

Matching every episode against the *same photograph* removes the problem
rather than solving it. There is no notion of "the same person in video A
and video B" anywhere in the batch: each video is matched independently
against a vector that is identical every time, computed once before the
first scan. A test asserts that identity -- literally, that the same object
reaches every video.

### Nothing aborts the run

A season is twenty scans of several minutes each. Losing the other nineteen
because episode three is unreadable, or holds nobody who matches, or was
never a video, is the one failure that would make this unusable. Every
video's outcome is recorded and the run continues:

    --- Batch summary ---
      episode-slice.mp4 -> episode-slice-reel.mp4 (26.7s)
      test.mp4 -- No identity in this video matches person-01.jpg. The closest
                  scored 0.08, under the 0.35 floor.
      test_2.MOV -- No identity in this video matches person-01.jpg. The closest
                    scored 0.04, under the 0.35 floor.

    1/3 produced a reel, 26.7s of footage in total.

That run is the verification: a person present in one video and absent from
two others, matched at 0.81 where they appear and 0.08 and 0.04 where they
do not, with the reel decoding back to 640 frames over 26.69s.

This is why selection failures raise rather than exit. `_resolve_selection`
and `cut_segments` used to `sys.exit(1)` from inside `run_export`, which is
correct for one video and fatal for twenty; they now raise, and each caller
decides whether that is a failed command or a skipped episode.

## 21. Keeping a scan (`app/scans.py`)

A scan is the expensive thing this app does, and until now every one was
thrown away -- when the window closed, when a command returned. The
documented workflow made it worse rather than better: `group` to see the
montage, then `export --select-index 0`, which scanned the same footage a
second time to reach the same identities.

    test_3.mp4 at a 1.0s interval
      scan        55.3s
      save         0.32s
      load         0.05s
      on disk      6.3 MB   (38 groups, 3110 detections)

The export that follows a `group` now spends its time encoding rather than
rediscovering who is in the video.

### Freshness is a question about the key

There is no staleness check, because there is nothing to check. The key
covers the video's identity (path, size, modification time) and every
setting that can change what the scan produces, so a hit means the same
scan would have produced the same answer. Getting that wrong would be
worse than having no cache: it would serve one scan's answer to another
scan's question.

Every argument to `cache_key` is required and named. A caller that forgets
one gets a TypeError rather than a key that quietly differs from the other
caller's -- which is exactly what would stop the window and the command
line sharing an entry for the same work.

A **scan-format number** is part of the key, and it moves only when
detection, embedding, tracking or grouping would give a different answer.
Thresholds here have been retuned against real footage more than once (15,
17), and a scan from before such a change is not merely old, it is wrong.

This was the app's own version until 1.9.4, which was too blunt by far --
see section 28.

The video is identified by size and modification time rather than by
hashing its contents. Hashing an 815 MB file to avoid re-reading it is a
poor trade, and an edit preserving both would have to be deliberate.

### What is stored, and what is deliberately not

The identities, not the footage: every observation's embedding, box,
landmarks and timestamp, plus **one** representative crop per person for
the gallery to draw. Nothing downstream reads the other crops -- only the
representative becomes a thumbnail -- and dropping them is what keeps an
episode's scan to 6 MB rather than hundreds.

Verified by round-tripping a real scan: 38 groups, every embedding, box,
landmark, timestamp and frame index identical, and the rebuilt gallery
producing the same 38 cards with byte-identical thumbnails.

### Nothing here raises

Every caller's fallback is to do the work, so a corrupt or half-written
entry has to cost a rescan rather than a failed scan -- turning a bad
cache file into a broken pipeline would make the feature a liability. A
file that cannot be read is deleted on the way past so it stops being
tried, and writes go to a temporary file and are moved into place so an
interrupted one leaves the previous entry intact.

### Two things found by writing it

`json` refuses numpy scalars, and the detector and sharpness measure both
hand them back -- so the first save failed at the very end of a scan that
had already cost a minute. Everything is coerced to a plain Python number
on the way in.

`list.index()` cannot be used to find an observation. `FaceObservation` is
a frozen dataclass holding numpy arrays, so its generated `__eq__`
compares embeddings elementwise and returns an array; `index()` raises
"truth value is ambiguous" on the first non-matching member it tries. The
representative is found by identity instead.

### Verified end to end

`timestamps --select-index 0` with and without `--rescan` produce the same
618 lines and the same 152 appearance intervals.

## 22. A reel of several people (`combined_group`)

"Every scene either lead is in" is one reel. Clicking a second card used
to displace the first; it now joins it, and `--select-index` takes several.

The union is built by pooling the chosen people's detections into one
carrier group and running the existing interval stage over the combined
timeline -- not by building each person's intervals and merging the
results. `build_appearance_intervals` reads nothing but sorted timestamps,
so this is less code, and it is also the more correct answer.

The case that separates them: one lead leaves a scene and the other
arrives a second later. On the pooled timeline that is one continuous
appearance. Two separately-built interval lists would already have cut it
in two and padded both halves, producing a visible seam in the middle of a
continuous shot. A test pins exactly that.

On the test episode: person #1 alone is 538 detections over 152 appearance
intervals, person #2 is 283 over 100, and the two together are 821 over
**124** -- fewer intervals than either count suggests, because the scenes
they share merge into one.

The carrier group has no centroid and no representative face, because a
group of two people has neither.

## 23. Showing the reel before encoding it (`preview_frames`)

Selecting a card said "14 cuts, about 4:31" and then asked the user to
commit minutes of encoding to it on faith.

Six frames, spread across the reel's length rather than taken from its
opening -- a 100-cut reel has to be represented by more than its first
minute. Measured at **0.30s** for a 22-minute episode.

This is the one place in the project where seeking is right. Section 18
measured it losing 3.5-5x for sampling, where a frame is wanted every 0.5s
and each seek decodes the whole GOP in front of it; six frames spread over
twenty minutes is the opposite case, and the same reasoning says seeking
wins. The measurement did not say "seeking is slow", it said "seeking is
slow when samples are denser than keyframes", and that distinction is what
makes this cheap.

Frames come from the middle of each segment, not its start: a cut's first
frame often lands mid-transition and shows a face nobody would recognise.

It behaves like a convenience throughout. It runs off the main thread; a
result arriving after the user has clicked again is dropped rather than
drawn over a selection it does not belong to; and any failure -- a moved
file, a codec that will not seek, an exception escaping the thread --
leaves the strip absent instead of stopping an export that would work.

## 24. Correcting the identities (`app/faces/edits.py`)

Grouping is fitted to real footage and gets most of a video right. What it
still does is written down in this file already: one actor across two
cards (15), and the show's logo clustering into a convincing phantom
identity. The only recourse was re-tuning thresholds and rescanning.

Three corrections, chosen because they are the ones a person can make and
the clustering cannot: **merge**, **split**, **discard**.

### Splitting along tracks and nowhere else

A track is the only unit here that is provably one person -- spatial
continuity across consecutive frames proves it (7), and nothing else in
this module does. So it is the only honest place to divide a group.

Those boundaries were being destroyed: `_build_groups` flattens units into
one observation list and the runs cannot be recovered afterwards. Groups
now record the unit sizes as they are built, and consolidation
concatenates them -- which matters, because the consolidated groups are
exactly the ones most likely to need splitting.

Verified on the test episode: the recorded boundaries account for every
observation in all 38 groups, with a median of 21 tracks per person and a
maximum of 279.

### The picker reads its pictures back from the video

A split picker has to show one shot per track, and the cache deliberately
keeps only one crop per person (21). Storing a crop per observation would
take an episode's scan from 6 MB to hundreds, to serve a correction made
rarely and looked at once. So the thumbnails are seeked out of the footage
on demand, through the same path as the reel preview, showing the longest
tracks first -- a mistakenly merged card is two substantial runs of
somebody, not a scattering of single frames.

### Corrections outlive the window

The edited groups are written back over the scan they came from, under the
same key, and the entry is marked as edited. The key still identifies the
scan that produced it; the groups are no longer only what the clustering
said, and that is the point. `--rescan` is the way back to what it
actually said.

### Two defects the tests found

Both are the kind that would have been noticed late and blamed on
something else:

- **Numbering the edited gallery stamped an id onto the groups it was
  handed.** Groups survive an edit untouched -- a discard keeps every other
  card exactly as it was -- so renumbering in place reached back into the
  caller's gallery and into the result of any earlier edit still holding
  the same object. Numbering now produces new group objects.
- **A group with no cover picture is dropped by the gallery**, so an edit
  could make a card the user never touched disappear. Every edited group
  is now guaranteed one.

### Verified on the episode

    38 people
    merge #2 + #3        -> 37 people, 283 + 188 = 471 detections
    split #1 on 2 tracks -> 38 people
    discard the smallest -> 37 people
    reopen the video     -> 37 people, reused

## 25. Naming the people (`rename_group`)

A card said "Person #2", which is a position rather than a person -- and
the position moves. Every correction in section 24 renumbers the gallery
largest-first, so the index a user memorised at breakfast means somebody
else by lunchtime. That is also why `--select-index` was the only durable
way to name a person on the command line and was not durable at all.

The name lives on the group, not on the card. Everything follows from
that: it travels through the scan cache, through a merge, through a split,
and through the renumbering that follows a discard.

### What happens to a name when the group it is on stops existing

- **Merge** keeps the name of whichever card contributed most detections.
  Folding the stray half of an actor into the named one is the correction
  this feature exists for, so losing the name there would punish it.
- **Split** leaves the name with whoever stays behind. A split says "those
  shots are somebody else", so the person keeping the name is the one the
  user did not point at.
- **Discard** takes it with the card.

### A name is a label, but it becomes a filename

`episode-jamie-lee.mp4` says what is in a file in a way
`episode-person-2.mp4` never did. So the characters a path cannot carry
are refused when the name is set rather than mangled when the file is
written, and names are capped at 60 characters.

With nobody named, both the filename and the window's own text keep their
old compact forms -- `person-1+3`, "People #1 and #2" -- rather than
becoming `person-1+person-3` and "Person #1 and Person #2". A name is only
worth spelling out where there is one.

### The third instance of one bug

Renaming deliberately skips the reordering the other edits go through: a
card must not move out from under the cursor that just named it. That also
meant it skipped the guarantee added in section 24 that every edited group
has a cover picture -- and a group without one is dropped by the gallery.
So renaming somebody in a scan whose groups lacked covers silently emptied
the gallery.

That is the same defect as before, in a third place, which is the signal
that it wanted to be one function rather than a line repeated at each
site. `_with_covers` is now applied by every edit including the one that
only changes a name.

### Verified on the episode

    38 people, all "Person #N"
    name #1 "Jamie Lee"      -> label follows
    merge #1 + #4            -> "Jamie Lee", 702 detections
    discard #2               -> still named
    reopen the video         -> reused, still "Jamie Lee"

    python -m app timestamps ... --select-name "jamie lee"
      -> Jamie Lee, 702 detections, 155 appearance intervals

Matching is case-insensitive, and a name nobody has lists the ones that
exist rather than failing blankly.

## 26. Removing `app/settings.py`

It remembered per-mode threshold overrides, and nothing read it. The one
thing it used to decide -- which mode a run used -- was taken away from it
deliberately in `eac7e4d`, because a mode chosen once in the window
silently became the default for every later command-line run. After that
it had no consumer at all: 115 lines of module and 87 of tests, exercised
only by each other.

The regression test that mattered stays. `test_modes.py` writes a settings
file and asserts the mode is **not** read back from it, which is the
behaviour `eac7e4d` established, and it holds whether or not the module
exists.

`app/faces/visualize_detector.py` was left alone: it is in `.gitignore`
and was never part of the repository, so it is a local scratch script
rather than duplicate code shipped with the project.

## 27. Audio drifting behind the picture (`app/video/cutter.py`)

Reported from a real reel: the audio was slower than the video. Measured
on the file it came from -- 102 cuts of the 22-minute footage -- the audio
ran **3.288 seconds** longer than the video by the end.

### Both streams are cut together, which is not the same as equally

One decode loop reads video and audio interleaved and filters both against
the same segment start and end. The trouble is that each stream is cut to
whole frames *of its own kind*, and those are not the same length: 41.7ms
of video against 21.3ms of AAC audio at 48kHz. A cut therefore keeps
slightly more or less of one than the other, depending only on where its
boundaries happen to fall between frames.

Per cut that is inaudible. What made it audible is that the counters run
continuously across segments -- they are what makes the reel one timeline
rather than 102 that each restart at zero -- so every cut's error was added
to a running total instead of being corrected.

It is a random walk rather than a bias, which is why it hid for so long.
Cutting the same 120 seconds three ways, before the fix:

    4 segments x 30s    +6.4ms per cut
    20 segments x 6s    -3.0ms per cut
    40 segments x 3s    +9.2ms per cut

Short reels come out fine. A two-cut reel measured 13ms out, which nobody
would notice. The error only becomes visible once there are a hundred of
them pulling in the same direction for long enough.

### Anchoring the audio to the video's clock

At the end of every segment, the number of audio samples that *should*
have been written is computed from the total video frames written so far,
and the audio is brought up to it.

Cumulatively, which is the part that matters. A target computed per
segment would round 102 times and accumulate its own error; a target
computed from the running video count means a segment that comes up short
is simply made up by the next one. The residual is bounded by one encoder
frame and cannot grow.

Whole encoder frames are emitted and the remainder is left for the next
segment, because AAC wants 1024 samples and a short frame mid-stream is
padded by the encoder -- which would add samples and reintroduce the drift
it is there to prevent.

Two smaller things fall out of the same change. Audio left buffered at a
cut is now dropped rather than carried across the join, where it belonged
to time the reel does not contain and came from a different part of the
source. And a segment whose audio runs out before its picture is padded
with silence, which is what keeps a video-only tail level.

### After

    4 segments x 30s    -0.017s total
    20 segments x 6s    -0.017s total
    40 segments x 3s    -0.017s total

Constant, not per cut: the same 17ms whether the reel has four cuts or
forty, which is the property that was missing. On the 102-cut reel that
started this, +3.288s became **-0.018s** -- under one AAC frame.

Checked that the fix is not silence: 31210 audio frames, none silent, mean
RMS steady across every minute of the reel. Levelling two timelines by
padding one of them would have passed a duration check and failed the only
test that matters.

### Checking the fix against what a viewer notices

Matching durations is not the same as matching content: audio can be the
right length and still be the wrong audio. The duration measurements above
would not have caught a reel whose sound was uniformly half a second early.

So the fix is checked against footage generated for the purpose, with a
white frame and a 50ms tone burst on every whole second -- sound and
picture marked at the same instants. Cut it up, and the distance between
each flash and its click says whether they are still together.

    source, 60 markers      drift -0.003ms per marker  (-0.2ms overall)
    20-cut reel, 20 markers drift -0.074ms per marker  (-1.5ms overall)

Content does not slide. The constant ~30ms offset is the measurement, not
the reel: a burst is timed from the audio frame containing it, so it reads
up to one frame early.

This is asserted as a *spread* rather than a fitted slope. Each offset is
quantised to one audio frame, so across eight cuts a single step of that
size fits a slope of 2.6ms per cut out of nothing at all -- a slope
threshold would be measuring the sampling rather than the sync. Sliding
timelines fan the offsets out; quantisation does not.

The generated footage needs no sample video, so unlike the cutter's other
tests these run in CI, where the assets are absent. That matters more than
it sounds: this is the bug class that shipped, and it is now checked in
the environment that gates releases rather than only on a developer's
machine.

### Still there: audio carried past a cut

The same harness found a smaller thing the fix does not address. Fifteen of
twenty cuts carried a fragment of audio from *after* the cut point -- a
single frame, up to 21ms, of the moment the reel is supposed to have cut
away.

The cause is the same asymmetry: a decoded audio frame that begins before
the segment's end is written whole, and can extend past it. The video has
no equivalent, because it is cut at frame boundaries that are the
timeline's own units.

It is bounded at one audio frame per cut and does not accumulate, so it is
a texture problem rather than a sync one -- most likely audible, if at all,
as a click at a join.

Fixed by giving each segment an entitlement: the window from the first
audio frame kept to the cut itself says how many samples it may keep, and
emitting no more than that leaves the overshoot in the part that is
dropped. Whole encoder frames are still the unit, so the last one emitted
ends at or before the cut and the frame carrying the overshoot is never
written.

    before   15 of 20 cuts carried a fragment
    after     0 of 20

Drift and duration alignment are unchanged by it (-0.074ms per marker,
-0.005s overall), and on the real 22-minute footage the reel came out with
the same number of samples as before.

**That last observation was misread, and the conclusion drawn from it was
wrong.** The totals matched because the shortfall the entitlement created
was being filled with manufactured silence -- see "Silence at every join"
below. The entitlement did take right audio with it, and put zeros in its
place.

### Silence at every join -- found by review, and the real fix

A code review of the two fixes above, run because they went straight to
main without a pull request, found that both of them manufactured silence.
Measured on 100 appearance-shaped cuts of the 22-minute footage, counting
every block of zeros the cutter wrote:

    v1.9.1  drift fix only       31 of 100 joins   661ms of dead air
    v1.9.2  drift + leak fix     98 of 100 joins  2091ms of dead air

About 21ms at nearly every cut, heard as a dropout. v1.9.2 was published.

**Why nothing caught it.** Silence is exactly the right length. Every check
written for the drift -- durations, marker offsets, the spread of those
offsets -- measures timing, and manufactured silence keeps timing perfect.
The one test meant to rule it out asserted that the loudest moment in the
whole reel was louder than 0.001, which a reel of 98 dropouts passes
comfortably. The "no silent frames" check run by hand looked for decoded
frames that were exactly zero, and AAC frames overlap, so a zeroed block
between two loud ones never decodes to exactly zero.

**Why it happened.** Both fixes cut the audio in its own frames and then
patched the difference. The picture a segment keeps is on screen from the
first video frame kept to the moment the last one ends -- up to a frame
past `end`. The audio was taken from a different window: audio frames
straddling the start were dropped whole, and the tail stopped at `end`.
The cumulative levelling then found the audio short of the picture at
almost every join, and the loop meant for "the source ran out of sound"
filled it. The entitlement added for the leak capped the tail at `end`,
which widened the gap and tripled the silence.

**The fix is to stop cutting audio in audio frames at all.** Each segment's
audio is buffered around the cut -- frames straddling the start are kept,
and reading continues a frame past `end` -- and then exactly the picture's
span is taken out by sample offset: from the first video frame kept, for as
many samples as the video frames kept last. Both streams then describe the
same stretch of the source to the sample. The length is still counted from
the running video total, so rounding cannot accumulate, and what is taken
goes into one buffer spanning the whole reel that is encoded in whole
frames; a part-frame left over is just the start of the next frame, which
moves nothing, because the output is one contiguous run of samples.

Silence is now written only where the source genuinely has no sound: a
video-only segment, or a stream whose audio starts late or ends early.

    100 cuts, real footage   silence written 0 times
    drift                    +0.015s overall, constant (AAC priming/padding)
    fragments across a cut   0 of 20 (the leak stays fixed)

The leak stays fixed without the entitlement because the span ends where
the last picture ends. Audio between `end` and that moment is the sound of
the frame still on screen, not of the moment cut away.

**Two tests that fail on the code that shipped.** One counts the samples of
silence the cutter writes while cutting footage whose tone never stops, at
cut points that fall mid-frame the way real appearance intervals do; the
published cutter wrote 6144. The other decodes the reel and checks that no
21ms block is quieter than half the median; the published cutter had a join
at RMS 0.003 against a median of 0.353. Both were run against the v1.9.2
cutter before being accepted, because a test that cannot fail on the bug it
names is the reason this shipped.

## 28. Keying a kept scan on what the answer depends on

Keying a kept scan on `app.__version__` was one line and looked
conservative. It meant every release discarded every kept scan.

1.9.1, 1.9.2 and 1.9.3 changed only how audio is cut. Nothing about
cutting audio changes who is in a video, yet each of those releases made
every kept scan unreadable -- a rescan of 55.3s per episode, and, far
worse, the loss of every name, merge, split and discard stored in it,
because corrections live inside the entry they correct.

That is not a cache miss. A cache miss costs time; this cost work that a
person did by hand and cannot be recomputed.

### The distinction the key was missing

`SCAN_FORMAT` replaces the app version. It moves when the identity
pipeline would answer differently and at no other time, so a release that
touches cutting, the window, or the command line leaves kept scans alone.

The protection the old key provided is kept intact: a test asserts that
raising `SCAN_FORMAT` still invalidates. What is gone is the coupling
between that and the version number on the box.

The one thing this cannot check for itself is a release that changes
grouping and forgets to raise the number. That is a human step, so it is
written down where the constant is defined rather than left to memory.

### Rescuing what the old scheme filed

Entries written by 1.7.0 through 1.9.3 are still on disk under keys built
from those versions. `find` tries the current key, and on a miss tries
each of those, re-filing anything it finds under the current key and
removing the stale copy.

The file layout never changed, which is what makes this possible --
`CACHE_VERSION` stays at 1 deliberately, since bumping it would have made
exactly the entries being rescued unreadable.

Migration moves a file. It does not rescan, and it runs once per video.

### Verified on the real footage

    scanned as 1.9.2          38 people, 53.2s
      named #1 "Jamie Lee"
      merged #2 + #3          37 people
    upgraded to 1.9.4         37 people, 0.4s, reused
      still named             Jamie Lee
      cache entries           1 before, 1 after -- re-filed, not duplicated

The merge survives as well as the name: 37 people rather than the 38 a
fresh scan finds.

## 29. Release notes from the tag's message (`packaging/release_notes.py`)

The release job wrote the same body into every draft: how to get past
Gatekeeper and SmartScreen, and nothing about what changed. So the real
notes were typed into the draft by hand afterwards, once per release, from
memory. That held for 1.9.0 through 1.9.4, and 1.9.0 nearly went out
without mentioning that Export worked again -- the one thing its users
needed to know.

An annotated tag already carries that text, written at the moment the
release is cut. The job now reads it and puts it above the install text.
The install text moved out of the workflow into `packaging/release-body.md`
unchanged, byte for byte, so nothing already published reads differently.

### Two things git does that would have shipped wrong notes

Both were found by writing the tests against a real repository rather than
a fixture. Neither is visible from reading the code.

**A lightweight tag hands back the commit's message.** `%(contents)` on a
tag that is only a name pointing at a commit answers with that commit's
message, not an empty string. The first draft of this would have published

    Merge pull request #5 from Na-bra/fix/keep-names-across-upgrades

as the headline of a release. The object type is now checked -- only
`objecttype == tag` has a message anyone wrote to be read -- and anything
else falls back to the install text alone, which is what every release
before this shipped.

**git deletes markdown headings from a tag message.** The default cleanup
strips every line beginning with `#`, so

    git tag -a v1.2.3 -m "## What changed

    The paragraph."

stores the paragraph and drops the heading. The text never reaches the tag
object, so nothing downstream can put it back.

This one cannot be fixed in code. Cut a release with:

    git tag -a --cleanup=whitespace v1.2.3 -F notes.md

`tests/test_release_notes.py` pins both behaviours, because the second
belongs to whoever types the command and a test is the only place that
knowledge runs.

### What it would have produced

Against the real `v1.9.4` tag, whose message was written by hand into the
draft the old way, the job now composes notes that match it apart from the
markdown formatting -- the tag's message is plain text, since headings
would not have survived being written with `-m`.

A tag with no message, or one whose message only repeats the tag's own
name, produces exactly the notes every release before this one carried. The
change can only add.

## 30. Clicks at the joins (`FADE_SECONDS`)

Measured before fixing, as the backlog asked -- the three releases before
this were all audio that looked right and wasn't.

### The measurement

A steady 440Hz tone cut at the ten irregular intervals the silence tests
use. The largest step between two samples a 0.5-amplitude 440Hz sine can
take is 0.049; the decoded source never exceeds that.

    join   1     2     3     4     5     6     7     8     9
    step   0.5x  12.2x 0.5x  12.9x 0.5x  6.1x  2.6x  13.2x 0.5x

Five of nine joins jumped by up to 13 times anything the tone does -- a
click at nearly full scale. The encoder spread each one into the samples
around it: away from the joins the reel still stepped 2.5x.

The four clean joins were a property of the test signal, not of the cutter.
440Hz advances exactly a third of a cycle per 24fps frame, so any gap of a
multiple of three frames lines the phase back up. Real sound has no period
to fall into step with.

### The probe that found nothing

The first attempt looked for the step at each join's sample index, taken
from the cutter's running audio count after each segment, and every join
measured clean. That count is what has been *encoded*; up to a frame of
the segment is still buffered waiting for a full 1024, so it trails the
real join by 160-768 samples. The spikes were all at exact multiples of
2000 samples -- one 24fps frame -- which briefly looked like a sync fault.

Settled with footage carrying seeded noise instead of a tone, where every
window matches exactly one place in the source. Every segment's sound
starts on the source sample of its first video frame (1.042s, frame 25,
for a cut requested at 1.013s), and every transition sits on a frame
boundary. Sync was exact; the probe was wrong. The test measures the whole
reel for that reason rather than predicting where the joins are.

### The fix

Each segment's sound is eased to zero over its first and last 5ms with a
raised cosine, silence included, so the fade lands at the join whatever the
segment is made of. A segment under 10ms gets two fades that meet in the
middle. No sample is added or removed, so sync and the no-silence count
cannot move.

    before  largest step anywhere in the reel  13.2x
    after                                        1.06x   (every join 0.0x)

`test_no_join_clicks` fails without the fix (0.648 against a limit of
0.098) and passes with it; the sync, silence and dip tests are unchanged.

## 31. One person across a folder of videos (`batch --person --combine`)

The first version of batch needed a photograph of the person. That misses
the point: the tool exists to find people so the user does not have to, and
asking them to go and find a picture of a character first is that work
handed back. A person is now named where they are already visible -- on
their card, in the window -- and found everywhere else from that.

### A name becomes a face

`find_person` reads the kept scans of the videos in the run; nothing is
scanned or decoded to do it, because a name only exists in a scan somebody
has opened. Every card carrying the name, in any video, is normalised and
averaged (`reference_from_groups`), so a card seen for twenty minutes does
not outweigh one seen for twenty seconds. That average is an ordinary
reference face, and from there matching is exactly the photo path: the same
floor, the same margin, the same refusal across modes.

In a scan where the person is named, the named cards are used as they are,
even if the face would have picked another card -- the user's own naming is
the stronger evidence. Everywhere else the face is matched, and an unclear
match skips the video with the scores, rather than cutting a guess into a
reel.

### Measured

The 22-minute episode was split into two files by copying packets, the same
characters appearing in both. Four were named in the first half through the
window's own save path (`apply_edit`), and matched against the second:

    name        card by eye   matched   score   next best
    Lead        #1            #1        0.97    0.13
    Friend      #2            #2        0.96    0.30
    Bald Man    #12           #12       0.93    0.15
    Glasses     #3            #3        0.92    0.22

Four of four, and far clearer than photographs: the photo floor is 0.35,
and a card's average face comes from the footage's own lighting and camera.
The name given to the woman in `test.mp4` was, correctly, found nowhere in
`test_2.MOV`, whose people are different (best score -0.02).

### One reel from many videos (`cut_clips`)

`cut_segments` is now the one-video case of `cut_clips`, so every earlier
cutter and sync test runs through the new path. Videos are all read before
anything is written:

- **Frame rate** must match within one part in ten thousand. Frames are
  stamped by count at the reel's one rate and each segment's sound length is
  derived from its frame count, so 25fps footage in a 24fps reel would play
  4% slow with the wrong span of sound. `test.mp4` (30.000) and
  `test_2.MOV` (30.0015, a phone recording) are within it. Batch leaves a
  mismatched video out by name; `cut_clips` itself refuses.
- **Shape** comes from the first video; another shape is fitted with black
  bars, never stretched.
- **Sound** takes the first video that has any. The resampler is now made
  fresh per segment: one fixes its input format on first use, so two sample
  rates cannot share it, and when converting it holds samples back that
  would otherwise open the next cut with sound from the last.
- **No sound** contributes silence of exactly the picture's length.

Each guard was checked by breaking it. Stretching, dropping the rate check,
and one shared resampler each failed a test. A silent clip one frame long
did not at first: sound is counted from the reel's start, so the error --
42.7ms -- is confined to the next segment and slipped under the usual 50ms
sync bound. That test holds 30ms against an honest spread of 16ms.

On the split episode, the reel for one character came out at 123.457s of
picture against 123.456s of sound.

### What a "scene" still is

Sampled frames from that reel show the other side of the character's
conversations as often as the character: segments are padded and short gaps
bridged, so a reverse shot within a second of a detection is kept. The
frames were checked against the source at the mapped times (mean difference
2-19 of 255) with the character detected 0.5-1.2s away, so this is the
editorial rules working as written, not a cutting error. Snapping cuts to
shot boundaries is the step that would change it.

### Also found

- `batch` parsed `--rescan` and never passed it on, so it always reused.
- The window scanned animation with live action's thresholds. See 32.

## 32. The window's animation scans used live action's thresholds

`ScanSettings` defaults are live action's numbers, and the window built its
settings by changing only `mode`. Animation was grouped with a similarity
floor of 0.35 where the mode's own is 0.75, and detected with live action's
face size and confidence floors. Found because the window's kept-scan key
for animation never matched the command line's, which would have made a
name given in the window invisible to `batch --person`.

On the 7-minute animation sample at a 0.5s interval:

    window as shipped      306 faces   76 tracks    2 people (163, 117)
    animation's own        344 faces   108 tracks   6 people

`ScanSettings.for_mode` resolves a mode as the command line resolves unset
flags, and a test holds the two equal for every mode. Live action's values
were already right, so its kept scans and names are unaffected; animation
scans kept by the window were wrong and scan again.

## 33. A folder's cast in the window (`app/faces/cast.py`, `app/ui/folder.py`)

The window's answer to "who is in this season?": one card per person across
every episode, picked by clicking a face from the footage. Nothing is
uploaded and no photo is needed.

### Linking cards across videos

Each episode's scan already groups its own faces into cards. Linking them
uses the rule `batch --person` uses to find a named person elsewhere, so
the window and the command line agree:

- Two cards in different videos link when **each is the other's clear best
  match** in its video: over the mode's floor, and ahead of that video's
  runner-up by the match margin. Mutual, so one card cannot be claimed by
  two people.
- **Cards with one name are one person**, and cards with different names
  never are.
- Anything plausible that fails the rule is a **"same person?" question**,
  shown as two faces. Answers override every score and are applied last:
  applied first, a "yes" joining a split card made the person look as if
  it already held a card from that video, and the clear link the answer
  relied on was refused.

Questions are only asked across videos. Within one, grouping has already
kept two cards apart, often because both faces share a frame; a split card
still surfaces through its match in another episode.

### Measured

The 22-minute episode split into two files, scanned as the window does:

    57 cards -> 47 people
    10 linked across both episodes     all 10 right by eye, none wrong
    4 questions                        all 4 genuine candidates

The four questions were the lead's split card against the lead, the masked
hero in two expressions, a girl in stage makeup, and the hero unmasked
against masked -- each a thing worth a person's look, and no pair of
strangers among them. Linked cards scored 0.92-0.97; no other card reached
0.30 against them.

Naming a person from the cast wrote the name into both episodes' kept
scans; reopened, it was on both cards, and the single-video view shows it
too. Their reel, exported from the window, came out at 123.457s of picture
against 123.456s of sound.

### What driving the window found

The folder view was built against tests of the bridge, then driven for
real: the page opened by pywebview, clicked from a script, and screenshotted.
That found three bugs the tests had not:

- **A line through every card's name.** The name block was an inline span,
  so its top border drew through the text once the face strip sat above it.
- **An empty orange box over the single-video gallery.** The question panel
  is a grid, and `display: grid` beat the `hidden` attribute. The page now
  makes `hidden` always win.
- **A reel saved under the wrong video's name** -- and not only in the new
  view. The window forgot its own last file-name suggestion when a scan
  finished, so that suggestion then looked typed by hand and was kept:
  scan one video and pick someone, scan a second, and the box still named
  the first. It shipped in the single-video window. The suggestion is now
  kept across scans, with a test for each view.

A test also now holds the page and the bridge to each other: every method
the page calls must exist on the bridge, and every event the bridge sends
must have a handler. Neither side fails loudly when the other is missing
one -- a missing method is a button that does nothing.

### Not yet

Answers to "same person?" last for the session; a "yes" is kept beyond it
only by naming the person, since cards with one name are one person. Merge,
split and discard are single-video corrections and are hidden in folder
view.

## 34. Correcting the cast, and keeping it corrected

Folder view had no way to fix what the cast got wrong, and what it was told
lasted only as long as the window.

**Answers are kept** in `cast-answers.json` beside the kept scans, keyed by
the scan a card belongs to and the card's first sighting and detection
count -- not its gallery position, which moves with every correction. A card
merged or split since gets a new key, and old answers stop applying to it.

**Merge, Split and Not a person** now work across the folder. Merge and
split are recorded as answers, so they are kept the same way. Split shows
one face per card with its video, and clears a detached card's copy of a
shared name, since a name would otherwise put it straight back. Not a person
removes the cards from every video's kept scan, never emptying a video.
Several people can be chosen at once, for a reel of every scene any of them
is in.

Driven in the real window across three separate launches -- merge two
people, reopen, split them, reopen:

    merge      47 -> 46 people
    reopened   still one person
    split      46 -> 47
    reopened   still apart

That found two bugs:

- **A merge that did nothing.** An earlier split had left "different"
  answers between the two people's cards, and those vetoed the new "same".
  Nothing said so. A merge is now the newest word about those people and
  overrides earlier answers between them, and the window never reports a
  join the cast does not show.
- **A split picker off-screen.** It sat below the grid, which for a cast of
  forty is far below the fold, so Split... looked like it did nothing. It
  now opens under the buttons, in both views.

## 35. Joining videos at different frame rates

A reel counts its frames at one rate and measures each segment's sound from
how many frames it kept, so a video at another rate could not be copied
across: 25fps footage in a 24fps reel would play 4% slow with the wrong
span of sound. Such videos were left out by name. They are now converted.

Each source frame is written for as many reel frames as start while it is
on screen -- every k with k/reel < n/source for the n-th frame read --
counted in whole frames with exact fractions, so a long segment cannot drift
the way summing float timestamps would. A faster source drops frames, a
slower one repeats them. Rates within FRAME_RATE_TOLERANCE are still copied
one for one.

One more thing had to move: a segment's sound is read to one frame past its
cut, and for a slower source that has to be one of *its* frames, which
covers more of the reel. Read one reel frame instead, the shortfall was
filled with silence -- but only visibly below about 20fps, because sound is
decoded in ~21ms blocks and the last one read always overruns the cut. The
test uses 10fps footage (58ms short) for that reason; at 20fps (8ms short)
the overrun hid the bug.

Measured on real footage, 10 seconds of the 23.976fps episode followed by
10 seconds of the 30fps square clip:

    reel         20.020s of picture, 20.032s of sound (the last AAC block)
    30fps part   every checked frame matches the source frame at exactly
                 the time it should, 1.0s to 9.0s in -- no drift

Batch and the folder view no longer leave a video out for its frame rate;
only one that cannot be read.

## 36. Leaving repeated footage out of season reels (`app/video/repeats.py`)

A season reel cut naively has a character in the recap twice, and one in
the opening credits once per episode.

### Faces could not find repeats

The scans already hold every face they saw, so the first attempt looked
there. A two-minute recap was made from the test episode the way a real
one is -- re-encoded, at lower quality, starting at 30.23s so the scan's
0.5s sampling falls between the original's samples. Its faces matched
their originals at a median of only 0.82 (each is caught a fraction of a
second later), while new footage of the same people reached 0.86.

Two refinements did not rescue it. Voting for one time offset per run of
sightings let chance alignments through (64-82% of an unrelated episode's
runs "aligned"). Pooling votes over the whole cast and requiring the face
in the same place on screen still flagged thirteen stretches of an episode
that repeats nothing, and found only 29 of the recap's 120 seconds.

### Frames can

A repeat is the same pixels, whoever is in them. A 63-bit perceptual hash
of a frame every 0.5s:

    closest frame, bits apart   median   within 8   within 12
    recap vs its episode           4        72%        84%
    other episode vs it           18         0%         1%

Frames within 10 bits vote for the offset between the videos; the frames
agreeing with the best one, joined across gaps of up to 5s (fast motion
hashes less alike), are the repeated stretches. On the three videos:

    recap repeats episode 1     0.0-119.2s at +30.239s (made at +30.23s)
    every other pair            nothing, at 8, 10 and 12 bits

The offset was first taken as the centre of the winning vote window,
0.15s off; the median of the matches in it is 9ms off. Frames with almost
no detail -- black, a flat card -- look alike everywhere and do not vote.

### What is left out

Only footage the reel really does show from an earlier video. A recap of a
scene the reel skipped is the only time it appears, so it stays; slivers
under 0.5s left by a cut are dropped.

On episode 1, episode 2 and the recap, `batch --person Lead --combine`:

    recap      79.9s planned -> 5.5s kept, 74.4s left out
    episodes   unchanged (300.9s, 355.8s)

The 5.5s kept are two recap moments (2.0s and 3.5s) that episode 1's own
cut of the lead does not include -- checked by mapping them back.

Fingerprinting costs one more read of each video, about 11-13s per 11
minutes of 720p, and is kept beside the scans keyed by the file alone;
finding the repeats between two videos then takes about 10ms. The folder
view does both when a folder is scanned, so choosing a person stays
instant.


## 37. Slivers of another shot at the edges of a cut (`SLIVER_FRAMES`)

The plan was to "snap cuts to shot boundaries" and, with it, stop reels
showing the other side of conversations. Measuring first split that in two.

### Dropping shots without the person: not worth it

On the lead's reel from the first half of the test episode, 172 shots:

    shots with the lead detected     93, 271.1s (90%)
    shots with no detection          79,  30.8s (10%), none over 1.9s

and looking at the twelve longest of the second kind, about half show the
lead anyway -- from behind, or at the edge of an over-the-shoulder shot.
Dropping them would cut real shots of the character to save some fifteen
seconds of other people in five minutes. Not built.

### Slivers at the edges: common, and visible

A segment is an appearance padded at both ends, and the padding keeps
reaching across a camera cut:

    segment edges within half a second of a cut   63% (half 1), 75% (half 2)
    frames of the neighbouring shot left there    median 7, about 0.3s

-- a flash of somebody else as a clip opens or closes.

### Trimming them in the cutter

The cutter decodes every frame of a segment anyway, so it holds the last
SLIVER_FRAMES (12, half a second at 24fps) back, and at either edge leaves
out what lies across a cut within them. Sound needs nothing of its own: it
is taken from the first frame written for as long as the frames written
last, so it follows the trim.

A cut is a spike in how much two frames' 64x36 colour thumbnails differ:
at least 12 of 255 on average, four times what the frames either side
change, with those frames steady. Relative, because a cut between two
angles of one set can change by 20 while fast motion changes more; colour,
because two shots differed by 24 in grey and 91 in colour; steady either
side, because a one-frame white flash is two big changes that undo each
other. On the real footage all twelve sampled detections were genuine cuts.

The lead's reel: 301.9s -> 281.7s, 20.2s of slivers gone; picture and sound
10ms apart at the end; 23.8s to cut against 23.6s. `trim_slivers=False`
turns it off.

## 38. Remembering people across videos (`app/faces/library.py`)

A name belonged to one video's kept scan. Naming someone in season 1 did
nothing for season 2, and `batch --person` could only find a name in the
folder it was given.

### An index of the names already given

The people library keeps every named card's face, by name and embedding
space. It is written from `scans.save`, so naming, renaming and clearing a
name reach it the moment they reach a scan, with nothing of their own to
keep in step. Faces outlive the scan they came from -- a scan pruned for
space still taught the library -- and it lives beside the scans, not in
them, so clearing kept scans does not forget anybody. The first time it is
used it fills itself from names already in kept scans.

One face per person per video rather than one average: a character across
a season changes lighting, costume and age, and each face they were named
on is matched separately.

Suggestions use the rule used everywhere else, both ways round: the
person's best card clears the floor and the scan's other cards by the
margin, and that card's best person clears the library's other people.
Nothing is named until someone says yes.

### Measured

With four characters named in the first half of the test episode:

    episode 2      Lead 0.97, Friend 0.96, Glasses 0.92 -- all right
    the recap      Lead 0.98, Friend 0.97
    test.mp4       nothing
    test_2.MOV     nothing (other people entirely)

Filling itself from the kept scans on first use found names the user had
already given in the window -- Henry, Ray, Pheobe and others -- and offered
"Ray" (0.74) for episode 2's masked hero, which is who Ray is. It also found
names given while testing this project, which is how two names came to be
offered for one face; those were cleared from the kept scans before this
shipped, leaving only the user's own.

In the real window, episode 2's gallery opened with Lead?, Friend?,
Glasses? and Ray? on their cards; choosing Glasses? asked in the rail, and
That's them named the card.

## 39. Detection settings in the window (`app/ui/tuning.py`)

The window offered a mode and a sampling interval; every other setting was
a command-line flag. The Advanced panel adds the three that change most
what a scan finds, stores only what was changed, per mode, and applies it
to every scan the window starts, one video or a folder.

### What each does

Measured on the 22-minute test episode at a 1s interval, every other
setting at live action's own (38 people):

    how alike (floor, merge moved with it)
        0.25   38 people   lead's card 545 faces
        0.35   38          538   (live action's own)
        0.45   41          509
        0.55   40          491
    least screen time
        1s    118 people   3s  48   automatic (7s here)  38
        10s    35          30s 21   -- main cards identical at every value
    smallest face
        24px   41 people   lead's card 548
        40px   38          538   (live action's own)
        80px   18          442

The descriptions in the panel were first written from what the settings
are meant to do, and two were wrong. Lowering "how alike" did not reduce
the number of cards here; it moved more of each person's faces onto their
card. And a larger smallest face did not make anyone "recognised more
reliably" -- it dropped people, and took the distant shots out of the
cards that stayed. The panel now says what was measured.

### How alike is two numbers

The pipeline joins faces above a floor and later merges cards above a
looser second threshold. Moved alone, the floor is partly undone:

    floor 0.55, merge moved to 0.575   lead's card 491
    floor 0.55, merge left at 0.375    lead's card 546

so the one control moves both and keeps the gap each mode was tuned with.

### Nothing changed means nothing changed

Only changed values are stored, and a value set back to the mode's own
counts as unchanged. With nothing changed the window builds exactly the
settings the command line does, so the two still share kept scans --
checked in the real window: the unchanged scan reused its kept one, the
80px scan did not, and after a relaunch the 80px setting was still there
until "Use Live Action's own values" put it back.

Screen time left to the mode is worked out per video from its length, so
it shows as Automatic, with the slider dimmed until someone sets a value.

## 40. The season report (`app/report.py`)

Once a folder is scanned, every episode's people are on disk and linked
into one cast, so who is in which episode, and for how long, costs counting
rather than scanning.

**Screen time** in a video is the total of the person's appearance
intervals there, built by the same function `timestamps` uses. Checked on
the test season -- the 22-minute episode as two halves and a recap -- for
the three biggest characters in both halves, against the real `timestamps`
command on the same card:

    report 239.90s   timestamps 239.91s      report 281.95s   timestamps 281.90s
    report 136.48s   timestamps 136.47s      report 148.83s   timestamps 148.79s
    report 143.58s   timestamps 143.60s      report  35.99s   timestamps  36.00s

The largest difference, 0.05s, is `timestamps` rounding its printed times.

**Repeats.** The backlog asked for both "matches timestamps" and "repeats
count once", which disagree: a recap is real screen time in the episode
that shows it. So each episode's figure includes it, and the total says how
much of it repeats footage already counted earlier -- the lead's 9:43 with
0:58 repeated, from the recap.

From kept scans the whole report takes 0.6s for the three videos. It is a
CSV (seconds to a tenth, for a spreadsheet) and a self-contained page --
faces inlined, nothing fetched -- whose cells are shaded by each person's
share of that episode's busiest, so each episode's leads stand out down its
column. The name cell was first laid out as a flex box, which took it out
of the table and let its row lines drift from the others'; it now wraps its
contents instead.
