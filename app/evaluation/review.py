"""Reviewing drafted truth: a page to look over and correct, and the corrections.

The page is a local file, written beside the other output and never
published: it shows faces from the user's own videos. Faces are shown a run
at a time -- one person's faces in consecutive sampled frames, at the same
place -- since a run is what a label is usually wrong about as a whole. The
page names people, moves runs from one person to another, and saves what
changed as corrections, which `apply_corrections` writes back into the truth.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.truth import NOT_A_FACE, UNKNOWN, GroundTruth, TruthFace, overlap

# Faces this close in time and this overlapping are one run.
RUN_GAP_SECONDS = 1.01
RUN_OVERLAP = 0.1
THUMB = 88


@dataclass
class Run:
    label: str
    faces: list[TruthFace] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.faces[0].t


def runs_of(truth: GroundTruth) -> list[Run]:
    """The truth's faces as runs, in order of their first face."""
    runs: list[Run] = []
    open_runs: list[Run] = []
    for face in sorted(truth.faces, key=lambda f: (f.t, f.box)):
        open_runs = [r for r in open_runs if face.t - r.faces[-1].t <= RUN_GAP_SECONDS]
        home = next(
            (
                r for r in open_runs
                if r.label == face.label and r.faces[-1].t < face.t
                and overlap(r.faces[-1].box, face.box) >= RUN_OVERLAP
            ),
            None,
        )
        if home is None:
            home = Run(face.label)
            runs.append(home)
            open_runs.append(home)
        home.faces.append(face)
    return runs


def _thumbnails(video: Path, runs: list[Run], interval: float) -> list[str]:
    """A JPEG of each run's largest face, as a data URI."""
    import cv2

    from app.video.frames import extract_frames
    from app.video.loader import load_video

    def area(face):
        x0, y0, x1, y1 = face.box
        return (x1 - x0) * (y1 - y0)

    shown = [max(run.faces, key=area) for run in runs]
    # Truth keeps timestamps to 4 places; a frame is matched to within a hair of that.
    wanted: dict[int, list[int]] = {}
    for i, face in enumerate(shown):
        wanted.setdefault(round(face.t * 100), []).append(i)
    thumbs = [""] * len(runs)
    remaining = len(shown)
    with load_video(video) as container:
        for t, frame in extract_frames(container, sample_interval=interval):
            key = round(t * 100)
            for near in (key - 1, key, key + 1):
                for i in wanted.get(near, ()):
                    if thumbs[i] or abs(shown[i].t - t) > 0.001:
                        continue
                    x0, y0, x1, y1 = shown[i].box
                    crop = frame[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
                    scale = THUMB / max(crop.shape[:2])
                    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
                    crop = cv2.cvtColor(cv2.resize(crop, size), cv2.COLOR_RGB2BGR)
                    ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    thumbs[i] = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode() if ok else ""
                    remaining -= 1
            if not remaining:
                break
    return thumbs


def write_review(truth_path: Path, video: Path, out: Path) -> Path:
    """Writes the review page for one truth file."""
    truth = GroundTruth.load(truth_path)
    runs = runs_of(truth)
    thumbs = _thumbnails(video, runs, truth.interval)
    data = {
        "name": truth_path.stem,
        "video": truth.video,
        "notes": truth.notes,
        "reviewed": truth.reviewed,
        "people": truth.people,
        "special": [UNKNOWN, NOT_A_FACE],
        "runs": [
            {"label": run.label, "thumb": thumb, "faces": [[f.t, list(f.box)] for f in run.faces]}
            for run, thumb in zip(runs, thumbs)
        ],
    }
    page = _PAGE.replace("__TITLE__", html.escape(f"Review {truth_path.stem}"))
    page = page.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def apply_corrections(truth_path: Path, corrections: dict) -> tuple[int, GroundTruth]:
    """Writes a review's corrections into the truth. Returns faces relabelled."""
    truth = GroundTruth.load(truth_path)
    for label, name in corrections.get("people", {}).items():
        if label not in (UNKNOWN, NOT_A_FACE):
            truth.people[label] = name
    relabel = {(round(t, 4), tuple(box)): label for t, box, label in corrections.get("faces", [])}
    changed = 0
    faces = []
    for face in truth.faces:
        label = relabel.get((round(face.t, 4), face.box), face.label)
        changed += label != face.label
        faces.append(TruthFace(face.t, face.box, label))
    truth.faces = faces
    used = {f.label for f in faces} | {m for c in truth.checked for m in c.missed}
    for label in used - set(truth.people) - {UNKNOWN, NOT_A_FACE}:
        truth.people[label] = label
    if corrections.get("reviewed"):
        truth.reviewed = True
        truth.labelled_by = "draft, reviewed"
    truth.save(truth_path)
    return changed, truth


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --soft:#5d6677; --line:#dde1e8; --accent:#2f6fde;
        --moved:#d9822b; --bad:#c2453d; --chip:#eef1f6; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171d; --panel:#1c2028; --ink:#e6e9ef; --soft:#9aa3b2;
        --line:#2c323d; --accent:#6c9cf0; --moved:#e59a4c; --bad:#e0726b; --chip:#252a34; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system, "Segoe UI", sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:16px; margin:0 12px 0 0; }
header .count { color:var(--soft); font-variant-numeric:tabular-nums; }
button, select, input { font:inherit; color:inherit; }
button { background:var(--chip); border:1px solid var(--line); border-radius:6px; padding:5px 10px; cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button:focus-visible, select:focus-visible, input:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
select, input { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
main { padding:12px 16px 60px; max-width:1500px; margin:0 auto; }
.notes { color:var(--soft); max-width:70ch; margin:4px 0 14px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:8px; margin:0 0 12px; }
section > .head { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:8px 12px; border-bottom:1px solid var(--line); }
section > .head code { font-weight:600; }
section > .head .n { color:var(--soft); font-variant-numeric:tabular-nums; }
.grid { display:flex; flex-wrap:wrap; gap:6px; padding:10px 12px; }
.run { width:96px; border:2px solid transparent; border-radius:6px; padding:2px; cursor:pointer; background:var(--chip);
       text-align:center; font-size:11px; color:var(--soft); font-variant-numeric:tabular-nums; }
.run img { display:block; width:88px; height:88px; object-fit:contain; margin:0 auto 2px; background:#0002; border-radius:4px; }
.run.sel { border-color:var(--accent); }
.run.moved { border-style:dashed; border-color:var(--moved); }
.run .to { color:var(--moved); font-weight:600; }
.help { color:var(--soft); font-size:13px; }
</style></head><body>
<header>
  <h1 id="title"></h1>
  <span class="count" id="count"></span>
  <label>Move selected to <select id="target"></select></label>
  <button id="move">Move</button>
  <button id="clear">Clear selection</button>
  <label><input type="checkbox" id="reviewed"> Reviewed</label>
  <button class="primary" id="save">Save corrections</button>
</header>
<main>
  <p class="help">Each tile is one run: someone's faces in consecutive sampled frames. Click tiles to select them, pick a person
  and press Move. Rename anyone in the box beside their label. Save corrections, then run
  <code id="cmd"></code>.</p>
  <p class="notes" id="notes"></p>
  <div id="sections"></div>
</main>
<script>
const DATA = __DATA__;
const people = Object.assign({}, DATA.people);
const runs = DATA.runs.map((r, i) => ({...r, i, to: null}));
const selected = new Set();
const labelOf = r => r.to ?? r.label;
document.getElementById("title").textContent = "Review " + DATA.name;
document.getElementById("notes").textContent = DATA.notes;
document.getElementById("reviewed").checked = DATA.reviewed;
document.getElementById("cmd").textContent = "python -m app evaluate " + DATA.name + " --correct corrections-" + DATA.name + ".json";

function labels() {
  const count = {};
  for (const r of runs) count[labelOf(r)] = (count[labelOf(r)] || 0) + r.faces.length;
  const named = Object.keys(people).filter(l => !DATA.special.includes(l));
  for (const l of Object.keys(count)) if (!named.includes(l) && !DATA.special.includes(l)) named.push(l);
  named.sort((a, b) => (count[b] || 0) - (count[a] || 0));
  return [...named, ...DATA.special];
}
function fillTarget() {
  const t = document.getElementById("target"), keep = t.value;
  t.innerHTML = "";
  for (const l of labels()) t.add(new Option(l + (people[l] && people[l] !== l ? " — " + people[l] : ""), l));
  t.add(new Option("a new person…", "__new"));
  if ([...t.options].some(o => o.value === keep)) t.value = keep;
}
function render() {
  const host = document.getElementById("sections");
  host.innerHTML = "";
  const moved = runs.filter(r => r.to !== null).length;
  document.getElementById("count").textContent = runs.length + " runs, " + moved + " moved, " + selected.size + " selected";
  for (const l of labels()) {
    const mine = runs.filter(r => r.label === l);
    if (!mine.length && !runs.some(r => r.to === l)) continue;
    const sec = document.createElement("section");
    const head = document.createElement("div");
    head.className = "head";
    const faces = runs.filter(r => labelOf(r) === l).reduce((n, r) => n + r.faces.length, 0);
    head.innerHTML = "<code></code><span class='n'></span>";
    head.querySelector("code").textContent = l;
    head.querySelector(".n").textContent = faces + " faces";
    if (!DATA.special.includes(l)) {
      const name = document.createElement("input");
      name.id = "name-" + l; name.value = people[l] ?? l; name.setAttribute("aria-label", "Name of " + l);
      name.addEventListener("change", () => { people[l] = name.value.trim() || l; fillTarget(); });
      head.appendChild(name);
    }
    const all = document.createElement("button");
    all.textContent = "Select all";
    all.addEventListener("click", () => { mine.forEach(r => selected.add(r.i)); render(); });
    head.appendChild(all);
    sec.appendChild(head);
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const r of mine) {
      const tile = document.createElement("div");
      tile.className = "run" + (selected.has(r.i) ? " sel" : "") + (r.to !== null ? " moved" : "");
      tile.tabIndex = 0;
      tile.innerHTML = "<img alt=''><div></div>";
      tile.querySelector("img").src = r.thumb;
      tile.querySelector("div").innerHTML = r.faces[0][0].toFixed(1) + "s · " + r.faces.length + "f" +
        (r.to !== null ? "<br><span class='to'></span>" : "");
      if (r.to !== null) tile.querySelector(".to").textContent = "→ " + r.to;
      const toggle = () => { selected.has(r.i) ? selected.delete(r.i) : selected.add(r.i); render(); };
      tile.addEventListener("click", toggle);
      tile.addEventListener("keydown", e => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
      grid.appendChild(tile);
    }
    sec.appendChild(grid);
    host.appendChild(sec);
  }
}
document.getElementById("move").addEventListener("click", () => {
  let to = document.getElementById("target").value;
  if (to === "__new") {
    let n = 1; while (people["new" + n] || runs.some(r => labelOf(r) === "new" + n)) n++;
    to = prompt("Label for the new person (no spaces):", "new" + n);
    if (!to) return;
    to = to.trim().replace(/\s+/g, "-");
    people[to] = people[to] ?? to;
  }
  for (const i of selected) runs[i].to = runs[i].label === to ? null : to;
  selected.clear(); fillTarget(); render();
});
document.getElementById("clear").addEventListener("click", () => { selected.clear(); render(); });
document.getElementById("save").addEventListener("click", () => {
  const faces = [];
  for (const r of runs) if (r.to !== null) for (const [t, box] of r.faces) faces.push([t, box, r.to]);
  const out = {name: DATA.name, people, faces, reviewed: document.getElementById("reviewed").checked};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "corrections-" + DATA.name + ".json";
  a.click();
});
fillTarget(); render();
</script></body></html>
"""
