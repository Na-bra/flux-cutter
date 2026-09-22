"""A season report: who is in which episode, and for how long.

Everything it needs is already on disk once a folder has been scanned --
each episode's people, linked into one cast with their names -- so a report
costs counting, not scanning.

**Screen time** for a person in an episode is the total length of their
appearance intervals there, built exactly as `timestamps` builds them, so
each cell can be checked against that command's output for the same person.

**Repeats.** A recap really is screen time in the episode that shows it, so
each episode's figure includes it. The season total says how much of it
repeats footage already counted in an earlier episode -- found the way the
reels find it (app/video/repeats.py) -- so the total is not quietly inflated
by every "previously on".
"""

from __future__ import annotations

import base64
import csv
import html
import io
from dataclasses import dataclass
from pathlib import Path

from app.faces.cast import Answers
from app.ui.folder import CastPerson, FolderScan, cast_of
from app.ui.worker import combined_group
from app.video.repeats import without_repeats
from app.video.timeline import build_appearance_intervals


@dataclass(frozen=True)
class ReportRow:
    """One person across the season."""

    label: str
    named: bool
    # Seconds on screen in each video, in folder order; 0.0 where absent.
    seconds: list[float]
    # Of the total, how much repeats footage an earlier video already had.
    repeated: float
    person: CastPerson

    @property
    def total(self) -> float:
        return sum(self.seconds)

    @property
    def episodes(self) -> int:
        return sum(1 for s in self.seconds if s > 0)


@dataclass(frozen=True)
class SeasonReport:
    title: str
    videos: list[str]
    rows: list[ReportRow]
    sample_interval: float


def _intervals(folder: FolderScan, video: int, cards) -> list:
    result = folder.videos[video]
    return build_appearance_intervals(
        combined_group(cards),
        video_duration=result.video_duration,
        sample_interval=result.sample_interval,
    )


def season_report(folder: FolderScan, answers: Answers | None = None, title: str = "") -> SeasonReport:
    """Everyone in a scanned folder, against every video in it."""
    cast, _ = cast_of(folder, answers)
    rows = []
    for person in cast:
        by_video: dict[int, list] = {}
        for video, card in person.appearances:
            by_video.setdefault(video, []).append(card)

        per_video = []
        seconds = [0.0] * len(folder.videos)
        for video, cards in sorted(by_video.items()):
            intervals = _intervals(folder, video, cards)
            seconds[video] = sum(i.end_time - i.start_time for i in intervals)
            per_video.append((video, intervals))

        _, removed = without_repeats(per_video, folder.repeats)
        rows.append(
            ReportRow(
                label=person.label,
                named=bool(person.name),
                seconds=seconds,
                repeated=sum(removed.values()),
                person=person,
            )
        )

    rows.sort(key=lambda row: (-row.total, row.label.casefold()))
    return SeasonReport(
        title=title or (folder.videos[0].video_path.parent.name if folder.videos else "Season"),
        videos=[result.video_path.name for result in folder.videos],
        rows=rows,
        sample_interval=folder.settings.sample_interval,
    )


def clock(seconds: float) -> str:
    """m:ss, or h:mm:ss past an hour."""
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# ------------------------------------------------------------------ files


def write_csv(report: SeasonReport, path: Path) -> Path:
    """Seconds, to a tenth, so a spreadsheet can add them up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["person", "episodes", "total seconds", "repeated seconds", *report.videos]
        )
        for row in report.rows:
            writer.writerow(
                [
                    row.label,
                    row.episodes,
                    f"{row.total:.1f}",
                    f"{row.repeated:.1f}",
                    *(f"{s:.1f}" for s in row.seconds),
                ]
            )
    return path


def _face(person: CastPerson) -> str:
    _, card = max(person.appearances, key=lambda a: a[1].detection_count)
    buffer = io.BytesIO()
    card.thumbnail.convert("RGB").resize((56, 56)).save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_html(report: SeasonReport, path: Path) -> Path:
    """A readable page: one row per person, a cell per episode.

    Self-contained -- faces inlined, no scripts or fonts fetched -- so it
    opens from disk anywhere and can be sent as a single file. A cell's
    shade is that person's share of the episode's busiest person, so the
    leads of each episode stand out down its column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    busiest = [max((row.seconds[i] for row in report.rows), default=0.0) or 1.0 for i in range(len(report.videos))]
    total_repeated = sum(row.repeated for row in report.rows)

    head_cells = "".join(
        f'<th scope="col" title="{html.escape(name)}">{html.escape(Path(name).stem)}</th>'
        for name in report.videos
    )
    body = []
    for row in report.rows:
        cells = []
        for index, seconds in enumerate(row.seconds):
            if seconds <= 0:
                cells.append('<td class="none">·</td>')
                continue
            share = min(1.0, seconds / busiest[index])
            cells.append(f'<td style="--share:{share:.2f}">{clock(seconds)}</td>')
        repeated = f'<span class="rep">{clock(row.repeated)} repeated</span>' if row.repeated >= 0.5 else ""
        body.append(
            f"<tr><th scope=\"row\"><span class=\"person\"><img src=\"{_face(row.person)}\" alt=\"\">"
            f"<span class=\"who{'' if row.named else ' unnamed'}\">{html.escape(row.label)}</span></span></th>"
            f"<td class=\"total\">{clock(row.total)}{repeated}</td>"
            f"<td class=\"count\">{row.episodes} of {len(report.videos)}</td>{''.join(cells)}</tr>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report.title)} — who is in it</title>
<style>
  :root {{ --ground:#F4F6F5; --surface:#FFFFFF; --ink:#12171A; --muted:#5A676C; --faint:#8A9599; --rule:#D3DAD9; --heat:14,110,107; --signal:#9A5610; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --ground:#0B0F11; --surface:#151C1F; --ink:#E6ECEA; --muted:#9AA8AB; --faint:#6E7C80; --rule:#2A3538; --heat:70,189,180; --signal:#E0A44B; }} }}
  body {{ margin:0; background:var(--ground); color:var(--ink); font:14px/1.5 "Helvetica Neue", Arial, sans-serif; padding:32px 18px; }}
  main {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; margin:0 0 4px; }}
  p.meta {{ color:var(--muted); margin:0 0 20px; }}
  .scroll {{ overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:6px; }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
  th, td {{ padding:6px 10px; border-bottom:1px solid var(--rule); text-align:right; white-space:nowrap; }}
  thead th {{ position:sticky; top:0; background:var(--surface); font:500 11px ui-monospace, Menlo, monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--faint); }}
  tbody th {{ text-align:left; font-weight:600; }}
  tbody th .person {{ display:flex; align-items:center; gap:10px; }}
  tbody th img {{ width:36px; height:36px; border-radius:4px; object-fit:cover; }}
  .unnamed {{ color:var(--muted); font-weight:500; }}
  td.total {{ font-weight:600; }}
  td.total .rep {{ display:block; font-weight:400; font-size:11px; color:var(--signal); }}
  td.count {{ color:var(--muted); }}
  td[style] {{ background:rgba(var(--heat), calc(var(--share) * .38)); }}
  td.none {{ color:var(--faint); text-align:center; }}
  footer {{ color:var(--faint); font-size:12px; margin-top:14px; max-width:70ch; }}
</style></head><body><main>
<h1>{html.escape(report.title)} — who is in it</h1>
<p class="meta">{len(report.rows)} people across {len(report.videos)} videos. Screen time is each person's appearances in that video, as the <code>timestamps</code> command counts them.</p>
<div class="scroll"><table>
<thead><tr><th scope="col" style="text-align:left">Person</th><th scope="col">Total</th><th scope="col">In</th>{head_cells}</tr></thead>
<tbody>{''.join(body)}</tbody>
</table></div>
<footer>Each episode's figure includes any recap it shows; the total notes how much of it repeats footage an earlier episode already had{f' — {clock(total_repeated)} across everyone' if total_repeated >= 0.5 else ''}. Sampled every {report.sample_interval:g}s, so figures are good to about that.</footer>
</main></body></html>
"""
    path.write_text(page, encoding="utf-8")
    return path
