"""The season report: who is in which video, and for how long.

The scans are stand-ins built from observations at known times, and the
intervals are built by the same function the `timestamps` command uses,
so each figure is checked against that command's own arithmetic.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.report import clock, season_report, write_csv, write_html
from app.ui.folder import FolderScan
from app.ui.worker import Person, ScanResult, ScanSettings
from app.video.repeats import Repeat
from app.video.timeline import build_appearance_intervals

SETTINGS = ScanSettings.for_mode("live")


def face(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def person(index, vector, seen_at, name=None):
    observations = [
        FaceObservation(
            embedding=vector,
            detection=FaceDetection(box=BoundingBox(0, 0, 50, 50), confidence=0.9),
            face_crop=np.zeros((8, 8, 3), np.uint8),
            source_timestamp=t,
            embedding_space="arcface-w600k-r50",
        )
        for t in seen_at
    ]
    group = FaceIdentityGroup(group_id=index, observations=observations, representative_embedding=vector, name=name)
    return Person(
        index=index, thumbnail=Image.new("RGB", (16, 16)), detection_count=len(observations),
        first_seen=min(seen_at), last_seen=max(seen_at), group=group, name=name,
    )


def video(name, *people):
    return ScanResult(video_path=Path("/season-1") / name, video_duration=600.0, sample_interval=0.5, people=list(people))


LEAD, FRIEND = face(1, 0, 0), face(0, 1, 0)
# The lead: 10-19.5s in e1, 100-104.5s in e2. The friend: only in e2.
LEAD_E1 = [10 + 0.5 * k for k in range(20)]
LEAD_E2 = [100 + 0.5 * k for k in range(10)]
FRIEND_E2 = [200 + 0.5 * k for k in range(30)]


@pytest.fixture
def season():
    return FolderScan(
        videos=[
            video("e1.mp4", person(0, LEAD, LEAD_E1, name="Lead")),
            video("e2.mp4", person(0, FRIEND, FRIEND_E2), person(1, LEAD, LEAD_E2)),
        ],
        settings=SETTINGS,
    )


def expected(seen_at):
    group = person(0, LEAD, seen_at).group
    return sum(
        i.end_time - i.start_time
        for i in build_appearance_intervals(group, video_duration=600.0, sample_interval=0.5)
    )


def test_each_figure_is_the_screen_time_timestamps_would_report(season):
    report = season_report(season)
    lead = next(row for row in report.rows if row.label == "Lead")

    assert lead.seconds == [pytest.approx(expected(LEAD_E1)), pytest.approx(expected(LEAD_E2))]
    assert lead.episodes == 2
    assert lead.total == pytest.approx(expected(LEAD_E1) + expected(LEAD_E2))


def test_someone_not_in_an_episode_has_nothing_there(season):
    report = season_report(season)
    friend = next(row for row in report.rows if row.label != "Lead")

    assert friend.seconds[0] == 0.0
    assert friend.episodes == 1
    assert friend.named is False


def test_people_are_listed_most_on_screen_first(season):
    totals = [row.total for row in season_report(season).rows]
    assert totals == sorted(totals, reverse=True)


def test_a_recap_counts_in_its_episode_and_is_marked_repeated(season):
    """e2 opens with e1's footage from 0-30s: the lead's e2 appearance at
    100s is new, but add one at 10-19.5s and it repeats e1's exactly."""
    season.videos[1] = video(
        "e2.mp4", person(0, FRIEND, FRIEND_E2), person(1, LEAD, LEAD_E1 + LEAD_E2)
    )
    season.repeats = {(1, 0): [Repeat(0.0, 30.0, 0.0, 60)]}

    lead = next(row for row in season_report(season).rows if row.label == "Lead")

    assert lead.seconds[1] == pytest.approx(expected(LEAD_E1 + LEAD_E2))
    assert lead.repeated == pytest.approx(expected(LEAD_E1))


def test_the_title_is_the_folder(season):
    assert season_report(season).title == "season-1"
    assert season_report(season, title="Series 2").title == "Series 2"


def test_the_csv_has_a_row_per_person_and_a_column_per_video(season, tmp_path):
    path = write_csv(season_report(season), tmp_path / "report.csv")

    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert rows[0] == ["person", "episodes", "total seconds", "repeated seconds", "e1.mp4", "e2.mp4"]
    lead = next(r for r in rows if r[0] == "Lead")
    assert float(lead[4]) == pytest.approx(expected(LEAD_E1), abs=0.05)
    assert float(lead[2]) == pytest.approx(expected(LEAD_E1) + expected(LEAD_E2), abs=0.05)


def test_the_page_is_one_self_contained_file(season, tmp_path):
    path = write_html(season_report(season), tmp_path / "report.html")
    page = path.read_text(encoding="utf-8")

    assert "Lead" in page and "e1" in page and "e2" in page
    assert "data:image/jpeg;base64," in page
    # Opens from disk anywhere: nothing fetched.
    assert "http://" not in page and "https://" not in page
    assert clock(expected(LEAD_E1)) in page


def test_the_page_escapes_names(season, tmp_path):
    season.videos[0] = video("e1.mp4", person(0, LEAD, LEAD_E1, name="<b>Lead</b>"))
    page = write_html(season_report(season), tmp_path / "report.html").read_text()

    assert "<b>Lead</b>" not in page
    assert "&lt;b&gt;Lead&lt;/b&gt;" in page


def test_clock_reads_like_a_running_time():
    assert clock(5) == "0:05"
    assert clock(605.4) == "10:05"
    assert clock(3725) == "1:02:05"


# ------------------------------------------------------------ in the window


web = pytest.importorskip("app.ui.web")


def test_the_window_writes_the_report_beside_the_reels(season, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    bridge = web.Bridge()
    bridge._folder = season
    bridge._rebuild_cast()

    saved = bridge.save_report(str(tmp_path))

    assert saved["saved"] is True
    assert Path(saved["html"]) == tmp_path / "season-1-report.html"
    assert Path(saved["csv"]).is_file()
    assert saved["people"] == 2 and saved["videos"] == 2
    assert opened and opened[0].startswith("file://")


def test_no_report_without_a_folder():
    assert web.Bridge().save_report("/tmp")["saved"] is False
