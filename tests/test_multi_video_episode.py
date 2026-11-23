"""Videos picked one by one, checked against real episodes.

The bridge's own tests show the plumbing is right: that adding scans only
what was added, that removing takes a video's cards out of the cast. What
they cannot show is that a set of files picked from three different folders
finds the same people, in the same reel, that the folder view finds when
the same three sit in one directory -- which is the promise the videos view
makes, and the only thing that makes it worth having.

So this scans three real episodes once, as a set picked from three folders,
adds the third after the first two are on screen, exports a scene pack
across all three, and then opens the same three as a folder and compares
the two casts. The episodes are symlinked into the scratch folders, and a
kept scan is keyed by the resolved path, so the folder view reuses the
scans the videos view just made rather than paying for them twice.

It scans for real -- about four minutes for the three when this was
written, against three and a half for the whole suite -- which is why it is
marked `measured` and left out of an ordinary run. To take the measurement:

    .venv/bin/python -m pytest tests/test_multi_video_episode.py --measure -s
"""

import json
import time
from pathlib import Path

import pytest

from app.ui.worker import ScanSettings, default_encoder
from app.video.loader import get_video_info, load_video

pytest.importorskip("webview")

from app.ui import web  # noqa: E402

VIDEOS = Path(__file__).resolve().parents[1] / "assets" / "test-videos"
# Three consecutive episodes of one show, which is what makes the cast
# worth looking at: they share characters, and the evaluation labels
# already say which (`evaluation/truth/animation5.json` and its
# neighbours). The third is added after the first two are on screen, so
# it is the one that must be scanned alone.
EPISODES = ["animation5.mp4", "animation6.mp4", "animation7.mp4"]
MODE = "animation"


class PickingWindow:
    """A window that answers the file dialog and swallows the rest."""

    def __init__(self):
        self.chosen = []

    def evaluate_js(self, script):
        pass

    def create_file_dialog(self, *args, **kwargs):
        return self.chosen


def linked(folder: Path, episode: str) -> Path:
    """The episode, reachable from a folder of its own."""
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / episode
    if not link.exists():
        link.symlink_to(VIDEOS / episode)
    return link


def people_of(payload) -> list[tuple[str, int, int]]:
    """Each person as (label, videos they are in, detections)."""
    return sorted((p["label"], p["videos"], p["detections"]) for p in payload["people"])


def seconds_of(path: Path) -> float:
    with load_video(path) as container:
        return get_video_info(container)["duration"] or 0.0


@pytest.mark.measured
def test_three_episodes_picked_from_three_folders(tmp_path, monkeypatch):
    missing = [name for name in EPISODES if not (VIDEOS / name).is_file()]
    if missing:
        pytest.skip(f"not on this Mac: {', '.join(missing)}")

    first = linked(tmp_path / "show-a", EPISODES[0])
    second = linked(tmp_path / "show-b", EPISODES[1])
    third = linked(tmp_path / "show-c", EPISODES[2])

    bridge = web.Bridge()
    window = PickingWindow()
    bridge.window = window
    # A filmstrip is six seeks of footage this test has no use for.
    monkeypatch.setattr(bridge, "_start_cast_preview", lambda: None)
    # Jobs run where they are started, so the measurement happens in order.
    monkeypatch.setattr(bridge, "_start", lambda target, *args, **about: target(*args))

    # Two videos, picked from two folders.
    window.chosen = [str(first), str(second)]
    picked = bridge.choose_videos()
    assert [v["name"] for v in picked["picked"]] == EPISODES[:2]
    started = time.monotonic()
    bridge.start_videos_scan(MODE, 0.5)
    two_seconds = time.monotonic() - started
    two = bridge._cast_payload()
    assert two["videoCount"] == 2
    assert any(p["videos"] == 2 for p in two["people"]), "nobody was linked across both"

    # A third video, added to the set already on screen. Only it is
    # scanned: the other two keep the scans they are holding.
    window.chosen = [str(third)]
    started = time.monotonic()
    added = bridge.choose_videos()
    added_seconds = time.monotonic() - started
    assert added["scanning"] is True
    assert [v["scanned"] for v in added["picked"]] == [True, True, True]
    assert [r.video_path for r in bridge._folder.videos] == [first, second, third]
    three = bridge._cast_payload()
    picked_cast = people_of(three)

    # The lead across all three, and a scene pack cut from the lot.
    lead = max(bridge._cast, key=lambda person: person.detection_count)
    chosen = bridge.select_cast_person(lead.index, "")
    bridge.set_scene_pack("0:30")
    output = tmp_path / f"pack.{bridge._format}"
    started = time.monotonic()
    assert bridge.start_folder_export(
        str(tmp_path), output.name, default_encoder(), "Standard"
    )["started"]
    export_seconds = time.monotonic() - started
    assert output.is_file()

    # A video taken back out again.
    removed = bridge.remove_picked_video(str(third))
    assert removed["removed"] is True

    # The same three as one folder, which is how it had to be done before.
    # Every scan is reused, so this is seconds rather than minutes.
    season = tmp_path / "season"
    for episode in EPISODES:
        linked(season, episode)
    started = time.monotonic()
    bridge._folder_worker(season, ScanSettings.for_mode(MODE))
    folder_seconds = time.monotonic() - started
    folder_cast = people_of(bridge._cast_payload())
    reused = [result.reused for result in bridge._folder.videos]

    print(
        "\n"
        + json.dumps(
            {
                "episodes": EPISODES,
                "two_videos_two_folders": {
                    "people": len(two["people"]),
                    "in_both": sum(1 for p in two["people"] if p["videos"] == 2),
                    "questions": len(two["questions"]),
                    "scan_seconds": round(two_seconds, 1),
                },
                "third_video_added": {
                    "people": len(three["people"]),
                    "in_all_three": sum(1 for p in three["people"] if p["videos"] == 3),
                    "questions": len(three["questions"]),
                    "scan_seconds": round(added_seconds, 1),
                },
                "the_lead": {
                    "name": chosen["name"],
                    "videos": chosen["videos"],
                    "cuts": chosen["cuts"],
                    "reel": chosen["reel"],
                    "on_screen": chosen["onScreen"],
                    "repeated_left_out": chosen["repeated"],
                    "longest_pack": (chosen["maximum"] or {}).get("upTo"),
                },
                "scene_pack": {
                    "asked_for": "0:30",
                    "came_out_seconds": round(seconds_of(output), 1),
                    "megabytes": round(output.stat().st_size / 1e6, 1),
                    "export_seconds": round(export_seconds, 1),
                },
                "after_removing_the_third": {
                    "people": len(removed["people"]),
                    "videos": removed["videoCount"],
                },
                "same_three_as_a_folder": {
                    "people": len(folder_cast),
                    "scans_reused": reused,
                    "scan_seconds": round(folder_seconds, 1),
                    "same_cast_as_picking_them": picked_cast == folder_cast,
                },
            },
            indent=1,
        )
    )

    # The point of the whole thing: picking the files finds what putting
    # them in one folder finds.
    assert picked_cast == folder_cast
    assert all(reused), "the folder view should have reused the scans just made"
    assert removed["videoCount"] == 2
    assert seconds_of(output) > 0
