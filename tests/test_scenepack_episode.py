"""The longest scene pack, checked against a whole episode.

The figure beside the length box -- "up to 3:12" -- is only as good as the
shots and the faces it rests on, and both come out of a real scan. A
synthetic two-minute clip can show that the arithmetic holds; it cannot
show that the answer about 44 minutes of television is the truth. So this
scans `assets/test-videos/test_3.mp4` once and checks the three things the
figure promises: that it is there as soon as someone is chosen, that asking
for exactly it gives a pack within the usual slack of it, and that asking
for more is short by the same figure.

It scans the episode for real -- 2 min 16 s when this was written, against
three and a half minutes for the whole suite -- which is why it is marked
`measured` and left out of an ordinary run. To take the measurement:

    .venv/bin/python -m pytest tests/test_scenepack_episode.py --measure -s
"""

import json
from pathlib import Path

import pytest

from app.ui.worker import ScanSettings, scan
from app.video import scenepack

pytest.importorskip("webview")

from app.ui import web  # noqa: E402

EPISODE = Path(__file__).resolve().parents[1] / "assets" / "test-videos" / "test_3.mp4"


class QuietWindow:
    """A window that swallows what the bridge tells the page."""

    def evaluate_js(self, script):
        pass


@pytest.mark.measured
def test_the_longest_pack_of_the_lead_of_an_episode(monkeypatch, capsys):
    if not EPISODE.is_file():
        pytest.skip(f"no episode at {EPISODE}")

    settings = ScanSettings.for_mode("live")
    result = scan(EPISODE, settings=settings)
    assert result.shots, "the scan should have recorded where the camera cuts"

    bridge = web.Bridge()
    bridge.window = QuietWindow()
    bridge._scan_result = result
    bridge._scan_settings = settings
    # A filmstrip is six seeks of footage this test has no use for.
    monkeypatch.setattr(bridge, "_start_preview", lambda: None)

    lead = max(result.people, key=lambda person: person.detection_count)
    chosen = bridge.select_person(lead.index, "")
    longest = chosen["maximum"]["upTo"]
    assert "waiting" not in chosen["maximum"], "the scan carries its own shots"

    # Exactly the figure: a pack within the slack of it, and not called short.
    asked = scenepack.parse_length(longest)
    exactly = bridge.set_scene_pack(longest)
    came_out = scenepack.expected_seconds(bridge._cuts, bridge._pack_edge_loss)
    assert exactly["pack"]["short"] is None
    assert abs(came_out - asked) <= scenepack.slack(asked)

    # More than there is: short by the same figure, which is the number the
    # panel now says up front instead.
    more = bridge.set_scene_pack(web._clock(asked * 2))
    assert more["pack"]["short"] == longest

    print(
        "\n"
        + json.dumps(
            {
                "episode": EPISODE.name,
                "minutes": round(result.video_duration / 60, 1),
                "people": len(result.people),
                "lead": f"#{lead.index + 1}",
                "lead_detections": lead.detection_count,
                "longest_pack": longest,
                "pieces_available": len(bridge._pack_pieces),
                "asked_for": longest,
                "pack_came_out": web._clock(came_out),
                "pack_pieces": exactly["kept"],
                "asked_for_more": web._clock(asked * 2),
                "said_short_of": more["pack"]["short"],
            },
            indent=1,
        )
    )
