"""Finding one person across a folder of videos, without a photo of them.

A name given to a card in one video is enough: the scans where somebody
named them use those cards, and every other video is searched for the same
face. None of this decodes footage -- the scans, the plans and the cut are
stand-ins, and each is tested where it lives.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import main as app_main
from app.faces.grouper import FaceIdentityGroup, FaceObservation
from app.faces.reference import ReferenceError, reference_from_groups
from app.main import ExportPlan, SelectionError, find_person, run_batch
from app.scans import CachedScan
from app.video.cutter import CutResult, CutterError
from app.video.timeline import AppearanceInterval

SPACE = "arcface-w600k-r50"


def card(vector, name=None, space=SPACE):
    vector = np.asarray(vector, dtype=np.float32)
    observation = FaceObservation(
        embedding=vector,
        detection=None,
        face_crop=None,
        source_timestamp=0.0,
        embedding_space=space,
    )
    return FaceIdentityGroup(
        group_id=0,
        observations=[observation],
        representative_embedding=vector,
        representative_observation=observation,
        name=name,
    )


def video(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"pretend footage")
    return path


@contextmanager
def fake_container(path):
    yield object()


# --------------------------------------------------- a face from named cards


def test_named_cards_become_one_face_to_look_for():
    reference = reference_from_groups(
        [card([1.0, 0.0], "Jamie"), card([0.0, 1.0], "Jamie")], label="Jamie"
    )

    assert reference.name == "Jamie"
    assert reference.source is None
    assert reference.embedding_space == SPACE
    np.testing.assert_allclose(np.linalg.norm(reference.embedding), 1.0, atol=1e-6)
    np.testing.assert_allclose(reference.embedding, [0.7071, 0.7071], atol=1e-3)


def test_a_card_seen_for_longer_does_not_outweigh_one_seen_briefly():
    """Each card is normalised before averaging, so the person's look across
    episodes counts, not whichever episode they filled."""
    reference = reference_from_groups(
        [card([10.0, 0.0], "Jamie"), card([0.0, 0.1], "Jamie")], label="Jamie"
    )
    assert reference.embedding[0] == pytest.approx(reference.embedding[1], abs=1e-6)


def test_cards_from_different_modes_cannot_be_combined():
    with pytest.raises(ReferenceError, match="different modes"):
        reference_from_groups(
            [card([1.0, 0.0], "Jamie"), card([0.0, 1.0], "Jamie", space="ccip")],
            label="Jamie",
        )


def test_cards_with_no_faces_are_refused():
    empty = FaceIdentityGroup(group_id=0, name="Jamie")
    with pytest.raises(ReferenceError):
        reference_from_groups([empty], label="Jamie")


# ------------------------------------------------- finding the name in scans


def kept_scans(monkeypatch, by_video: dict[str, list[FaceIdentityGroup] | None]):
    """Pretends these are the kept scans, keyed by video file name."""

    def find(path, **settings):
        groups = by_video.get(Path(path).name)
        return "key", (None if groups is None else CachedScan(groups=groups))

    monkeypatch.setattr(app_main, "find_scan", find)


def test_a_person_named_in_one_video_is_looked_for_in_all(tmp_path, monkeypatch, capsys):
    videos = [video(tmp_path, "e1.mp4"), video(tmp_path, "e2.mp4"), video(tmp_path, "e3.mp4")]
    kept_scans(
        monkeypatch,
        {"e1.mp4": [card([1.0, 0.0], "Jamie Lee"), card([0.0, 1.0], "Sam")], "e2.mp4": []},
    )

    reference = find_person(videos, "jamie lee", settings={})

    # Reported as it was saved, not as it was typed.
    assert reference.name == "Jamie Lee"
    np.testing.assert_allclose(reference.embedding, [1.0, 0.0], atol=1e-6)
    assert "named in 1 of 3 videos; finding them in the other 2 by face" in capsys.readouterr().out


def test_nobody_by_that_name_is_said_before_anything_is_scanned(tmp_path, monkeypatch):
    videos = [video(tmp_path, "e1.mp4")]
    kept_scans(monkeypatch, {"e1.mp4": [card([1.0, 0.0], "Sam"), card([0.0, 1.0], "Alex")]})

    with pytest.raises(SelectionError) as refused:
        find_person(videos, "Jamie", settings={})

    message = str(refused.value)
    assert "Nobody is named 'Jamie'" in message
    assert "Alex, Sam" in message


def test_videos_never_scanned_have_no_names_to_offer(tmp_path, monkeypatch):
    videos = [video(tmp_path, "e1.mp4")]
    kept_scans(monkeypatch, {})

    with pytest.raises(SelectionError, match="give it that name"):
        find_person(videos, "Jamie", settings={})


def test_only_the_scan_settings_reach_the_cache_lookup(tmp_path, monkeypatch):
    """The key is built from scan settings alone; an encoder choice in the
    same dictionary would make every lookup miss."""
    videos = [video(tmp_path, "e1.mp4")]
    seen = {}

    def find(path, **settings):
        seen.update(settings)
        return "key", CachedScan(groups=[card([1.0, 0.0], "Jamie")])

    monkeypatch.setattr(app_main, "find_scan", find)
    find_person(
        videos,
        "Jamie",
        settings={"mode": "live", "sample_interval": 0.5, "video_encoder": "libx264"},
    )

    assert seen == {"mode": "live", "sample_interval": 0.5}


# ------------------------------------------------------------ separate reels


def test_each_video_gets_the_name_and_the_face(tmp_path, monkeypatch):
    """Where they are named the name wins; elsewhere the face is matched."""
    folder = tmp_path / "season"
    video(folder, "e1.mp4")
    video(folder, "e2.mp4")
    kept_scans(monkeypatch, {"e1.mp4": [card([1.0, 0.0], "Jamie")]})
    monkeypatch.setattr(app_main, "load_video", fake_container)
    seen = []

    def export(container, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(output_path=kwargs["output_path"], exported_seconds=5.0)

    monkeypatch.setattr(app_main, "run_export", export)

    run_batch([folder], None, tmp_path / "reels", export_settings={}, person="Jamie")

    assert [kwargs["select_name"] for kwargs in seen] == ["Jamie", "Jamie"]
    assert all(kwargs["reference"].name == "Jamie" for kwargs in seen)


def test_rescan_reaches_every_video(tmp_path, monkeypatch):
    """batch accepted --rescan and never passed it on, so it always reused."""
    folder = tmp_path / "season"
    video(folder, "e1.mp4")
    monkeypatch.setattr(app_main, "load_video", fake_container)
    seen = []

    def export(container, **kwargs):
        seen.append(kwargs["use_cache"])
        return None

    monkeypatch.setattr(app_main, "run_export", export)
    reference = reference_from_groups([card([1.0, 0.0])], label="them")

    run_batch([folder], reference, tmp_path / "reels", export_settings={}, use_cache=False)

    assert seen == [False]


def test_no_way_of_naming_someone_is_refused(tmp_path):
    video(tmp_path, "e1.mp4")
    with pytest.raises(SelectionError):
        run_batch([tmp_path], None, tmp_path / "reels", export_settings={})


# ------------------------------------------------------------ one joined reel


def span(start, end):
    return AppearanceInterval(start, end)


@pytest.fixture
def season(tmp_path, monkeypatch):
    """Four episodes with stand-in scans, plans, probes and a cut.

    e2 cannot be read, e3 holds nobody to cut, and e4 is at another frame
    rate. The cut records the clips it was given.
    """
    folder = tmp_path / "season"
    for name in ("e1.mp4", "e2.mp4", "e3.mp4", "e4.mp4", "e5.mp4"):
        video(folder, name)

    reference = reference_from_groups([card([1.0, 0.0])], label="Jamie")
    state = SimpleNamespace(clips=None, fail_cut=None, rates={"e4.mp4": 25})

    @contextmanager
    def load(path):
        if Path(path).name == "e2.mp4":
            raise app_main.VideoLoadError("could not open e2")
        yield object()

    def scan(container, video_path, **kwargs):
        return CachedScan()

    def plan(scan, video_path, **kwargs):
        if video_path.name == "e3.mp4":
            return None
        return ExportPlan(
            video_path=video_path,
            selection_name="Jamie",
            selected_group=card([1.0, 0.0]),
            intervals=[],
            segments=[span(1.0, 3.0), span(5.0, 6.0)],
        )

    def probe(path, include_audio=True):
        return SimpleNamespace(frame_rate=state.rates.get(Path(path).name, 24))

    def cut(clips, output_path, **kwargs):
        if state.fail_cut:
            raise CutterError(state.fail_cut)
        state.clips = clips
        return CutResult(
            output_path=output_path,
            segment_count=sum(len(c.segments) for c in clips),
            exported_seconds=3.0 * len(clips),
            encode_seconds=1.0,
            clip_seconds=tuple(3.0 for _ in clips),
        )

    monkeypatch.setattr(app_main, "load_video", load)
    monkeypatch.setattr(app_main, "scan_or_reuse", scan)
    monkeypatch.setattr(app_main, "plan_export", plan)
    monkeypatch.setattr(app_main, "probe_clip", probe)
    monkeypatch.setattr(app_main, "cut_clips", cut)
    return SimpleNamespace(folder=folder, reference=reference, state=state, tmp=tmp_path)


def test_a_season_becomes_one_reel_in_episode_order(season):
    reel = season.tmp / "jamie.mp4"
    outcomes = run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=reel,
    )

    assert [Path(c.video).name for c in season.state.clips] == ["e1.mp4", "e5.mp4"]
    assert [o.video_path.name for o in outcomes] == [
        "e1.mp4", "e2.mp4", "e3.mp4", "e4.mp4", "e5.mp4",
    ]
    assert [o.succeeded for o in outcomes] == [True, False, False, False, True]
    assert {o.output_path for o in outcomes if o.succeeded} == {reel}
    assert outcomes[0].reel_seconds == 3.0


def test_every_video_that_is_left_out_says_why(season):
    outcomes = run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=season.tmp / "jamie.mp4",
    )
    reasons = {o.video_path.name: o.skipped_because for o in outcomes if not o.succeeded}

    assert "could not open e2" in reasons["e2.mp4"]
    assert reasons["e3.mp4"] == "nothing to cut for this person"
    assert "25.000fps" in reasons["e4.mp4"] and "24.000fps" in reasons["e4.mp4"]


def test_the_first_contributing_video_sets_the_frame_rate(season):
    """Not the first video in the folder -- that one may have nothing to cut."""
    season.state.rates = {"e1.mp4": 25, "e4.mp4": 25}
    run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=season.tmp / "jamie.mp4",
    )
    assert [Path(c.video).name for c in season.state.clips] == ["e1.mp4", "e4.mp4"]


def test_a_failed_cut_is_reported_against_every_video_that_was_in_it(season):
    season.state.fail_cut = "disk full"
    outcomes = run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=season.tmp / "jamie.mp4",
    )

    assert not any(o.succeeded for o in outcomes)
    assert "disk full" in outcomes[0].skipped_because
    assert "disk full" in outcomes[4].skipped_because


def test_nothing_to_cut_anywhere_writes_nothing(season, monkeypatch, capsys):
    monkeypatch.setattr(app_main, "plan_export", lambda *a, **k: None)
    reel = season.tmp / "jamie.mp4"

    outcomes = run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=reel,
    )

    assert season.state.clips is None
    assert not any(o.succeeded for o in outcomes)
    assert "was not written" in capsys.readouterr().out


def test_settings_are_shared_out_to_the_stage_that_uses_them(season, monkeypatch):
    seen = {}

    def scan(container, video_path, **kwargs):
        seen.setdefault("scan", kwargs)
        return CachedScan()

    def plan(scan, video_path, **kwargs):
        seen.setdefault("plan", kwargs)
        return None

    monkeypatch.setattr(app_main, "scan_or_reuse", scan)
    monkeypatch.setattr(app_main, "plan_export", plan)

    run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={"mode": "live", "bridge_gap_seconds": 2.0, "quality": 20},
        combine_path=season.tmp / "jamie.mp4", use_cache=False,
    )

    assert seen["scan"] == {"mode": "live", "use_cache": False}
    assert set(seen["plan"]) == {
        "mode", "bridge_gap_seconds", "reference", "select_name", "reference_threshold",
    }


# ----------------------------------------------- choosing within one scan


def gallery(*groups):
    return SimpleNamespace(groups=list(groups))


def test_a_scan_where_they_are_named_uses_the_named_cards():
    """Even when the face would have matched someone else: the user's own
    naming is the stronger evidence."""
    reference = reference_from_groups([card([1.0, 0.0])], label="Jamie")
    chosen = app_main._resolve_selection(
        gallery(card([1.0, 0.0]), card([0.0, 1.0], "Jamie")), None, reference,
        select_name="Jamie",
    )
    assert chosen == [1]


def test_a_scan_where_they_are_not_named_is_searched_by_face(capsys):
    reference = reference_from_groups([card([1.0, 0.0])], label="Jamie")
    chosen = app_main._resolve_selection(
        gallery(card([0.0, 1.0]), card([0.99, 0.14])), None, reference,
        reference_threshold=0.5, select_name="Jamie",
    )
    assert chosen == [1]
    assert "Jamie matched Person #2" in capsys.readouterr().out


def test_two_equally_likely_faces_skip_the_video_rather_than_guess():
    reference = reference_from_groups([card([1.0, 0.0])], label="Jamie")
    with pytest.raises(SelectionError, match="Jamie matches two people"):
        app_main._resolve_selection(
            gallery(card([0.9, 0.1]), card([0.9, -0.1])), None, reference,
            reference_threshold=0.5, select_name="Jamie",
        )


def test_a_name_alone_still_refuses_a_scan_without_it():
    with pytest.raises(SelectionError, match="Nobody in this scan is called"):
        app_main._resolve_selection(
            gallery(card([1.0, 0.0], "Sam")), None, None, select_name="Jamie"
        )


# ------------------------------------------------------------ repeats


def with_repeats(monkeypatch, found):
    """Every video fingerprints; `found` says what repeats what, by name."""
    from app.video.repeats import Repeat

    monkeypatch.setattr(app_main, "fingerprint_kept", lambda path, directory: Path(path).name)
    monkeypatch.setattr(
        app_main, "find_repeats",
        lambda later, earlier: [Repeat(*r) for r in found.get((later, earlier), [])],
    )


def test_a_combined_reel_leaves_out_what_a_later_video_repeats(season, monkeypatch, capsys):
    # e5 repeats e1 from 0-10s at no offset: e5's 1-3s and 5-6s are e1's.
    with_repeats(monkeypatch, {("e5.mp4", "e1.mp4"): [(0.0, 10.0, 0.0, 20)]})

    outcomes = run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=season.tmp / "jamie.mp4",
    )

    reasons = {o.video_path.name: o.skipped_because for o in outcomes if not o.succeeded}
    assert "already in the reel" in reasons["e5.mp4"]
    assert [Path(c.video).name for c in season.state.clips] == ["e1.mp4"]


def test_keep_repeats_keeps_them(season, monkeypatch):
    with_repeats(monkeypatch, {("e5.mp4", "e1.mp4"): [(0.0, 10.0, 0.0, 20)]})

    run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=season.tmp / "jamie.mp4", keep_repeats=True,
    )

    assert "e5.mp4" in [Path(c.video).name for c in season.state.clips]


def test_part_of_a_video_repeated_is_trimmed_and_the_summary_says_so(season, monkeypatch, capsys):
    # Only e5's 1-3s repeats e1's 1-3s.
    with_repeats(monkeypatch, {("e5.mp4", "e1.mp4"): [(0.5, 3.5, 0.0, 7)]})

    run_batch(
        [season.folder], season.reference, season.tmp / "unused",
        export_settings={}, combine_path=season.tmp / "jamie.mp4",
    )

    e5 = next(c for c in season.state.clips if Path(c.video).name == "e5.mp4")
    assert e5.segments == [span(5.0, 6.0)]
    assert "left out 2.0s already in the reel" in capsys.readouterr().out
