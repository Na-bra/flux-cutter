"""Tests for the UI's worker layer.

Deliberately no Tkinter here. app/ui/worker.py is the half of the desktop
UI that does the work, and keeping it importable without a display is
what lets these run anywhere the rest of the suite does.
"""

import threading
from pathlib import Path

import numpy as np
import pytest

from app.faces.detector import BoundingBox, FaceDetection
from app.faces.grouper import (
    DEFAULT_CONSOLIDATION_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_MIN_GROUP_EYE_SPAN,
    DEFAULT_SIMILARITY_THRESHOLD,
    FaceIdentityGroup,
    FaceObservation,
)
from app.ui.worker import (
    DEFAULT_QUALITY_LEVEL,
    Cancelled,
    ExportSettings,
    Person,
    ScanSettings,
    _tracked_frames,
    available_encoders,
    default_encoder,
    plan_export,
    quality_for,
)


def observation(timestamp: float) -> FaceObservation:
    return FaceObservation(
        embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
        detection=FaceDetection(
            box=BoundingBox(x_min=0, y_min=0, x_max=80, y_max=80),
            confidence=0.9,
        ),
        face_crop=np.zeros((80, 80, 3), dtype=np.uint8),
        source_timestamp=timestamp,
    )


def person_at(*timestamps: float) -> Person:
    group = FaceIdentityGroup(
        group_id=0,
        observations=[observation(t) for t in timestamps],
    )
    return Person(
        index=0,
        thumbnail=None,
        detection_count=len(group.observations),
        first_seen=min(timestamps),
        last_seen=max(timestamps),
        group=group,
    )


def frames(count: int):
    for index in range(count):
        yield float(index), np.zeros((4, 4, 3), dtype=np.uint8)


def test_tracked_frames_passes_every_frame_through_untouched():
    """The wrapper observes the stream; it must not filter or reorder it."""
    wrapped = list(_tracked_frames(frames(5), total_frames=5, cancel=None, report=None))

    assert [timestamp for timestamp, _ in wrapped] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_tracked_frames_reports_progress_as_a_fraction():
    reported = []
    list(
        _tracked_frames(
            frames(4), total_frames=4, cancel=None, report=lambda f, t: reported.append((f, t))
        )
    )

    assert reported == [(0.25, 0.0), (0.5, 1.0), (0.75, 2.0), (1.0, 3.0)]


def test_progress_fraction_is_capped_when_the_estimate_runs_short():
    """Duration-derived frame counts are an estimate, not a guarantee.

    A container that yields more frames than duration/interval predicted
    would otherwise drive the progress bar past full.
    """
    reported = []
    list(
        _tracked_frames(
            frames(6), total_frames=4, cancel=None, report=lambda f, t: reported.append(f)
        )
    )

    assert max(reported) == 1.0


def test_progress_is_zero_when_the_duration_is_unknown():
    """Some containers report no duration; there is then nothing to divide by."""
    reported = []
    list(
        _tracked_frames(
            frames(3), total_frames=0, cancel=None, report=lambda f, t: reported.append(f)
        )
    )

    assert reported == [0.0, 0.0, 0.0]


def test_cancelling_stops_the_stream():
    cancel = threading.Event()
    seen = []

    def report(_fraction, timestamp):
        seen.append(timestamp)
        if len(seen) == 3:
            cancel.set()

    with pytest.raises(Cancelled):
        list(_tracked_frames(frames(100), total_frames=100, cancel=cancel, report=report))

    assert len(seen) == 3


def test_cancellation_is_checked_before_the_frame_is_handed_on():
    """Setting the event before iteration starts must yield nothing at all."""
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(Cancelled):
        list(_tracked_frames(frames(5), total_frames=5, cancel=cancel, report=None))


def test_plan_export_turns_one_person_into_cuttable_segments():
    intervals, segments = plan_export(
        person_at(10.0, 10.5, 11.0, 40.0, 40.5),
        video_duration=100.0,
        sample_interval=0.5,
    )

    assert len(intervals) == 2
    # Two appearances 29s apart stay apart; each is grown to the minimum
    # segment length rather than cut as a sub-second flash.
    assert len(segments) == 2
    assert all(s.end_time - s.start_time >= 2.0 for s in segments)


def test_plan_export_bridges_a_brief_absence():
    _, segments = plan_export(
        person_at(10.0, 10.5, 11.0, 12.0, 12.5),
        video_duration=100.0,
        sample_interval=0.5,
    )

    assert len(segments) == 1


def test_scan_settings_default_to_the_cli_defaults():
    """The UI and the CLI must not drift into different results.

    Both drive run_identity_pipeline, so a default that disagreed here
    would silently produce a different set of people from the same video
    depending on which front end asked.
    """
    settings = ScanSettings()

    assert settings.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert settings.margin_threshold == DEFAULT_MARGIN_THRESHOLD
    assert settings.consolidation_threshold == DEFAULT_CONSOLIDATION_THRESHOLD
    assert settings.min_confidence == DEFAULT_MIN_CONFIDENCE
    assert settings.min_face_size == DEFAULT_MIN_FACE_SIZE
    assert settings.min_group_eye_span == DEFAULT_MIN_GROUP_EYE_SPAN
    assert settings.min_detections is None


def test_export_settings_default_to_a_portable_encoder():
    """The window may offer videotoolbox, but the setting itself stays portable."""
    assert ExportSettings().video_encoder == "libx264"
    assert ExportSettings().include_audio is True


def test_quality_levels_translate_per_encoder():
    """The two encoders' scales run in opposite directions.

    -crf is 0-51 and lower is better; -q:v is 0-100 and higher is better.
    Passing one number to both silently produced a 1.7 Mbps videotoolbox
    file where libx264 gave 13.9 Mbps on the same clip.
    """
    assert quality_for("libx264", "Maximum") < quality_for("libx264", "Standard")
    assert quality_for("h264_videotoolbox", "Maximum") > quality_for(
        "h264_videotoolbox", "Standard"
    )


def test_unknown_encoder_gets_the_crf_style_number():
    """crf is the common convention; videotoolbox is the one special case."""
    assert quality_for("libsvtav1", "High") == quality_for("libx264", "High")


def test_unknown_quality_level_falls_back_to_the_default():
    assert quality_for("libx264", "nonsense") == quality_for(
        "libx264", DEFAULT_QUALITY_LEVEL
    )


def test_available_encoders_never_comes_back_empty():
    """libx264 is the floor; the export dropdown must always have an option."""
    assert available_encoders()
    assert "libx264" in available_encoders()


def test_available_encoders_are_offered_best_first():
    """A hardware encoder, where present, outranks libx264 by about 5x."""
    encoders = available_encoders()

    assert encoders[-1] == "libx264"
    assert default_encoder() == encoders[0]


def test_every_offered_encoder_can_actually_be_constructed():
    """The list is a promise the export will not fail on.

    The dropdown used to hardcode h264_videotoolbox, which does not exist
    off macOS -- a Windows user could pick an encoder that then failed at
    encode time.
    """
    from av.codec import Codec

    for name in available_encoders():
        Codec(name, "w")


# ------------------------------------------------------- reusing a scan


class FakePipelineResult:
    """What `_scan_footage` hands back, without any footage behind it."""

    def __init__(self, groups):
        self.grouper = type(
            "Grouper", (), {"groups": groups, "unassigned": [1, 2]}
        )()
        self.total_detections = 9
        self.track_count = 4
        self.frame_count = 120
        self.last_timestamp = 119.0
        self.embedding_time = 1.0
        self.grouping_time = 0.5


@pytest.fixture
def footage(tmp_path):
    """A file that is a video as far as everything here is concerned.

    Nothing decodes it: the scan under test is monkeypatched out, and
    VideoSource only validates and opens the path.
    """
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"pretend footage")
    return path


@pytest.fixture
def scanned(monkeypatch):
    """Counts how many times the footage is actually scanned."""
    from app.ui import worker as worker_module

    calls = []

    def fake_scan_footage(source, settings, cancel, on_progress):
        calls.append(settings)
        crop = np.full((4, 4, 3), 3, dtype=np.uint8)
        group = FaceIdentityGroup(group_id=1, observations=[observation(1.0)])
        group.representative_embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        group.representative_observation = FaceObservation(
            embedding=group.representative_embedding,
            detection=FaceDetection(box=BoundingBox(0, 0, 4, 4), confidence=0.9),
            face_crop=crop,
            source_timestamp=1.0,
            frame_index=1,
        )
        group.observations = [group.representative_observation]
        return FakePipelineResult([group]), 120.0, 3

    monkeypatch.setattr(worker_module, "_scan_footage", fake_scan_footage)
    monkeypatch.setattr(worker_module, "fetch_models", lambda **kwargs: None)
    return calls


def test_a_second_scan_of_the_same_video_reuses_the_first(footage, scanned):
    from app.ui.worker import scan

    first = scan(footage)
    first.close()
    second = scan(footage)
    second.close()

    assert len(scanned) == 1
    assert first.reused is False
    assert second.reused is True
    assert [p.detection_count for p in second.people] == [
        p.detection_count for p in first.people
    ]


def test_a_reused_scan_reports_what_the_original_cost(footage, scanned):
    from app.ui.worker import scan

    scan(footage).close()
    reused = scan(footage)
    reused.close()

    assert reused.original_seconds > 0
    assert reused.frame_count == 120
    assert reused.detection_count == 9
    assert reused.min_detections == 3


def test_different_settings_are_a_different_scan(footage, scanned):
    from app.ui.worker import scan

    scan(footage, ScanSettings(sample_interval=0.5)).close()
    scan(footage, ScanSettings(sample_interval=1.0)).close()

    assert len(scanned) == 2


def test_a_rescan_can_be_demanded(footage, scanned):
    from app.ui.worker import scan

    scan(footage).close()
    again = scan(footage, use_cache=False)
    again.close()

    assert len(scanned) == 2
    assert again.reused is False


def test_the_models_are_not_fetched_for_a_scan_that_will_be_reused(
    footage, scanned, monkeypatch
):
    """A reused scan detects and embeds nothing, so a machine that has
    never downloaded the models can still open a video it already knows."""
    from app.ui import worker as worker_module

    scan_module = worker_module.scan
    scan_module(footage).close()

    def refuse(**kwargs):
        raise AssertionError("the models were fetched for a reused scan")

    monkeypatch.setattr(worker_module, "fetch_models", refuse)
    reused = scan_module(footage)
    reused.close()

    assert reused.reused is True


# --------------------------------------------------- more than one person


def one_person(index, timestamps):
    group = FaceIdentityGroup(
        group_id=index + 1,
        observations=[observation(t) for t in timestamps],
    )
    return Person(
        index=index,
        thumbnail=None,
        detection_count=len(timestamps),
        first_seen=min(timestamps),
        last_seen=max(timestamps),
        group=group,
    )


def test_one_person_is_their_own_group_untouched():
    from app.ui.worker import combined_group

    alone = one_person(0, [1.0, 2.0])

    assert combined_group([alone]) is alone.group


def test_two_people_pool_their_detections():
    from app.ui.worker import combined_group

    combined = combined_group([one_person(0, [1.0, 2.0]), one_person(1, [5.0, 6.0])])

    assert len(combined.observations) == 4
    assert sorted(o.source_timestamp for o in combined.observations) == [
        1.0,
        2.0,
        5.0,
        6.0,
    ]


def test_a_combined_group_claims_no_identity():
    """It is a carrier for a timeline, not a person: a group of two people
    has no centroid and no representative face."""
    from app.ui.worker import combined_group

    combined = combined_group([one_person(0, [1.0]), one_person(1, [2.0])])

    assert combined.representative_embedding is None
    assert combined.representative_observation is None


def test_combining_needs_somebody():
    from app.ui.worker import combined_group

    with pytest.raises(ValueError, match="no people"):
        combined_group([])


def test_a_handover_between_two_people_becomes_one_appearance():
    """One lead leaves and the other arrives a second later. On the pooled
    timeline that is one continuous appearance; merging two separately
    built interval lists would have cut it in two and padded both halves."""
    from app.ui.worker import plan_export

    leaving = one_person(0, [10.0, 11.0, 12.0])
    arriving = one_person(1, [13.0, 14.0, 15.0])

    intervals, _ = plan_export(
        [leaving, arriving], video_duration=60.0, sample_interval=1.0
    )

    assert len(intervals) == 1
    assert intervals[0].start_time < 10.0
    assert intervals[0].end_time > 15.0


def test_people_who_never_share_a_scene_keep_separate_appearances():
    from app.ui.worker import plan_export

    early = one_person(0, [1.0, 2.0])
    late = one_person(1, [50.0, 51.0])

    intervals, _ = plan_export([early, late], video_duration=60.0, sample_interval=1.0)

    assert len(intervals) == 2


# ------------------------------------------------------------- the preview


class FakeSource:
    def __init__(self):
        self.opened = 0

    def open(self):
        self.opened += 1
        raise RuntimeError("no footage behind this")


def a_result(people, source=None, duration=600.0):
    from app.ui.worker import ScanResult

    return ScanResult(
        video_path=Path("/videos/episode.mp4"),
        video_duration=duration,
        sample_interval=1.0,
        people=people,
        source=source,
    )


def test_a_preview_needs_somebody_and_some_footage():
    from app.ui.worker import preview_frames

    person = one_person(0, [1.0, 2.0])

    assert preview_frames(a_result([person], source=FakeSource()), []) == []
    assert preview_frames(a_result([person], source=None), [person]) == []


def test_a_preview_that_cannot_be_drawn_is_not_an_error():
    """It is a convenience. Failing to draw one must never stop an export
    that would otherwise work."""
    from app.ui.worker import preview_frames

    person = one_person(0, [1.0, 2.0, 3.0])
    source = FakeSource()

    assert preview_frames(a_result([person], source=source), [person]) == []
    assert source.opened == 1


def test_the_filmstrip_is_spread_across_the_whole_reel(monkeypatch):
    """Not the first six cuts: a 100-cut reel has to be represented by its
    length, or the preview only ever shows its opening minute."""
    from app.ui import worker as worker_module

    asked = []

    def fake_frames_at(source, timestamps, width):
        asked.extend(timestamps)
        return []

    monkeypatch.setattr(worker_module, "_frames_at", fake_frames_at)
    spread = one_person(0, [float(t) for t in range(0, 600, 20)])

    worker_module.preview_frames(
        a_result([spread], source=FakeSource()), [spread], limit=4
    )

    assert len(asked) == 4
    assert asked == sorted(asked)
    assert asked[0] < 60 and asked[-1] > 500


def test_a_short_reel_shows_every_cut_it_has(monkeypatch):
    from app.ui import worker as worker_module

    asked = []
    monkeypatch.setattr(
        worker_module,
        "_frames_at",
        lambda source, timestamps, width: asked.extend(timestamps) or [],
    )
    brief = one_person(0, [1.0, 2.0, 100.0, 101.0])

    worker_module.preview_frames(
        a_result([brief], source=FakeSource()), [brief], limit=6
    )

    assert len(asked) == 2


def test_frames_come_from_inside_the_cuts_not_their_edges(monkeypatch):
    """A cut's first frame often lands mid-transition and shows a face
    nobody would recognise."""
    from app.ui import worker as worker_module

    asked = []
    monkeypatch.setattr(
        worker_module,
        "_frames_at",
        lambda source, timestamps, width: asked.extend(timestamps) or [],
    )
    person = one_person(0, [10.0, 11.0, 12.0])

    intervals, segments = worker_module.plan_export(
        [person], video_duration=600.0, sample_interval=1.0
    )
    worker_module.preview_frames(a_result([person], source=FakeSource()), [person])

    assert len(asked) == 1
    assert segments[0].start_time < asked[0] < segments[0].end_time
