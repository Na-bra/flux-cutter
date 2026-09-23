"""The cut list: a reel as the cuts it is made of, before anything is encoded.

Everything upstream of here decides what a reel *should* contain --
`build_appearance_intervals` answers when the person was on screen,
`merge_for_export` answers what is worth cutting, `without_repeats` drops
what an earlier episode already showed. Those are rules, and rules are
right most of the time. The reel of a 22-minute episode is a hundred cuts
or more, and a handful of them are wrong in ways no threshold can know:
the cut that opens on the back of someone's head, the one that runs three
seconds past the line, the one that caught the wrong person entirely.

Until now the answer to those was to correct the gallery and scan again,
or to live with them. This module is the third answer: the plan is handed
to the user as a list they can drop from and nudge, and the export cuts
the list rather than re-deriving it.

It holds no footage and opens nothing. A cut is four numbers, and every
operation here returns a new list, so the window can show what an edit
does before committing to it and the rules can be tested without encoding
a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.video.timeline import AppearanceInterval

# A cut may not be shortened below this.
#
# Below about half a second a cut reads as a flash rather than a shot, and
# the sliver trimming that runs during the encode is already working at
# that scale (app/video/cutter.py holds back 12 frames). Someone who wants
# less than half a second of footage wants the cut gone, which is what
# dropping it is for.
MIN_CUT_SECONDS = 0.5

# How far one press moves an edge, in seconds. A frame is offered as well,
# and is not a fixed figure -- it comes from the video's own rate.
STEP_SECONDS = 1.0

START, END = "start", "end"


@dataclass(frozen=True)
class Cut:
    """One segment of the reel, as planned and as it now stands.

    `video` indexes the folder's videos, and is 0 for a single video. The
    planned times are kept beside the current ones so a row can say it was
    changed and put itself back without re-planning the whole reel.
    """

    video: int
    start: float
    end: float
    planned_start: float
    planned_end: float
    dropped: bool = False

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def changed(self) -> bool:
        return (
            self.dropped
            or self.start != self.planned_start
            or self.end != self.planned_end
        )


def cuts_from_plans(plans: list[tuple[int, list[AppearanceInterval]]]) -> list[Cut]:
    """The planned segments of every video, as one numbered list.

    In folder order, which is the order the reel plays in, so a cut's
    position in this list is its position in the finished file.
    """
    return [
        Cut(
            video=video,
            start=segment.start_time,
            end=segment.end_time,
            planned_start=segment.start_time,
            planned_end=segment.end_time,
        )
        for video, segments in plans
        for segment in segments
    ]


def kept(cuts: list[Cut]) -> list[Cut]:
    """The cuts that would be exported, in order."""
    return [cut for cut in cuts if not cut.dropped]


def reel_seconds(cuts: list[Cut]) -> float:
    return sum(cut.seconds for cut in kept(cuts))


def plans_from_cuts(cuts: list[Cut]) -> list[tuple[int, list[AppearanceInterval]]]:
    """Back to per-video segments, for the cutter.

    Dropped cuts are gone and each video's segments are in order, which is
    what `cut_segments` and `cut_clips` require of any list they are given.
    A video every one of whose cuts was dropped does not appear at all,
    rather than appearing with nothing to cut.
    """
    plans: list[tuple[int, list[AppearanceInterval]]] = []
    for cut in kept(cuts):
        if not plans or plans[-1][0] != cut.video:
            plans.append((cut.video, []))
        plans[-1][1].append(AppearanceInterval(start_time=cut.start, end_time=cut.end))
    return plans


def drop(cuts: list[Cut], index: int) -> list[Cut]:
    """Takes one cut out of the reel, or puts it back.

    A dropped cut keeps its times rather than being deleted: it stays in
    the list, greyed, because dropping the wrong row and not being able to
    find it again is worse than the row being there.
    """
    return _replace_at(cuts, index, lambda cut: replace(cut, dropped=not cut.dropped))


def restore(cuts: list[Cut], index: int) -> list[Cut]:
    """Puts one cut back to what was planned for it."""
    return _replace_at(
        cuts,
        index,
        lambda cut: replace(
            cut, start=cut.planned_start, end=cut.planned_end, dropped=False
        ),
    )


def move(
    cuts: list[Cut], index: int, edge: str, delta: float, video_duration: float
) -> list[Cut]:
    """Moves one end of one cut, as far as it is allowed to go.

    Clamped rather than refused, because a press that does nothing and a
    press that does half of what was asked are hard to tell apart at the
    end of a drag, and the end that stops against something is the useful
    thing to see. What stops it:

    * the video's own bounds,
    * the other end of the same cut, never closer than MIN_CUT_SECONDS,
    * the nearest *kept* neighbour in the same video. A neighbour that has
      been dropped is not in the reel, so its footage is free to take
      back -- which is exactly how someone fixes a cut that was split in
      two: drop one half and grow the other over it.

    Touching a neighbour is allowed; overlapping is not, because the
    cutter is promised non-overlapping segments in chronological order.
    """
    if index < 0 or index >= len(cuts):
        return cuts
    cut = cuts[index]
    lower, upper = _room(cuts, index, video_duration)

    if edge == START:
        start = min(max(cut.start + delta, lower), cut.end - MIN_CUT_SECONDS)
        return _replace_at(cuts, index, lambda c: replace(c, start=start))
    if edge == END:
        end = max(min(cut.end + delta, upper), cut.start + MIN_CUT_SECONDS)
        return _replace_at(cuts, index, lambda c: replace(c, end=end))
    raise ValueError(f"unknown edge: {edge!r}")


def _room(cuts: list[Cut], index: int, video_duration: float) -> tuple[float, float]:
    """How far this cut may run in either direction, in its own video."""
    cut = cuts[index]
    lower, upper = 0.0, video_duration
    for position, other in enumerate(cuts):
        if position == index or other.dropped or other.video != cut.video:
            continue
        if other.end <= cut.start:
            lower = max(lower, other.end)
        elif other.start >= cut.end:
            upper = min(upper, other.start)
    return lower, upper


def _replace_at(cuts: list[Cut], index: int, change) -> list[Cut]:
    if index < 0 or index >= len(cuts):
        return cuts
    edited = list(cuts)
    edited[index] = change(edited[index])
    return edited
