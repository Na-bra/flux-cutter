"""Correcting the identities a scan produced.

Grouping is tuned against real footage and gets most of it right, but the
README says plainly what it still does: one actor can land on two cards,
and a logo or the back of a head can survive the non-face filter as a
convincing phantom person. Until now the only recourse was to re-tune
thresholds on the command line and scan the whole video again -- minutes
of work to fix something the user could see at a glance.

These are the three corrections that a person can make and the clustering
cannot:

- **Merge**, when one person came back as two cards. This is the common
  one, and the failure the consolidation pass already exists to catch.
- **Split**, when two people were pooled into one. It splits along track
  boundaries and nowhere else, because a track is the only unit here that
  is provably one person -- spatial continuity across consecutive frames
  proves it, and nothing else in this module does.
- **Discard**, for the phantom identities that are not people at all.

Every operation returns a new list, ordered and renumbered the way the
gallery orders it: largest first, ties broken by first appearance. That
is the same rule `_build_groups` applies, so an edited gallery is indexed
the way an unedited one is and `--select-index 0` keeps meaning "the one
at the top".

Nothing here re-clusters. These are corrections to a clustering that has
already run, applied on top of it.
"""

from dataclasses import replace

from app.faces.grouper import (
    FaceIdentityGroup,
    _observation_quality,
    mean_embedding,
)


class EditError(Exception):
    """Raised when an edit does not describe something that can be done."""


def _recompute(group: FaceIdentityGroup) -> FaceIdentityGroup:
    """Refreshes a group's centroid and cover picture from its members.

    The same rule the grouper uses, so an edited group is represented the
    way a clustered one is: the centroid is the mean over every member, and
    the cover is whichever observation best stands for it.
    """
    if not group.observations:
        group.representative_embedding = None
        group.representative_observation = None
        return group

    group.representative_embedding = mean_embedding(
        [observation.embedding for observation in group.observations]
    )
    centroid = group.representative_embedding
    group.representative_observation = max(
        group.observations,
        key=lambda observation: _observation_quality(observation, centroid),
    )
    return group


def _ordered(groups: list[FaceIdentityGroup]) -> list[FaceIdentityGroup]:
    """Largest first, renumbered -- the order the gallery is indexed by.

    Numbering produces new group objects rather than stamping an id onto
    the ones it was handed. Groups survive an edit -- a discard keeps every
    other card exactly as it was -- so renumbering in place reached back
    into the caller's gallery and into the results of any earlier edit that
    still held the same object, silently renumbering both.
    """
    ordered = sorted(
        (group for group in groups if group.observations),
        key=lambda group: (
            -len(group.observations),
            min(o.source_timestamp for o in group.observations),
        ),
    )
    numbered = [
        replace(group, group_id=position)
        for position, group in enumerate(ordered, start=1)
    ]
    # A group with no cover picture is dropped by the gallery, so an edit
    # that left one that way would make a card the user did not touch
    # disappear. Untouched groups from a scan always have one; this is
    # here so that "an edit loses somebody" cannot happen at all.
    return [
        group
        if group.representative_observation is not None
        and group.representative_embedding is not None
        else _recompute(group)
        for group in numbered
    ]


def _check(groups: list[FaceIdentityGroup], indexes) -> list[int]:
    wanted = sorted({int(index) for index in indexes})
    for index in wanted:
        if index < 0 or index >= len(groups):
            raise EditError(
                f"There is no person #{index + 1}; this scan found "
                f"{len(groups)}."
            )
    return wanted


def merge_groups(
    groups: list[FaceIdentityGroup], indexes: list[int]
) -> list[FaceIdentityGroup]:
    """Folds several cards into one person.

    Raises:
        EditError: If fewer than two distinct people were named, or any of
            them is not in this scan.
    """
    wanted = _check(groups, indexes)
    if len(wanted) < 2:
        raise EditError("Merging needs two people or more.")

    chosen = [groups[index] for index in wanted]
    combined = FaceIdentityGroup(
        group_id=0,
        observations=[o for group in chosen for o in group.observations],
        # Concatenated in the same order as the observations, so the merged
        # card can still be split back apart along the tracks it came from.
        unit_sizes=[size for group in chosen for size in group.unit_sizes],
    )
    kept = [group for position, group in enumerate(groups) if position not in set(wanted)]
    return _ordered(kept + [_recompute(combined)])


def split_group(
    groups: list[FaceIdentityGroup], index: int, tracks: list[int]
) -> list[FaceIdentityGroup]:
    """Peels tracks out of one card into a new one.

    Args:
        groups: The scan's identities.
        index: Which card to split.
        tracks: Which of its tracks to move out, by position.

    Raises:
        EditError: If the card is not in this scan, no valid track was
            named, or every track was -- which would move the whole group
            and change nothing.
    """
    (position,) = _check(groups, [index])
    group = groups[position]
    available = group.tracks
    if len(available) < 2:
        raise EditError(
            "This person was seen in a single unbroken track, so there is "
            "nothing to split them along."
        )

    wanted = sorted({int(track) for track in tracks})
    for track in wanted:
        if track < 0 or track >= len(available):
            raise EditError(
                f"There is no track {track} here; this person has "
                f"{len(available)}."
            )
    if not wanted:
        raise EditError("Splitting needs at least one track to move out.")
    if len(wanted) == len(available):
        raise EditError("Moving every track out would leave nobody behind.")

    moved = [track for i, track in enumerate(available) if i in set(wanted)]
    stayed = [track for i, track in enumerate(available) if i not in set(wanted)]

    peeled = FaceIdentityGroup(
        group_id=0,
        observations=[o for track in moved for o in track],
        unit_sizes=[len(track) for track in moved],
    )
    remainder = FaceIdentityGroup(
        group_id=0,
        observations=[o for track in stayed for o in track],
        unit_sizes=[len(track) for track in stayed],
    )
    kept = [g for i, g in enumerate(groups) if i != position]
    return _ordered(kept + [_recompute(peeled), _recompute(remainder)])


def discard_groups(
    groups: list[FaceIdentityGroup], indexes: list[int]
) -> list[FaceIdentityGroup]:
    """Removes cards that are not people.

    Raises:
        EditError: If nothing was named, any of it is not in this scan, or
            it is everything -- a gallery with nobody in it is not an edit
            anyone means to make.
    """
    wanted = _check(groups, indexes)
    if not wanted:
        raise EditError("Discarding needs somebody to discard.")
    if len(wanted) == len(groups):
        raise EditError("That would discard everyone this scan found.")

    return _ordered(
        [group for position, group in enumerate(groups) if position not in set(wanted)]
    )
