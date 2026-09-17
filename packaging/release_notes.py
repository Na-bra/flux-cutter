"""Compose a release's notes from the message on its tag.

Every draft the build has produced carried the same body: how to get past
Gatekeeper and SmartScreen, and nothing about what changed. The notes for
1.9.0 through 1.9.4 were each written by hand into the draft after it
appeared -- a step that can be forgotten, and nearly was: 1.9.0 came close
to going out without mentioning that Export worked again, the one thing its
users needed to know.

An annotated tag already carries that message, written at the moment the
release is cut and stored with it. This puts it above the standing install
text, so the draft arrives saying what changed.

    python packaging/release_notes.py --tag v1.2.3 > notes.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTALL_TEXT = HERE / "release-body.md"

# A horizontal rule between what changed and the standing install text, so
# the two read as separate things rather than one run-on note.
SEPARATOR = "\n\n---\n\n"

# `%(contents)` on a signed tag includes the signature block. Nobody wants
# to read it and GitHub will not verify it from inside the notes.
SIGNATURE_MARKER = "-----BEGIN PGP SIGNATURE-----"


def read_tag_message(tag: str, repo: Path | None = None) -> str:
    """The message on an annotated tag; "" for a lightweight or absent one.

    A lightweight tag is just a name pointing straight at a commit, so git
    answers `%(contents)` with that commit's message -- which here would
    dress "Merge pull request #5 from ..." up as release notes. Only an
    annotated tag carries a message somebody wrote to be read, so the
    object type is checked rather than assumed.
    """
    result = subprocess.run(
        ["git", "tag", "-l", "--format=%(objecttype)%0a%(contents)", tag],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    objecttype, _, contents = result.stdout.partition("\n")
    return contents if objecttype.strip() == "tag" else ""


def restore_tag(tag: str, remote: str, repo: Path | None = None) -> None:
    """Fetches the tag itself again, when what is checked out is not it.

    actions/checkout, on a tag push, fetches every tag and then fetches the
    tagged commit once more *under the tag's name* -- which replaces the
    annotated tag with a lightweight one pointing at the commit. The
    message is gone before anything can read it. v1.9.5 was the first
    release this ran for, and its draft arrived with the install text
    alone, because a lightweight tag is exactly what the fallback is for.

    Fetching the tag ref by name, forced, puts the annotated object back.
    """
    result = subprocess.run(
        ["git", "cat-file", "-t", f"refs/tags/{tag}"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() == "tag":
        return
    subprocess.run(
        ["git", "fetch", "--force", "--no-tags", remote, f"refs/tags/{tag}:refs/tags/{tag}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


def tag_message(raw: str, tag: str = "") -> str:
    """The part of a tag's message worth printing, or "" if there is none."""
    text = raw.split(SIGNATURE_MARKER, 1)[0].strip()
    if not text:
        return ""
    # `git tag -m v1.2.3 v1.2.3` carries a message that only repeats the
    # name the reader is already looking at. Leading the notes with a bare
    # version number is worse than not leading them at all, so it counts
    # as nothing to say. The notes then fall back to the install text
    # alone, which is exactly what every release before this one shipped.
    if text in {tag, tag.lstrip("v")}:
        return ""
    return text


def compose(raw_message: str, install_text: str, tag: str = "") -> str:
    """Tag message above install text, with a rule between them."""
    message = tag_message(raw_message, tag)
    install = install_text.strip()
    if not message:
        return install + "\n"
    return message + SEPARATOR + install + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compose a release's notes from its tag's message."
    )
    parser.add_argument("--tag", required=True, help="the tag's name, e.g. v1.2.3")
    parser.add_argument(
        "--repo", type=Path, default=None, help="repository to read the tag from"
    )
    parser.add_argument(
        "--fetch-from",
        default=None,
        metavar="REMOTE",
        help="fetch the tag again from this remote first, in case a checkout "
        "has replaced the annotated tag with a lightweight one",
    )
    parser.add_argument(
        "--require-message",
        action="store_true",
        help="fail rather than fall back to the install text when the tag "
        "has no message -- so a release job cannot quietly lose its notes",
    )
    parser.add_argument(
        "--install-file",
        type=Path,
        default=INSTALL_TEXT,
        help="the standing install text (default: packaging/release-body.md)",
    )
    args = parser.parse_args(argv)

    if args.fetch_from:
        restore_tag(args.tag, args.fetch_from, args.repo)
    raw = read_tag_message(args.tag, args.repo)

    if args.require_message and not tag_message(raw, args.tag):
        # The annotation form is what GitHub Actions shows on the run's
        # summary page, where a silent fallback would not be noticed.
        print(
            f"::error::{args.tag} has no message to put in the release notes. "
            "Tag releases with: git tag -a --cleanup=whitespace -F notes.md "
            f"{args.tag}",
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(
        compose(raw, args.install_file.read_text(encoding="utf-8"), args.tag)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
