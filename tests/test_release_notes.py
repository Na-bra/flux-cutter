"""Notes that say what changed, composed from the tag's own message.

The build used to write the same body into every draft -- the install
warnings, and nothing about what changed -- so the real notes were typed in
by hand afterwards, once per release, from memory. These check the
replacement: that an annotated tag's message reaches the notes, that a tag
without one still produces exactly what every earlier release shipped, and
that the workflow is still wired to the script that does it.

The module is loaded by path rather than imported. `packaging` is also the
name of a package on PyPI that half this project's dependencies import, and
shadowing it from the repository root is not worth the convenience.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging" / "release_notes.py"
INSTALL_TEXT = ROOT / "packaging" / "release-body.md"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"


def _load():
    spec = importlib.util.spec_from_file_location("_release_notes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_notes = _load()

INSTALL = "Unsigned builds. Both platforms will warn on first launch."


def test_the_tag_message_leads_the_notes():
    notes = release_notes.compose(
        "Upgrading no longer throws away the people you named\n",
        INSTALL,
        "v1.9.4",
    )
    assert notes.startswith("Upgrading no longer throws away")
    assert INSTALL in notes
    # A rule between them, so what changed and how to install read as two
    # things rather than one paragraph that changes subject.
    assert "\n---\n" in notes
    assert notes.index("---") < notes.index(INSTALL)


def test_a_multi_paragraph_message_survives_whole():
    message = "## Headline\n\nA paragraph.\n\n- a bullet\n- another"
    notes = release_notes.compose(message, INSTALL, "v2.0.0")
    assert message in notes


def test_no_message_leaves_the_notes_exactly_as_they_have_always_been():
    """A lightweight tag must not make a release worse than the last one."""
    assert release_notes.compose("", INSTALL, "v1.9.4") == INSTALL + "\n"
    assert release_notes.compose("   \n\n  ", INSTALL, "v1.9.4") == INSTALL + "\n"


def test_a_message_that_only_repeats_the_tag_says_nothing():
    """`git tag -m v1.2.3 v1.2.3` is a tag with nothing to tell anyone."""
    for raw in ("v1.2.3", "v1.2.3\n", "1.2.3"):
        assert release_notes.compose(raw, INSTALL, "v1.2.3") == INSTALL + "\n"


def test_a_signature_block_is_not_part_of_the_notes():
    signed = (
        "Real news for the reader\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "iQIzBAABCgAdFiEE...\n"
        "-----END PGP SIGNATURE-----\n"
    )
    notes = release_notes.compose(signed, INSTALL, "v1.2.3")
    assert "Real news for the reader" in notes
    assert "PGP" not in notes
    assert "iQIzBAAB" not in notes


def test_the_install_text_still_says_what_each_platform_will_do():
    """It is the one part of the notes no release can afford to lose."""
    text = INSTALL_TEXT.read_text(encoding="utf-8")
    assert "com.apple.quarantine" in text
    assert "SmartScreen" in text
    assert "174 MB" in text


def test_the_workflow_still_runs_the_script_it_depends_on():
    """Renaming either half without the other would ship blank notes."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "packaging/release_notes.py" in workflow
    assert "body_path: release-notes.md" in workflow
    # The message lives on the tag object; a shallow checkout would not
    # fetch it, and the notes would silently fall back to install text.
    assert "fetch-depth: 0" in workflow


HAS_GIT = subprocess.run(["git", "--version"], capture_output=True).returncode == 0
needs_git = pytest.mark.skipif(not HAS_GIT, reason="needs git")


def _repo(tmp_path):
    run = lambda *args: subprocess.run(
        args, cwd=tmp_path, check=True, capture_output=True, text=True
    )
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("x")
    run("git", "add", "a.txt")
    run("git", "commit", "-qm", "Merge pull request #5 from somewhere")
    return run


@needs_git
def test_end_to_end_from_a_real_annotated_tag(tmp_path):
    """The half only a real tag can check: that git hands the message over.

    `git tag -l --format=...` is the whole contract between this script and
    the repository. Asserting it against a real tag rather than a fixture
    is the difference between testing the pipeline and testing my memory
    of it.
    """
    run = _repo(tmp_path)
    # --cleanup=whitespace is not optional: see the test below.
    run(
        "git", "tag", "-a", "--cleanup=whitespace", "v9.9.9",
        "-m", "## What changed\n\nThe thing they came for.",
    )

    notes = subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", "v9.9.9", "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert notes.startswith("## What changed")
    assert "The thing they came for." in notes
    assert "com.apple.quarantine" in notes


@needs_git
def test_a_lightweight_tag_does_not_publish_a_commit_message(tmp_path):
    """git answers %(contents) for a lightweight tag with the commit's message.

    Which is how "Merge pull request #5 from somewhere" would have become
    the headline of a release. A tag nobody wrote a message on gets the
    notes every release before this one shipped, and nothing else.
    """
    run = _repo(tmp_path)
    run("git", "tag", "v9.9.9")

    assert release_notes.read_tag_message("v9.9.9", tmp_path) == ""
    notes = subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", "v9.9.9", "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Merge pull request" not in notes
    assert notes.strip() == INSTALL_TEXT.read_text(encoding="utf-8").strip()


@needs_git
def test_a_tag_that_is_not_there_does_not_stop_the_release(tmp_path):
    """A draft with the usual notes beats a red release job and no draft."""
    _repo(tmp_path)
    assert release_notes.read_tag_message("v0.0.0", tmp_path) == ""


@needs_git
def test_git_eats_markdown_headings_unless_told_not_to(tmp_path):
    """The one thing about this that cannot be fixed in code.

    git cleans a tag message by default, and cleaning means deleting every
    line that starts with `#` -- which in a message written as markdown is
    every heading. The text never reaches the tag object, so nothing
    downstream can put it back. A release cut with a plain `git tag -a -m`
    would have lost the headline of the notes and kept the paragraph under
    it, which is the kind of loss nobody notices until the release is out.

    Recorded here because the fix belongs to whoever types the command, and
    this is the only place that knowledge is executable.
    """
    run = _repo(tmp_path)
    message = "## The headline\n\nThe paragraph."

    run("git", "tag", "-a", "eaten", "-m", message)
    run("git", "tag", "-a", "--cleanup=whitespace", "kept", "-m", message)

    assert "## The headline" not in release_notes.read_tag_message("eaten", tmp_path)
    assert "The paragraph." in release_notes.read_tag_message("eaten", tmp_path)
    assert "## The headline" in release_notes.read_tag_message("kept", tmp_path)


@needs_git
def test_the_message_survives_what_actions_checkout_does_to_the_tag(tmp_path):
    """v1.9.5's draft arrived with no notes. On a tag push, actions/checkout
    runs these two fetches -- copied from that job's log -- and the second
    replaces the annotated tag with a lightweight one on the same commit."""
    origin = tmp_path / "origin"
    origin.mkdir()
    run = _repo(origin)
    run("git", "tag", "-a", "--cleanup=whitespace", "v9.9.9", "-m", "## What changed\n\nIt works.")
    commit = run("git", "rev-parse", "v9.9.9^{}").stdout.strip()

    runner = tmp_path / "runner"
    runner.mkdir()
    job = lambda *args: subprocess.run(
        args, cwd=runner, check=True, capture_output=True, text=True
    )
    job("git", "init", "-q")
    job("git", "remote", "add", "origin", str(origin))
    job("git", "fetch", "--prune", "origin", "+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*")
    job("git", "fetch", "--no-tags", "--prune", "origin", f"+{commit}:refs/tags/v9.9.9")

    # The damage, reproduced: the tag is now just a name for the commit.
    assert release_notes.read_tag_message("v9.9.9", runner) == ""

    notes = subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", "v9.9.9", "--repo", str(runner),
         "--fetch-from", "origin", "--require-message"],
        capture_output=True, text=True, check=True,
    ).stdout

    assert notes.startswith("## What changed")


@needs_git
def test_a_release_with_no_message_fails_loudly_when_asked_to(tmp_path):
    """The fallback to install text alone is what hid the v1.9.5 failure."""
    run = _repo(tmp_path)
    run("git", "tag", "v9.9.9")

    finished = subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", "v9.9.9", "--repo", str(tmp_path), "--require-message"],
        capture_output=True, text=True,
    )

    assert finished.returncode == 1
    assert "::error::" in finished.stderr
    assert "--cleanup=whitespace" in finished.stderr
    assert finished.stdout == ""


def test_the_workflow_restores_the_tag_and_requires_its_message():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "--fetch-from origin" in workflow
    assert "--require-message" in workflow
