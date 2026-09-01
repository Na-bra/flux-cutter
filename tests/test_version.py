"""Tests for the version number.

There is one of these because it went wrong quietly: the spec hardcoded
1.0.0 and stayed there while v1.1.0 was tagged and shipped, so the bundle
reported a version that had not existed for a release. These check the
shape and the single source, not the value -- whether a change is a patch
or a minor is a judgement no test can make.
"""

import re
from pathlib import Path

import app

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "FluxCutter.spec"


def test_the_version_looks_like_a_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", app.__version__), app.__version__


def test_the_spec_reads_the_version_rather_than_repeating_it():
    """A second copy is what went stale last time."""
    spec = SPEC.read_text()
    assert "CFBundleShortVersionString\": VERSION" in spec
    assert app.__version__ not in spec, (
        "the spec hardcodes the version again; it should read "
        "app/__init__.py so the two cannot drift"
    )


def test_the_spec_can_read_the_version_it_will_ship():
    """Exercises the spec's own reader, which nothing else would."""
    # Only the prelude: the rest of the spec needs SPECPATH and the
    # PyInstaller globals, which exist solely while PyInstaller is running it.
    namespace: dict = {}
    prelude = SPEC.read_text().split("IS_MACOS = ")[0]
    exec(compile(prelude, str(SPEC), "exec"), namespace)

    assert namespace["project_version"](ROOT) == app.__version__
