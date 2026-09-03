"""Shared test setup.

`assets/` is not in the repository -- the sample videos are not ours to
redistribute and the models are 174 MB -- so a fresh clone, and CI, has no
footage to test against. The tests that need real video skip rather than
error there, which keeps a clone's first `pytest` run green and honest about
what it did not check.

Everything that can be tested without footage still is: the grouping rules,
the appearance timeline, the editorial merge, the model downloader, the UI's
worker layer. That is the majority of the suite.
"""

from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "assets"
TEST_VIDEO = ASSETS / "test-videos" / "test.mp4"
MODELS = ASSETS / "models"

# Modules whose tests cannot run without real footage or the ONNX models.
#
# The gate is per module, which is coarse: a module lands here because most
# of it needs footage, and any test in it that does not gets skipped along
# with the rest. That is how the regression tests for the frame-sampling
# burst came to be written, committed and shipped without CI ever running
# them -- they drive the sampling schedule through a fake container and
# never open a file. Such a test carries RUNS_WITHOUT_ASSETS to opt back in.
NEEDS_ASSETS = {
    "test_cutter",
    "test_detector",
    "test_embedder",
    "test_frames",
    "test_video",
}


# Marker for a test inside a NEEDS_ASSETS module that needs no assets at all.
RUNS_WITHOUT_ASSETS = "runs_without_assets"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{RUNS_WITHOUT_ASSETS}: test in an asset-gated module that needs no "
        "footage or models, so it must still run on a bare clone and in CI.",
    )


def assets_available() -> bool:
    return TEST_VIDEO.is_file()


def pytest_collection_modifyitems(config, items):
    if assets_available():
        return

    skip = pytest.mark.skip(
        reason=(
            f"no sample footage at {TEST_VIDEO}; these tests need real video. "
            "See the README for where to put one."
        )
    )
    for item in items:
        if item.get_closest_marker(RUNS_WITHOUT_ASSETS):
            continue
        if item.module.__name__.rsplit(".", 1)[-1] in NEEDS_ASSETS:
            item.add_marker(skip)
