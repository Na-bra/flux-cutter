"""Tests for the face-quality measure.

Sharpness exists to be measured, not to be trusted blindly: the accuracy
investigation found it does not separate the identity errors it was expected
to (see Instructions.md 15), and it is used only to choose which crop appears
on a person card. These check it measures focus, and that it behaves on the
degenerate inputs a real video will hand it.
"""

import cv2
import numpy as np
import pytest

from app.faces.quality import sharpness


@pytest.fixture
def textured():
    """A crop with plenty of detail at every scale."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)


def test_blurring_a_crop_lowers_its_sharpness(textured):
    assert sharpness(cv2.GaussianBlur(textured, (9, 9), 4)) < sharpness(textured)


def test_more_blur_means_less_sharpness():
    """Monotonic, which is what makes it usable for ranking.

    Measured on structure rather than noise: heavy blur drives white noise to
    a near-flat field, where the remaining differences are too small to order
    reliably and the test would be asserting on numerical dust.
    """
    board = np.zeros((112, 112, 3), np.uint8)
    board[::8] = 255
    board[:, ::8] = 255
    scores = [sharpness(cv2.GaussianBlur(board, (9, 9), sigma)) for sigma in (0.5, 1.5, 3.0)]
    assert scores == sorted(scores, reverse=True), scores


def test_a_flat_crop_scores_zero():
    assert sharpness(np.full((112, 112, 3), 128, np.uint8)) == 0.0


def test_an_empty_crop_scores_zero_rather_than_raising():
    """A clamped-away face box yields an empty crop; that must not crash a scan."""
    assert sharpness(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_none_scores_zero():
    assert sharpness(None) == 0.0


def test_a_greyscale_crop_is_accepted(textured):
    grey = cv2.cvtColor(textured, cv2.COLOR_RGB2GRAY)
    assert sharpness(grey) > 0.0


def test_the_measure_is_deterministic(textured):
    """Identity grouping compares runs, so nothing here may wobble."""
    assert sharpness(textured) == sharpness(textured)
