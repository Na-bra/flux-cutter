"""How usable a detected face is, judged from the crop that will be embedded.

The identity pipeline had three quality gates before this module -- detector
confidence, face-box size, and a per-group landmark-geometry test -- and none
of them measures focus. That gap is visible in the accuracy notes for the
first full-length run: 23.7% of detections on `test_3.mp4` came out of an
aligned crop with Laplacian variance under 40, and the tail of one- and
two-detection identity groups was full of motion blur, backs of heads and
extreme profiles that YuNet still scored 0.72-0.89 -- comfortably above the
0.7 confidence floor, so nothing stopped them.

A blurred face does not merely fail to match; it matches the *wrong* things.
Blur removes exactly the high-frequency detail ArcFace encodes identity in,
so two unrelated blurred faces are more similar to each other than either is
to a sharp frame of itself. Left ungated they seed their own identity cards
and, worse, give the clustering stage spurious evidence to merge on.

Sharpness is measured on the aligned 112x112 crop rather than on the original
box, which matters more than it sounds: Laplacian variance scales with image
resolution, so the same face measured at 300px and at 60px gives very
different numbers. Aligning first puts every face at one size, which is what
makes a single threshold mean the same thing for a close-up and a wide shot.
"""

import cv2
import numpy as np


def sharpness(aligned_crop: np.ndarray) -> float:
    """Focus estimate for one aligned face crop: variance of its Laplacian.

    Higher is sharper. The absolute value is not meaningful on its own --
    it depends on the crop size, which is why callers should pass the
    aligned 112x112 crop rather than a raw box crop -- but it orders faces
    by focus reliably, which is all a gate needs.

    Args:
        aligned_crop: An aligned face crop, RGB or grayscale.

    Returns:
        The variance, or 0.0 for an empty crop.
    """
    if aligned_crop is None or aligned_crop.size == 0:
        return 0.0

    if aligned_crop.ndim == 3:
        grey = cv2.cvtColor(aligned_crop, cv2.COLOR_RGB2GRAY)
    else:
        grey = aligned_crop

    return float(cv2.Laplacian(grey.astype(np.uint8), cv2.CV_64F).var())
