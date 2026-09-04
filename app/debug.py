"""One switch for diagnostic output a developer wants and a user does not.

    FLUXCUTTER_DEBUG=1 python -m app group video.mp4

Set it and the inference runtimes report everything they normally would.
Leave it unset and they report only errors, which is what someone running
a scan wants to see: an actual failure, not a running commentary.

This exists because of a specific flood rather than in principle. ArcFace
declares its output as {1, 512}, and the CoreML path deliberately feeds it
a fixed batch of two -- see FaceEmbedder._forward_coreml, where holding the
batch shape constant is worth 94x. onnxruntime notices the mismatch and
warns about it on every single call, so a scan of a 22-minute video wrote
roughly 1,900 identical lines to stderr. The warning is correct and
harmless, and no user should ever have to read it.

It is a severity floor rather than a redirect: an error that stops a scan
still reaches the terminal with this off.
"""

import os

DEBUG_VARIABLE = "FLUXCUTTER_DEBUG"

# onnxruntime's severity scale: 0 verbose, 1 info, 2 warning, 3 error, 4 fatal.
_QUIET = 3
_TALKATIVE = 1

_OFF = {"", "0", "false", "no", "off"}


def debug_enabled() -> bool:
    """Whether the user asked for the noisy version."""
    return os.environ.get(DEBUG_VARIABLE, "").strip().lower() not in _OFF


def onnx_log_severity() -> int:
    """How much onnxruntime should say."""
    return _TALKATIVE if debug_enabled() else _QUIET
