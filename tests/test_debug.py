"""The switch between a quiet run and a diagnostic one."""

import pytest

from app import debug


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", "  "])
def test_these_all_mean_quiet(monkeypatch, value):
    monkeypatch.setenv(debug.DEBUG_VARIABLE, value)
    assert debug.debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "please"])
def test_anything_else_means_debug(monkeypatch, value):
    monkeypatch.setenv(debug.DEBUG_VARIABLE, value)
    assert debug.debug_enabled() is True


def test_unset_means_quiet(monkeypatch):
    """The default a user gets, and the one that matters most."""
    monkeypatch.delenv(debug.DEBUG_VARIABLE, raising=False)
    assert debug.debug_enabled() is False


def test_quiet_still_lets_errors_through(monkeypatch):
    """A severity floor, not a redirect.

    Silencing warnings must not silence a failure -- onnxruntime's scale
    puts errors at 3 and fatals at 4, so the floor has to sit no higher
    than error.
    """
    monkeypatch.delenv(debug.DEBUG_VARIABLE, raising=False)
    ERROR = 3
    assert debug.onnx_log_severity() <= ERROR


def test_debug_lowers_the_floor(monkeypatch):
    monkeypatch.setenv(debug.DEBUG_VARIABLE, "1")
    talkative = debug.onnx_log_severity()

    monkeypatch.setenv(debug.DEBUG_VARIABLE, "0")
    assert talkative < debug.onnx_log_severity()


def test_the_severity_reaches_the_session(monkeypatch, tmp_path):
    """The helper being right is no use if nothing passes it along."""
    pytest.importorskip("onnxruntime")
    import onnxruntime

    from app.faces.embedder import _open_coreml_session

    seen = {}

    class Options:
        log_severity_level = None

    class Session:
        def get_providers(self):
            return ["CoreMLExecutionProvider"]

    def fake_session(path, options=None, providers=None):
        seen["severity"] = options.log_severity_level
        return Session()

    monkeypatch.setattr(onnxruntime, "SessionOptions", Options)
    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)

    monkeypatch.delenv(debug.DEBUG_VARIABLE, raising=False)
    _open_coreml_session(tmp_path / "unused.onnx")
    quiet = seen["severity"]

    monkeypatch.setenv(debug.DEBUG_VARIABLE, "1")
    _open_coreml_session(tmp_path / "unused.onnx")

    assert seen["severity"] < quiet
