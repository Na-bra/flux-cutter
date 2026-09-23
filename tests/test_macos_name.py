"""Tests for naming the process in the macOS menu bar.

The thing this fixes cannot be asserted on directly -- reading the menu bar
needs Accessibility permission, which a test suite should not be asking for.
What can be asserted is the value Tk reads when it builds that menu, which is
CFBundleName on the main bundle.
"""

import sys

import pytest

from app.ui.macos import set_application_name

macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_it_is_a_no_op_off_macos():
    """Windows and Linux take the name from elsewhere; this must not run there."""
    if sys.platform == "darwin":
        pytest.skip("this asserts the non-macOS branch")
    assert set_application_name("FluxCutter") is False


@macos_only
def test_the_name_is_written_and_reads_back():
    """Returns True only after confirming the write landed.

    A version of this that trusted setObject: not to raise reported success
    on a dictionary it had not changed, which is the failure mode worth
    guarding: it looks fixed and is not.
    """
    assert set_application_name("FluxCutter") is True


@macos_only
def test_the_bundle_reports_the_new_name():
    """Independent of the helper's own return value."""
    import ctypes
    import ctypes.util

    set_application_name("FluxCutter")

    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p

    def send(receiver, selector, *args, restype=ctypes.c_void_p, argtypes=()):
        fn = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes),
        )
        return fn(receiver, objc.sel_registerName(selector), *args)

    def string(text):
        return send(
            objc.objc_getClass(b"NSString"),
            b"stringWithUTF8String:",
            text.encode(),
            argtypes=(ctypes.c_char_p,),
        )

    bundle = send(objc.objc_getClass(b"NSBundle"), b"mainBundle")
    info = send(bundle, b"infoDictionary")
    value = send(info, b"objectForKey:", string("CFBundleName"), argtypes=(ctypes.c_void_p,))
    text = ctypes.cast(
        send(value, b"UTF8String", restype=ctypes.c_void_p), ctypes.c_char_p
    ).value.decode()

    assert text == "FluxCutter"


@macos_only
def test_calling_it_twice_is_harmless():
    assert set_application_name("FluxCutter") is True
    assert set_application_name("FluxCutter") is True


# ------------------------------------------------------------------ the icon


def test_the_window_points_at_artwork_that_is_there():
    """A path one directory off is a rocket in the Dock and no error."""
    web = pytest.importorskip("app.ui.web")
    assert web.APP_ICON.is_file()


def test_a_missing_icon_leaves_the_app_as_it_was(tmp_path):
    from app.ui.macos import set_application_icon

    assert set_application_icon(tmp_path / "nothing.png") is False


def test_the_icon_is_a_macos_matter(monkeypatch):
    from app.ui import macos

    monkeypatch.setattr(macos.sys, "platform", "win32")
    assert macos.set_application_icon("/anything.png") is False


# ------------------------------------------------- the Dock's hover label


class _Signed:
    returncode = 0


def _fake_files(tmp_path):
    stub = tmp_path / "Python"
    stub.write_bytes(b"interpreter")
    icon = tmp_path / "icon.icns"
    icon.write_bytes(b"icns")
    return stub, icon


def test_the_bundle_is_named_fluxcutter_and_holds_the_interpreter(tmp_path, monkeypatch):
    from app.ui import macos

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Signed())
    stub, icon = _fake_files(tmp_path)

    executable = macos.build_dev_bundle(stub, icon, tmp_path / "dev")

    contents = tmp_path / "dev" / "FluxCutter.app" / "Contents"
    assert executable == contents / "MacOS" / "FluxCutter"
    assert executable.read_bytes() == b"interpreter"
    assert "<key>CFBundleName</key><string>FluxCutter</string>" in (contents / "Info.plist").read_text()
    assert (contents / "Resources" / "icon.icns").is_file()


def test_the_bundle_is_made_once_and_remade_when_python_changes(tmp_path, monkeypatch):
    """A Homebrew upgrade replaces the stub; a copy of the old one would
    link against a framework that is gone."""
    from app.ui import macos

    signings = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: signings.append(a) or _Signed())
    stub, icon = _fake_files(tmp_path)

    macos.build_dev_bundle(stub, icon, tmp_path / "dev")
    macos.build_dev_bundle(stub, icon, tmp_path / "dev")
    assert len(signings) == 1

    stub.write_bytes(b"a newer interpreter")
    executable = macos.build_dev_bundle(stub, icon, tmp_path / "dev")
    assert len(signings) == 2
    assert executable.read_bytes() == b"a newer interpreter"


def test_a_bundle_that_cannot_be_signed_is_not_used(tmp_path, monkeypatch):
    from app.ui import macos

    class Refused:
        returncode = 1

    monkeypatch.setattr("subprocess.run", lambda *a, **k: Refused())
    stub, icon = _fake_files(tmp_path)

    assert macos.build_dev_bundle(stub, icon, tmp_path / "dev") is None
    assert not (tmp_path / "dev" / "FluxCutter.app").exists()


@pytest.fixture
def exec_calls(monkeypatch, tmp_path):
    from app.ui import macos

    calls = []
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.delenv(macos.BUNDLED_ENV, raising=False)
    monkeypatch.setattr(macos, "interpreter_stub", lambda: tmp_path / "Python")
    monkeypatch.setattr(macos, "build_dev_bundle", lambda stub, icon: tmp_path / "FluxCutter")
    monkeypatch.setattr("os.execve", lambda path, argv, env: calls.append((path, argv, env)))
    return calls


def test_a_checkout_restarts_from_the_bundle_with_its_environment(exec_calls):
    from app.ui import macos

    macos.relaunch_from_bundle(["-m", "app", "ui", "episode.mp4"], "icon.icns")

    path, argv, env = exec_calls[0]
    assert argv[1:] == ["-m", "app", "ui", "episode.mp4"]
    assert env[macos.BUNDLED_ENV] == "1"
    assert env["__PYVENV_LAUNCHER__"] == sys.executable


def test_the_restarted_process_goes_straight_on(exec_calls, monkeypatch):
    from app.ui import macos

    monkeypatch.setenv(macos.BUNDLED_ENV, "1")
    macos.relaunch_from_bundle(["-m", "app", "ui"], "icon.icns")
    assert exec_calls == []


def test_a_built_app_and_other_systems_are_left_alone(exec_calls, monkeypatch):
    from app.ui import macos

    monkeypatch.setattr(macos.sys, "frozen", True, raising=False)
    macos.relaunch_from_bundle(["-m", "app", "ui"], "icon.icns")
    monkeypatch.setattr(macos.sys, "frozen", False)
    monkeypatch.setattr(macos.sys, "platform", "win32")
    macos.relaunch_from_bundle(["-m", "app", "ui"], "icon.icns")

    assert exec_calls == []


def test_the_window_asks_for_a_bundle_icon_that_is_there():
    web = pytest.importorskip("app.ui.web")
    assert web.APP_ICON.with_suffix(".icns").is_file()
