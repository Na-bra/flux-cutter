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
