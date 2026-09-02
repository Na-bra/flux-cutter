"""Making macOS call the app FluxCutter when it is run from a checkout.

The menu bar and the Dock take an app's name from its bundle -- specifically
from `CFBundleName` in `NSBundle.mainBundle`. The frozen .app has its own
bundle and so is already named correctly; `packaging/FluxCutter.spec` sets
the key and macOS registers the process as "FluxCutter".

Running from a checkout is different, and the reason is not obvious. A
framework build of CPython (Homebrew's is one) re-execs GUI processes through
a stub application inside the framework:

    .../Python.framework/Versions/3.12/Resources/Python.app

so `NSBundle.mainBundle` is *that* bundle, whose CFBundleName is "Python".
Every Tk app run this way is called Python in the menu bar, which is why it
looks like a FluxCutter bug and is not one.

The dictionary that bundle hands out turns out to be an `__NSDictionaryM` --
mutable -- so the name can simply be written into it. Verified by reading the
key back afterwards rather than by trusting that the call did not raise:

    infoDictionary class : __NSDictionaryM
    before               : 'Python'
    after                : 'FluxCutter'

This has to happen before Tk starts, because Tk builds the application menu
during initialisation and reads the name once.

Everything here is cosmetic, so every failure is swallowed: an unexpected
Objective-C runtime, an immutable dictionary on some other Python build, a
future macOS that stops handing out the real dictionary. The app is then
called Python, exactly as it was before, and still works.
"""

import ctypes
import ctypes.util
import sys


def _send(objc, receiver, selector, *args, restype=ctypes.c_void_p, argtypes=()):
    """One Objective-C message.

    objc_msgSend is variadic, so it must be cast to the exact signature of
    the call being made; using it through ctypes' default int-sized argument
    handling silently truncates pointers on arm64.
    """
    function = ctypes.cast(
        objc.objc_msgSend,
        ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes),
    )
    return function(receiver, objc.sel_registerName(selector), *args)


def set_application_name(name: str) -> bool:
    """Names the running process for the macOS menu bar and Dock.

    Call before any Tk window exists. A no-op off macOS.

    Returns:
        True if the name was written and read back, False if anything at all
        went wrong -- there is no half-applied state worth reporting.
    """
    if sys.platform != "darwin":
        return False

    try:
        library = ctypes.util.find_library("objc")
        if library is None:
            return False
        objc = ctypes.cdll.LoadLibrary(library)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p

        def string(text: str):
            return _send(
                objc,
                objc.objc_getClass(b"NSString"),
                b"stringWithUTF8String:",
                text.encode(),
                argtypes=(ctypes.c_char_p,),
            )

        bundle = _send(objc, objc.objc_getClass(b"NSBundle"), b"mainBundle")
        if not bundle:
            return False
        info = _send(objc, bundle, b"infoDictionary")
        if not info:
            return False

        _send(
            objc,
            info,
            b"setObject:forKey:",
            string(name),
            string("CFBundleName"),
            argtypes=(ctypes.c_void_p, ctypes.c_void_p),
        )

        # Read back rather than assume: the dictionary is mutable today, and
        # a silent no-op is exactly the failure this is prone to.
        written = _send(
            objc,
            info,
            b"objectForKey:",
            string("CFBundleName"),
            argtypes=(ctypes.c_void_p,),
        )
        if not written:
            return False
        text = ctypes.cast(
            _send(objc, written, b"UTF8String", restype=ctypes.c_void_p),
            ctypes.c_char_p,
        ).value
        return text is not None and text.decode() == name
    except Exception:  # noqa: BLE001 - cosmetic; never worth failing a launch
        return False
