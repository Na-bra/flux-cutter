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


def set_application_icon(path) -> bool:
    """Gives a run from a checkout FluxCutter's own Dock icon.

    The same cause as the name above: an unbundled process borrows the
    Python.app stub's bundle, so the Dock shows Python's rocket rather
    than FluxCutter. The built .app is not affected -- its bundle carries
    icon.icns (packaging/FluxCutter.spec) -- which is why the icon looks
    missing only in development.

    AppKit is there whenever the window is: pywebview draws through it on
    macOS and depends on PyObjC for that. Cosmetic, so any failure leaves
    the rocket and the app still opens.

    Returns:
        True if the Dock now holds the image, False otherwise.
    """
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import NSApplication, NSImage

        image = NSImage.alloc().initWithContentsOfFile_(str(path))
        if image is None:
            return False
        application = NSApplication.sharedApplication()
        application.setApplicationIconImage_(image)
        return application.applicationIconImage() is not None
    except Exception:  # noqa: BLE001 - cosmetic; never worth failing a launch
        return False


# ------------------------------------------------------- the Dock's label
#
# Renaming the running process fixes the menu bar, and the record macOS
# keeps for it reads "FluxCutter" throughout (`lsappinfo` says so: the
# display name, the bundle name, even WebKit's helpers become "FluxCutter
# Networking"). The Dock still labelled the icon "Python" on hover. It
# names a tile after the bundle on disk -- Python.app -- and nothing a
# running process can change will rename a file it did not launch from.
#
# So a checkout launches from a bundle that *is* called FluxCutter: a tiny
# .app holding a copy of the interpreter stub from Python.app, FluxCutter's
# Info.plist and its icon. Framework Python finds its standard library
# through the framework it links against, not through where the stub sits,
# and `__PYVENV_LAUNCHER__` is how the stub learns which virtual
# environment it belongs to -- the same variable the venv's own `python`
# sets before handing over to Python.app. Tried by hand first and checked
# by hovering: the Dock then says FluxCutter.

BUNDLED_ENV = "FLUXCUTTER_BUNDLED"
_BUNDLE_ID = "com.fluxcutter.dev"
_INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>FluxCutter</string>
<key>CFBundleDisplayName</key><string>FluxCutter</string>
<key>CFBundleIdentifier</key><string>{bundle_id}</string>
<key>CFBundleExecutable</key><string>FluxCutter</string>
<key>CFBundleIconFile</key><string>icon.icns</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
"""


def dev_bundle_dir():
    """Where the development bundle lives. FLUXCUTTER_DEV_BUNDLE_DIR moves it."""
    import os
    from pathlib import Path

    override = os.environ.get("FLUXCUTTER_DEV_BUNDLE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / "FluxCutter" / "dev"


def interpreter_stub():
    """The real interpreter inside a framework build's Python.app, or None.

    A Python that is not a framework build (some pyenv builds, for one) has
    no Python.app, is not re-executed through one, and is left alone.
    """
    from pathlib import Path

    stub = Path(sys.base_prefix) / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    return stub if stub.is_file() else None


def build_dev_bundle(stub, icon, directory=None):
    """Makes FluxCutter.app around `stub`, or reuses the one already made.

    Rebuilt when the stub or the icon changes -- a Homebrew upgrade of
    Python replaces the stub, and a copy of the old one would link against
    a framework that is no longer there. The copy is signed ad hoc, since
    Apple silicon will not run an unsigned binary and the copy's signature
    no longer matches the bundle around it.

    Returns:
        The executable inside the bundle, or None if it could not be made.
    """
    import shutil
    import subprocess
    from pathlib import Path

    directory = Path(directory) if directory is not None else dev_bundle_dir()
    bundle = directory / "FluxCutter.app"
    executable = bundle / "Contents" / "MacOS" / "FluxCutter"
    stamp = bundle / "Contents" / "fluxcutter-source"

    stub, icon = Path(stub), Path(icon)
    wanted = "\n".join(
        f"{path} {path.stat().st_size} {int(path.stat().st_mtime)}"
        for path in (stub, icon)
        if path.is_file()
    )
    try:
        if executable.is_file() and stamp.is_file() and stamp.read_text() == wanted:
            return executable

        shutil.rmtree(bundle, ignore_errors=True)
        (bundle / "Contents" / "MacOS").mkdir(parents=True)
        (bundle / "Contents" / "Resources").mkdir()
        shutil.copy2(stub, executable)
        if icon.is_file():
            shutil.copy2(icon, bundle / "Contents" / "Resources" / "icon.icns")
        (bundle / "Contents" / "Info.plist").write_text(_INFO_PLIST.format(bundle_id=_BUNDLE_ID))
        signed = subprocess.run(
            ["codesign", "--force", "--sign", "-", str(bundle)],
            capture_output=True,
        )
        if signed.returncode != 0:
            shutil.rmtree(bundle, ignore_errors=True)
            return None
        stamp.write_text(wanted)
        return executable
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(bundle, ignore_errors=True)
        return None


def relaunch_from_bundle(arguments, icon) -> None:
    """Restarts this run of the window from FluxCutter.app, if it should.

    Only on macOS, only from a checkout (a built app is already its own
    bundle), only from a framework Python, and only once: the restarted
    process carries BUNDLED_ENV and so goes straight on. If anything at
    all stands in the way this returns and the window opens as before,
    called Python in the Dock and otherwise the same.

    Args:
        arguments: What to run in the new process, after the interpreter
            -- `-m app ui [video]`.
        icon: The .icns the bundle shows.
    """
    import os
    from pathlib import Path

    if sys.platform != "darwin" or getattr(sys, "frozen", False):
        return
    if os.environ.get(BUNDLED_ENV):
        return
    stub = interpreter_stub()
    if stub is None:
        return
    executable = build_dev_bundle(stub, icon)
    if executable is None:
        return

    environment = dict(os.environ)
    environment[BUNDLED_ENV] = "1"
    # Which virtual environment the interpreter belongs to, as the venv's
    # own launcher would have said; and where `app` is, so `-m app` finds
    # it from any working directory.
    environment["__PYVENV_LAUNCHER__"] = sys.executable
    checkout = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (checkout, environment.get("PYTHONPATH", "")) if part
    )
    try:
        os.execve(str(executable), [str(executable), *arguments], environment)
    except OSError:
        return
