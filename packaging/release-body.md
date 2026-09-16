Unsigned builds. Both platforms will warn on first launch.

**macOS** — the app is quarantined on download and Gatekeeper
will call it damaged. It is not. Either:

    xattr -dr com.apple.quarantine /path/to/FluxCutter.app

or open System Settings → Privacy & Security → **Open Anyway**.
(Right-click → Open no longer works; Apple removed it in Sequoia.)

**Windows** — SmartScreen shows "Windows protected your PC".
Click **More info** → **Run anyway**.

The face models (~174 MB) download on first use, with a progress
bar, and are verified against a pinned checksum.
