#!/bin/zsh
# Turns the nightly run on: a launchd job that runs scripts/nightly/run.sh at
# 02:00 every night (or as soon as the Mac wakes, if it was asleep), and the
# "claude" label issues are filed under. Undo with scripts/nightly/uninstall.sh.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="com.fluxcutter.nightly"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

gh label create claude --color 6f42c1 \
  --description "A task for the nightly Claude run" 2>/dev/null || true

mkdir -p "$HOME/Library/LaunchAgents"
cat >"$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>$REPO/scripts/nightly/run.sh</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>2</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/fluxcutter-nightly/launchd.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/fluxcutter-nightly/launchd.log</string>
</dict>
</plist>
PLIST
mkdir -p "$HOME/Library/Logs/fluxcutter-nightly"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed. It runs at 02:00; logs are in ~/Library/Logs/fluxcutter-nightly."
echo "Run it now with: launchctl kickstart gui/$(id -u)/$LABEL"
