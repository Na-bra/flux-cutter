#!/bin/zsh
# Turns the nightly run off. Logs and the "claude" label are left in place.
LABEL="com.fluxcutter.nightly"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Uninstalled. Nothing will run at night."
