#!/bin/zsh
# One unattended Claude Code run on the oldest open issue labelled "claude"
# that has no pull request yet. launchd starts it nightly (install.sh); it can
# also be run by hand, on a chosen issue: scripts/nightly/run.sh 42
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS="$HOME/Library/Logs/fluxcutter-nightly"
SCRATCH="$HOME/Library/Caches/fluxcutter-nightly"
mkdir -p "$LOGS" "$SCRATCH"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$REPO"

stamp="$(date +%Y-%m-%d-%H%M)"
exec >>"$LOGS/$stamp.log" 2>&1
echo "== $stamp"

# Your own unfinished work is never touched: no run on a dirty tree.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree has uncommitted changes; leaving it alone"
  exit 0
fi
git checkout -q main
git pull -q --ff-only

issue="${1:-}"
if [[ -z "$issue" ]]; then
  open_branches="$(gh pr list --state open --json headRefName --jq '.[].headRefName')"
  for n in $(gh issue list --label claude --state open --json number --jq 'sort_by(.number) | .[].number'); do
    if ! grep -q "^claude/issue-$n-" <<<"$open_branches"; then
      issue="$n"
      break
    fi
  done
fi
if [[ -z "$issue" ]]; then
  echo "no open claude issue without a pull request"
  exit 0
fi
echo "issue #$issue"

# Scans, people and settings for this run only, never your own.
export FLUXCUTTER_SCAN_DIR="$SCRATCH/scans"
export FLUXCUTTER_PEOPLE_DIR="$SCRATCH/people"
export FLUXCUTTER_SETTINGS_DIR="$SCRATCH/settings"

# A branch an earlier run left unfinished is continued, not started again.
git fetch -q origin
branch="$(git branch -r --list "origin/claude/issue-$issue-*" | head -1 | sed 's|^ *origin/||')"
prompt="$(sed -e "s/{{ISSUE}}/$issue/g" -e "s/{{DATE}}/$(date +%Y-%m-%d)/g" \
  -e "s|{{BRANCH}}|${branch:-none}|g" scripts/nightly/prompt.md)"

# Long evaluations run in the foreground: without this a command is cut
# off at 10 minutes, and test_3 alone takes 16.
export BASH_DEFAULT_TIMEOUT_MS=600000 BASH_MAX_TIMEOUT_MS=3600000

# dontAsk: anything not listed is refused rather than asked about, since
# nobody is there to answer. Merging, pushing to main, force-pushing and
# deleting files are deliberately not listed.
claude -p "$prompt" --permission-mode dontAsk --output-format text --allowedTools \
  "Read" "Edit" "Write" "Glob" "Grep" \
  "Bash(git status*)" "Bash(git diff*)" "Bash(git log*)" "Bash(git show*)" \
  "Bash(git branch --show-current)" "Bash(git checkout -b claude/*)" "Bash(git checkout claude/*)" "Bash(git add *)" \
  "Bash(git commit *)" "Bash(git push -u origin claude/*)" "Bash(git push origin claude/*)" \
  "Bash(.venv/bin/python -m pytest *)" "Bash(.venv/bin/python -m app evaluate*)" \
  "Bash(.venv/bin/python -m app info *)" "Bash(.venv/bin/python -m app shots *)" \
  "Bash(gh issue view *)" "Bash(gh issue comment *)" \
  "Bash(gh pr create *)" "Bash(gh pr view*)" "Bash(gh pr checks*)" \
  "Bash(ls*)" "Bash(head *)" "Bash(tail *)" "Bash(wc *)" "Bash(grep *)" "Bash(mkdir -p *)" \
  || echo "claude exited with status $?"

# Whatever the run did not commit is kept on its branch, never carried back
# onto main: commit it there as unfinished, and push it for the next run.
current="$(git branch --show-current)"
if [[ -n "$(git status --porcelain)" ]]; then
  if [[ "$current" == claude/* ]]; then
    git add -A
    git commit -q -m "wip: unfinished nightly run on #$issue" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    git push -q -u origin "$current"
    echo "left unfinished work on $current"
  else
    git stash push -q -u -m "nightly #$issue: changes made outside a claude branch"
    echo "stashed changes made on $current"
  fi
fi
git checkout -q main || true

# If no pull request came of it, say so on the issue, so a stopped run is
# not mistaken for one still going.
if ! gh pr list --state open --json headRefName --jq '.[].headRefName' | grep -q "^claude/issue-$issue-"; then
  gh issue comment "$issue" --body "The nightly run on $(date +%Y-%m-%d) stopped without opening a pull request.${current:+ Its work so far is on \`$current\`.} The next run carries on from there. Log: \`~/Library/Logs/fluxcutter-nightly/$stamp.log\`." || true
fi
echo "== done $(date +%H:%M)"
