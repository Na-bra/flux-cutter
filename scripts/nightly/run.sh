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

# Awake from here to the last line, not only while claude works: waiting
# for the network and saving the work at the end both need the Mac up.
# (On 22 November it slept mid-run; caffeinate -w lets go when we exit.)
caffeinate -i -w $$ &

# Runs a command up to TRIES times, PAUSE seconds apart, until it succeeds.
# The network here drops for minutes at a time: on 24 November a run gave
# up at 2am with GitHub unreachable, and two more stopped before they could
# push what they had done.
retry() {
  local tries=$1 pause=$2 n
  shift 2
  for n in {1..$tries}; do
    "$@" && return 0
    (( n < tries )) && { echo "retrying in ${pause}s: $*" >&2; sleep "$pause"; }
  done
  return 1
}

# Your own unfinished work is never touched: no run on a dirty tree.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree has uncommitted changes; leaving it alone"
  exit 0
fi
git checkout -q main
# Up to an hour for the network to come back before giving the night up.
if ! retry 12 300 git pull -q --ff-only; then
  echo "GitHub unreachable for an hour; no run tonight"
  exit 0
fi

issue="${1:-}"
if [[ -z "$issue" ]]; then
  open_branches="$(retry 3 60 gh pr list --state open --json headRefName --jq '.[].headRefName')"
  for n in $(retry 3 60 gh issue list --label claude --state open --json number --jq 'sort_by(.number) | .[].number'); do
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

# A branch an earlier run left unfinished is continued, not started again:
# the local one first, since a run whose push failed has work only here.
retry 3 60 git fetch -q origin || true
branch="$(git branch --list "claude/issue-$issue-*" | head -1 | sed 's|^[* ]*||')"
[[ -n "$branch" ]] || branch="$(git branch -r --list "origin/claude/issue-$issue-*" | head -1 | sed 's|^ *origin/||')"
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
    if retry 5 60 git push -q -u origin "$current"; then
      echo "left unfinished work on $current"
    else
      echo "left unfinished work on $current, committed here but not pushed"
    fi
  else
    git stash push -q -u -m "nightly #$issue: changes made outside a claude branch"
    echo "stashed changes made on $current"
  fi
fi
git checkout -q main || true

# If no pull request came of it, say so on the issue, so a stopped run is
# not mistaken for one still going.
if ! gh pr list --state open --json headRefName --jq '.[].headRefName' | grep -q "^claude/issue-$issue-"; then
  # Only a claude branch holds the run's work; a run that stopped before
  # making one has nothing saved, and naming main here sent you looking.
  if [[ "$current" == claude/* ]]; then
    where="Its work so far is on \`$current\`, and the next run carries on from there."
  else
    where="It stopped before making a branch, so the next run starts the issue afresh."
  fi
  gh issue comment "$issue" --body "The nightly run on $(date +%Y-%m-%d) stopped without opening a pull request. $where Log: \`~/Library/Logs/fluxcutter-nightly/$stamp.log\`." || true
fi
echo "== done $(date +%H:%M)"
