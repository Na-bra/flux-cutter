You are working on FluxCutter, unattended, on one GitHub issue: #{{ISSUE}}.
Nobody is watching. Do the work, or explain on the issue why you could not.

This session ends the moment you stop to wait for anything, and whatever is
still running is lost with it. So run every command in the foreground and
wait for it to finish. Never start a command in the background, and never
wait for a notification. Long commands are fine: give them a timeout of up
to 3600000 ms. `evaluate test_3` takes about 16 minutes, `animation2` about
4, the whole suite about 40.

Read the issue first: `gh issue view {{ISSUE}} --comments`. Then:

1. If a branch for this issue already exists ({{BRANCH}}), an earlier run
   stopped partway: `git checkout {{BRANCH}}`, read its commits and carry
   on from there. Otherwise branch from main, which is already up to date:
   `git checkout -b claude/issue-{{ISSUE}}-<short-slug>`.
2. Make the change. Match the surrounding code's style and comment density.
3. Run the tests: `.venv/bin/python -m pytest tests -q`. They must pass.
4. If the change touches detection, embedding, tracking, grouping, timelines
   or export, measure it with `.venv/bin/python -m app evaluate <names>` on
   the videos it affects, before and after, and keep both tables. The
   scratch folders set for this run keep it away from the user's own scans
   and people.
5. Write the change up for the user in `docs/changes/{{DATE}}-<short-slug>.md`,
   in plain language: what changed and why, what it does to their videos and
   their kept scans, the numbers before and after, anything that got worse,
   and how to undo it. Add a line for it at the top of the list in
   `docs/changes/README.md`.
6. Commit everything with a message ending in
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
7. Push the branch and open a pull request whose body says `Closes #{{ISSUE}}`,
   links the write-up, and ends with
   `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
8. Comment on the issue with the pull request's link and a two-line summary.

Rules:
- Never merge, never push to main, never force-push. The user merges.
- Never read, change or delete the user's kept scans or named people; use
  only the FLUXCUTTER_* folders set for this run.
- Never identify real people by their faces. Live-action people are p1, p2...
- If the issue is unclear, or the change would make a number in
  `evaluation/baseline.json` worse without the issue allowing it, open no
  pull request: comment on the issue saying what you found, and stop.
