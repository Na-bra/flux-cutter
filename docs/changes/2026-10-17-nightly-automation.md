# A nightly Claude run on GitHub issues

**Status:** written, not switched on. Nothing runs until you run
`scripts/nightly/install.sh`.

## What it does

Once a night, at 02:00 (or as soon as your Mac wakes up, if it was asleep
then), your Mac:

1. checks that the FluxCutter folder has no unfinished work of yours in it,
   and stops if it does;
2. updates `main` from GitHub;
3. picks the oldest open issue labelled **claude** that has no pull request
   yet, and stops if there is none;
4. runs Claude Code on it, unattended, with the instructions in
   `scripts/nightly/prompt.md`.

Claude then branches (`claude/issue-<number>-…`), makes the change, runs the
tests, and runs `python -m app evaluate` if the change touches finding or
grouping people. It writes a page in this folder explaining the change, the
same way this one does. Then it opens a pull request and comments on the
issue with the link. **You review and merge.** It never merges.

One issue per night, so each pull request is small enough to read.

## How you use it

1. Turn it on, once: `scripts/nightly/install.sh`.
2. File work as issues: on GitHub, **New issue → Task for Claude**. The form
   asks what should change, how you'll know it worked, and anything to leave
   alone. The issue gets the **claude** label automatically.
3. In the morning, read the pull request and its page in `docs/changes/`, then
   merge it, ask for changes on it, or close it.

To run it straight away rather than wait for 02:00:
`launchctl kickstart gui/$(id -u)/com.fluxcutter.nightly`. To run it on one
particular issue by hand: `scripts/nightly/run.sh 42`.

## What it is allowed to do, and what it isn't

It runs in Claude Code's `dontAsk` mode: anything not on the list in
`scripts/nightly/run.sh` is refused rather than asked about, since nobody is
awake to answer.

| Allowed | Not allowed |
|---|---|
| read and edit files in the repository | merge a pull request |
| branch, commit, and push branches named `claude/…` | push to `main`, or force-push |
| run the tests and `python -m app evaluate`, `info`, `shots` | delete files, or run other programs |
| read issues, comment on them, open pull requests | touch your kept scans, named people or settings |

Your kept scans and named people are safe twice over. The run uses its own
folders (`~/Library/Caches/fluxcutter-nightly`), and the instructions forbid
it to touch yours.

## What it can't do well

- **It can't use what isn't on this Mac.** The evaluation needs the videos in
  `assets/test-videos/`; S01E01 is currently missing from `~/Downloads`, so
  it is skipped.
- **It needs the Mac on.** A Mac that is shut down at 02:00 skips that night.
  One that is asleep runs when it wakes.
- **It uses your Claude plan.** Each night's run is one working session, and
  counts against your usage like any other.
- **It can be wrong.** Tests passing doesn't mean grouping got better, which
  is why it reports the evaluation numbers and why merging stays with you.

## Where to look when something goes wrong

Each run writes a log to `~/Library/Logs/fluxcutter-nightly/`, named by date
and time. "no open claude issue without a pull request" means there was
nothing to do. "working tree has uncommitted changes" means it saw your
unfinished work and left it alone.

## How to undo it

`scripts/nightly/uninstall.sh` turns it off. The files in `scripts/nightly/`
and the issue form do nothing on their own, so they can stay, or be deleted
in a pull request like any other change.
