# Personal Claude Code notes hook

An opt-in `SessionStart` hook that populates `CLAUDE.local.md` — which Claude Code already loads
automatically as project memory, and which is gitignored — from a personal branch you name for
yourself on a remote (`origin` by default), so your own workflow preferences ("always open my PRs
as drafts," "never touch branch X directly," etc.) persist across sessions without ever being
committed to a shared branch.

It works out of the box with no config at all: it reads from a branch named `claude/personal-notes`
on `origin` unless you tell it otherwise. Run [`create-personal-notes-branch.sh`](./create-personal-notes-branch.sh)
once to create that branch with an empty notes file, and every session from then on picks it up
automatically. It is collision-free for multiple contributors sharing one remote if you each
override the branch name via your own config instead of relying on the shared default.

## How it decides what to read

`session-start.sh` looks for a remote in this order, first one found wins:

1. **`git config claude.personalNotesRemote`** — local to one clone's `.git/config`.
2. **`CLAUDE_PERSONAL_NOTES_REMOTE` environment variable** — used only if the git config isn't set.
3. **`origin`** — the zero-config default, used if neither of the above is set.

The branch name on that remote follows the same precedence (`claude.personalNotesBranch` git
config, then `CLAUDE_PERSONAL_NOTES_BRANCH` env var, then the default `claude/personal-notes`), and
so does the path on that branch (`claude.personalNotesPath` git config, then
`CLAUDE_PERSONAL_NOTES_PATH` env var, then the default `.claude/personal/cram-notes.md`).

The hook is still a no-op in effect for anyone who never creates the branch it resolves to on
*neither* the resolved remote *nor* the fallback below: `git fetch` finds nothing, so it exits
without writing `CLAUDE.local.md`.

### Fallback: the current branch's own upstream remote

If the resolved remote doesn't have the branch, `session-start.sh` and `save-personal-notes.sh` try
one more thing before giving up: the remote your *currently checked-out branch* already tracks
(i.e. what `git rev-parse --abbrev-ref --symbolic-full-name @{upstream}` reports), if it has one and
it differs from the resolved remote. This covers a very common case with zero configuration at
all — a clone whose checked-out branch already tracks your own fork under some remote name other
than `origin` (e.g. a session environment where `origin` is the shared upstream repo and your fork
is a differently-named remote) — the hook finds your notes there automatically.

This fallback is read-only: it only ever changes where notes are *read* from, never where
`create-personal-notes-branch.sh` *creates* them (that script still targets exactly the resolved
remote, so creation stays a deliberate, unambiguous act — though it does refuse to create a second,
divergent copy if the branch already exists on the fallback remote; see its own comments). When
`save-personal-notes.sh` reads notes via the fallback, it also *writes* the edit back to that same
fallback remote, not the resolved one, so a save always lands wherever the notes actually came from.
The header `session-start.sh` writes always names whichever remote actually served the notes, so
it's never ambiguous which one was used.

### When you need to override the remote

The remote only needs overriding when your own notes don't live on this clone's `origin` — for
example, some session environments name the upstream repo `origin` and give your own fork a
different remote name, or don't add your fork as a remote at all. The value can be either form,
and `git fetch`/`git push` accept both identically:

- **A remote name already configured in the clone** (e.g. `myfork`) — the natural choice for a
  persistent local clone where you've already added your fork as a remote.
- **A raw git URL** (e.g. `https://github.com/<you>/<repo>`) — needs no `git remote add` first, so
  it works even in a clone that's never heard of your fork (a session environment that only added
  the upstream repo, or a fresh clone every session). This is usually the right choice for
  overriding the remote specifically, since it has no dependency on a particular remote alias
  existing.

Whether you need to override the default branch name depends on how your sessions start:

- **A persistent local clone** (you `git clone` once and keep working in it) → the default just
  works once you've run the setup script below. Only set git config if you want a different
  branch/path than the shared default (e.g. to keep your notes separate from other contributors').
- **A fresh clone every session** (e.g. a cloud/web session environment that clones the repo from
  scratch each time) → the default still just works, since it needs no `.git/config` entry to
  survive. Only set the environment variable if you want to override it — see Option A below.

## Setup: quick start (works for both persistent and fresh-clone sessions)

Once, from any clone with push access to `origin`:

```bash
"$CLAUDE_PROJECT_DIR/.claude/hooks/create-personal-notes-branch.sh"
```

This creates `claude/personal-notes` on `origin` with a single empty
`.claude/personal/cram-notes.md`, without touching your current branch or working tree. Every new
Claude Code session — local or fresh-clone — now runs the hook automatically and writes
`CLAUDE.local.md` from that branch, with no further configuration needed.

## Editing your notes

Just ask Claude, in any session: *"add \<X\> to my personal notes"* or *"edit my personal notes."*
No extra setup or explanation is needed — `session-start.sh` writes a short header at the top of
`CLAUDE.local.md` every session (see below), and since Claude Code already loads `CLAUDE.local.md`
as project memory, that header is always in context. It names the resolved branch/path and points
at [`save-personal-notes.sh`](./save-personal-notes.sh), so Claude edits the notes below the header
and runs that script to push the change back — deterministically, with no guessing at where notes
live or how to persist them.

The header looks like this (regenerated every session — editing it has no effect), naming whichever
remote actually served the notes (the resolved one, or the fallback):

```
<!--
Personal notes, synced from 'claude/personal-notes' (.claude/personal/cram-notes.md) on remote
'origin' by session-start.sh.
To edit: change the notes between the markers below, then run
  "$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
to push the change back. This header and the markers are regenerated every
session from git config/environment/default (plus a same-branch-upstream
fallback) - editing them has no effect; only content between the markers is
ever saved.
-->
<!-- BEGIN-PERSONAL-NOTES -->
<!-- END-PERSONAL-NOTES -->
```

To do it by hand instead: edit `CLAUDE.local.md` between those two marker lines, then run

```bash
"$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
```

It resolves the branch/path exactly like `session-start.sh` does, extracts only what's between the
markers, and pushes just that — in a scratch worktree, so your current branch and working tree are
untouched, and as a no-op if nothing actually changed.

## Tracking a PR's plan and progress

On any branch with a sensible "current PR" — not the default branch, a detached `HEAD`, or the
personal-notes branch itself — `CLAUDE.local.md` also gets a second section, keyed to that branch,
for the plan/progress/next-steps of whatever you're working on. Like the notes above, it's stored
only on the personal-notes branch (at `.claude/personal/pr-progress/<branch-name>.md`), so it can
never end up committed to the PR branch, or merged when the PR merges — that's a property of where
it's stored, not a rule anyone has to remember.

It's always present on a qualifying branch, even before anything's been saved — as a scaffold
prompting you (or Claude) to initialize it — so the section is there to nudge maintaining it from
the first session on a new PR, not only once something already exists:

```
<!--
PR progress for branch 'my-feature-branch', synced from 'claude/personal-notes'
(.claude/personal/pr-progress/my-feature-branch.md) on remote 'origin' by
session-start.sh. Maintain the current plan, what's done, and what's next
here throughout work on this PR. It is never merged: it lives only on
'claude/personal-notes', never on this branch. A stale file left behind after the
PR merges is harmless (just unread from then on) - delete it directly on
'claude/personal-notes' if you want to tidy it up.
To edit: change the notes between the markers below, then run
  "$CLAUDE_PROJECT_DIR/.claude/hooks/save-pr-progress.sh"
to push the change back. This header and the markers are regenerated every
session - editing them has no effect; only content between the markers is
ever saved.
-->
<!-- BEGIN-PR-PROGRESS -->
No progress recorded yet for this branch. Initialize it now: a short plan,
what's done so far, and what's next. Keep it current as you work.
<!-- END-PR-PROGRESS -->
```

Ask Claude to keep it updated the same way as your notes: it edits between the `BEGIN-PR-PROGRESS`/
`END-PR-PROGRESS` markers, then runs [`save-pr-progress.sh`](./save-pr-progress.sh) — which resolves
the same remote/branch as everything else, derives the path from whichever branch is currently
checked out (never configured directly, so one PR's progress can't accidentally get saved under
another PR's key), extracts only what's between its own markers, and pushes just that. A good anchor
for *when* to save: whenever the plan changes, and before ending any turn that changed it.

## Tracking a multi-PR/multi-session plan (plan dashboards)

A single PR's progress note (above) doesn't scale to an initiative spanning many PRs across several
branches — a stacked refactor, a multi-wave programme, anything you'd otherwise write up as a
one-off master-roadmap doc. For that, a **plan** is a structured
`.claude/personal/plans/<plan-id>/plan.yaml` (waves, tracks, and items — branch, PR number, status,
dependencies) plus a sibling `roadmap.md` for the narrative ("why", history, design decisions) that
doesn't belong in structured data. Both live only on the personal-notes branch, exactly like
everything else in this document. See
[`.claude/personal/plans/README.md`](../personal/plans/README.md) (on the personal-notes branch) for
the full schema, and [`.claude/skills/plan-dashboard/SKILL.md`](../skills/plan-dashboard/SKILL.md)
for how a plan gets turned into a live Artifact dashboard. New to this - want to see it end to end
before diving into the schema reference? See
[`plan-dashboard/example-walkthrough.md`](../skills/plan-dashboard/example-walkthrough.md): a short,
worked example with screenshots, from a plan-mode idea to a published dashboard.

**Auto-discovery.** If the branch you're on appears as an item in some plan, `CLAUDE.local.md` also
gets that plan's `plan.yaml` and `roadmap.md` pulled in — the same idea as PR progress above, but for
the wider initiative your branch belongs to, so you don't have to go find and read a roadmap doc by
hand. This is looked up via a generated branch→plan-id reverse index
(`.claude/personal/plans/_generated/branch-index.tsv`), never hand-maintained, so it can't drift out
of sync with the plans it's derived from. Unlike PR progress, there's no scaffold for a branch with
no plan — most branches don't belong to one, and `CLAUDE.local.md` simply gets no plan section that
session.

**Editing.** Change the manifest/roadmap between the `BEGIN-PLAN-MANIFEST`/`END-PLAN-MANIFEST` and
`BEGIN-PLAN-ROADMAP`/`END-PLAN-ROADMAP` markers, then run
[`save-plan.sh`](./save-plan.sh) `[<plan-id>]` — it pushes both files back and regenerates the
reverse index in the same commit (scanning every plan, so the index can't drift). The plan id is
optional if the current branch already resolves to one; pass it explicitly to save a plan from a
branch that isn't itself one of its tracked items, or to bootstrap a brand-new plan (see
`save-plan.sh`'s own header comment for that flow — there is still no separate create-plan.sh, but
see "Creating a new plan" below for the automated version of doing it by hand).

**Publishing the dashboard.** `save-plan.sh` only pushes data — it can't call the `Artifact` tool
itself (only a live Claude session can), so it prints a reminder to run `/plan-dashboard <plan-id>`
afterward. That skill re-reads the manifest, cross-checks every item against live GitHub PR/CI/review
state (so a manifest can never silently go stale the way a hand-maintained roadmap doc could), and
publishes/updates the dashboard Artifact. Run `/plan-dashboard` with no argument to publish the
master index listing every plan.

**Pull request labels this tooling relies on.** These are the only labels
[`build_dashboard.py`'s `PullRequestLabel`](../skills/plan-dashboard/build_dashboard.py) recognizes —
every one of them is a label this repo's own convention applies by hand, never one GitHub sets
itself:

- `merged` — applied when a pull request's changes actually landed but GitHub's own merge API never
  recorded it (its branch was pushed directly, then the pull request closed by hand rather than
  through GitHub's merge button). `build_dashboard.py`'s drift/status logic treats this exactly like
  `merged_at` being set — see `PullRequestRecord.was_merged`.
- `in-review` and `bug` — established conventions (a pull request under active review; a bug-fix
  pull request, per this file's personal PR-conventions precedent) that no script currently reads,
  documented here so they're recognized (via `PullRequestRecord.identified_labels`) rather than
  silently falling through as an arbitrary, unrecognized label.

Any other label a real pull request carries is preserved in `PullRequestRecord.labels` (GitHub's own
label vocabulary is open-ended, and other automation on this repo may add labels this dashboard has
no reason to know about) but isn't specially interpreted.

**Creating a new plan.** [`.claude/skills/plan-create/SKILL.md`](../skills/plan-create/SKILL.md)
(`/plan-create <plan-id>`) automates the bootstrap flow above end to end: it gathers the plan's
scope (from an existing freeform roadmap doc to migrate, from named branches/PRs to cross-check
live, or from conversation), drafts a schema-conformant `plan.yaml`/`roadmap.md`, validates it the
same way `plan-dashboard` does, surfaces any structural judgment calls to you via a question rather
than guessing, then runs `save-plan.sh` and `/plan-dashboard` itself. Doing it by hand (the marker +
`save-plan.sh` flow above) still works — the skill is a convenience over that same path, not a
different one. It also creates a **tracking issue** as a coordination mailbox: any session can make
a structural change (new phase, deferring a track, etc.) directly to `plan.yaml`, but should ask the
user in the session first (e.g. via `AskUserQuestion`) rather than deciding unilaterally, and always
also comments on the tracking issue describing it once confirmed — that's the shared record other
sessions working the plan can check, and the user reviews structural changes there. (Falls back to an empty-commit,
permanently-draft PR instead if a repo has Issues disabled.) See `.claude/personal/plans/README.md`'s
"Proposing structural changes" section for the full convention. `session-start.sh`'s written header
reminds a session actively working an item to subscribe to the tracking issue too, so a change
another session makes reaches it in real time.

## Setup: overriding the default remote/branch/path

Skip this section if the zero-config default above is all you need. The three settings are
independent — override only the one(s) you actually need (e.g. just the remote, if your fork isn't
this clone's `origin` but the default branch/path are fine).

### Persistent local clone

Once per clone, never committed:

```bash
git config claude.personalNotesRemote <remote-name-or-url>   # optional, defaults to origin
git config claude.personalNotesBranch <your-branch-name>
git config claude.personalNotesPath   <path-on-that-branch>   # optional, defaults to
                                                                 # .claude/personal/cram-notes.md
```

Push your notes file to that branch on that remote (any branch name, any path — it never merges
anywhere), e.g. by running the branch-creation script with overrides:

```bash
CLAUDE_PERSONAL_NOTES_REMOTE=<remote-name-or-url> \
  CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name> CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch> \
  "$CLAUDE_PROJECT_DIR/.claude/hooks/create-personal-notes-branch.sh"
```

### Cloud/web sessions (fresh clone every time)

Push your notes file exactly as above first. Then wire the environment variables into your session
environment's configuration — which of the two options below applies depends on what your specific
environment offers:

### Option A: your environment has a persistent environment-variable list

Copy [`personal-notes.env.example`](./personal-notes.env.example) into that list, with your own
values substituted:

```
CLAUDE_PERSONAL_NOTES_REMOTE=<remote-name-or-url>
CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>
```

`session-start.sh` reads these directly — nothing else to configure.

### Option B: your environment has a "setup script" (arbitrary commands run on every fresh clone)

Set the same variables however that setup script can see them (its own env-var mechanism, or
literal `export` lines above the call), then run
[`configure-personal-notes.sh`](./configure-personal-notes.sh), e.g.:

```bash
export CLAUDE_PERSONAL_NOTES_REMOTE=<remote-name-or-url>   # optional
export CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
export CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>   # optional
"$CLAUDE_PROJECT_DIR/.claude/hooks/configure-personal-notes.sh"
```

This seeds the fresh clone's git config from those variables, so `session-start.sh` finds them
exactly as it would for a persistent local clone. It's a no-op if none of the three are set, so
it's safe to include even before you've opted in.

See your environment provider's docs for exactly where to paste a setup script or persistent
environment variables (for Claude Code on the web: <https://code.claude.com/docs/en/claude-code-on-the-web>).

## Verifying it worked

Start a fresh session and check whether `CLAUDE.local.md` exists at the project root with your
notes content. To check the mechanics without waiting for a real session boot, run the hook
directly:

```bash
"$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh" && cat CLAUDE.local.md
```

## Safety

- No-op in effect for anyone who never creates the `claude/personal-notes` branch (or an override
  target) on either the resolved remote or its upstream-tracking fallback: `git fetch` finds
  nothing on either, so nothing gets written.
- Never merges anything: the hook only ever *reads* the resolved branch, off `FETCH_HEAD` via `git
  show` (not a `<remote>/<branch>` ref, since a URL-form remote creates no tracking ref for one). It
  never checks the branch out or merges it into your working branch.
- `create-personal-notes-branch.sh`, `save-personal-notes.sh` and `save-pr-progress.sh` never touch
  your current branch or working tree either — all three do their work in a scratch worktree.
  `create-personal-notes-branch.sh` refuses to run if the target branch already exists locally, on
  the resolved remote, or on the upstream-tracking fallback remote (see above); the two save scripts
  are each a no-op if there's nothing new to push.
- The sync headers `session-start.sh` writes are never themselves pushed back: `save-personal-notes.sh`
  and `save-pr-progress.sh` each extract only the content between their own markers
  (`BEGIN-PERSONAL-NOTES`/`END-PERSONAL-NOTES` and `BEGIN-PR-PROGRESS`/`END-PR-PROGRESS`
  respectively) before committing, so neither header, and neither script's section, can ever leak
  into the other's saved content.
- PR-progress content is never merged, by construction: it is written only to the branch-keyed path
  on the personal-notes branch, never to any file tracked on the PR branch itself, so there is no
  code path by which it could end up in a commit that gets merged.
- `CLAUDE.local.md` is gitignored, so populated notes can't accidentally end up in a commit on any
  branch, including this one.
- All four scripts always operate on *this* repo's project root specifically, never wherever they
  happen to be invoked from: a `SessionStart` hook's cwd isn't guaranteed to be the project root,
  and these scripts are also meant to be run directly. Each resolves its own location on disk
  (`resolve-personal-notes-config.sh`'s own path, not `$CLAUDE_PROJECT_DIR` or the caller's cwd,
  since neither is guaranteed) to find the project root deterministically, then both `cd`s there for
  every git operation and reads/writes `CLAUDE.local.md` at that exact path — so it's never created
  in, or read from, some other directory, even one that's a subdirectory of a *different* repo.
- Safe to re-run: `session-start.sh` only ever overwrites `CLAUDE.local.md`, and does nothing if
  the resolved branch or path isn't reachable (e.g. a fresh clone, or a fork that never created
  it).
- Coexists with your own hooks: Claude Code merges `SessionStart` hook arrays across all settings
  layers by concatenation, not override, so this hook runs alongside — never instead of — any
  `SessionStart` hook you already have configured for yourself.
