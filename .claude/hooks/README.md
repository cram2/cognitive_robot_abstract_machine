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
