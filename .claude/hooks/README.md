# Personal Claude Code notes hook

An opt-in `SessionStart` hook that populates `CLAUDE.local.md` — which Claude Code already loads
automatically as project memory, and which is gitignored — from a personal branch you name for
yourself on `origin`, so your own workflow preferences ("always open my PRs as drafts," "never
touch branch X directly," etc.) persist across sessions without ever being committed to a shared
branch.

It works out of the box with no config at all: it reads from a branch named `claude/personal-notes`
on `origin` unless you tell it otherwise. Run [`create-personal-notes-branch.sh`](./create-personal-notes-branch.sh)
once to create that branch with an empty notes file, and every session from then on picks it up
automatically. It is collision-free for multiple contributors sharing one `origin` remote if you
each override the branch name via your own config instead of relying on the shared default.

## How it decides what to read

`session-start.sh` looks for a branch name in this order, first one found wins:

1. **`git config claude.personalNotesBranch`** — local to one clone's `.git/config`.
2. **`CLAUDE_PERSONAL_NOTES_BRANCH` environment variable** — used only if the git config isn't set.
3. **`claude/personal-notes`** — the zero-config default, used if neither of the above is set.

The path on that branch follows the same precedence (`claude.personalNotesPath` git config, then
`CLAUDE_PERSONAL_NOTES_PATH` env var, then the default `.claude/personal/cram-notes.md`).

The hook is still a no-op in effect for anyone who never creates the branch it resolves to: `git
fetch` finds nothing, so it exits without writing `CLAUDE.local.md`.

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
`.claude/personal/cram-notes.md`, without touching your current branch or working tree. Then edit
that file (see below) to add your own notes. Every new Claude Code session — local or fresh-clone —
now runs the hook automatically and writes `CLAUDE.local.md` from that branch, with no further
configuration needed.

To edit your notes after the branch exists:

```bash
git fetch origin claude/personal-notes
git worktree add /tmp/personal-notes origin/claude/personal-notes
$EDITOR /tmp/personal-notes/.claude/personal/cram-notes.md
git -C /tmp/personal-notes commit -am "update personal notes"
git -C /tmp/personal-notes push origin HEAD:claude/personal-notes
git worktree remove /tmp/personal-notes
```

## Setup: overriding the default branch/path

Skip this section if the zero-config default above is all you need.

### Persistent local clone

Once per clone, never committed:

```bash
git config claude.personalNotesBranch <your-branch-name>
git config claude.personalNotesPath   <path-on-that-branch>   # optional, defaults to
                                                                 # .claude/personal/cram-notes.md
```

Push your notes file to that branch on `origin` (any branch name, any path — it never merges
anywhere), e.g. by running the branch-creation script with overrides:

```bash
CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name> CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch> \
  "$CLAUDE_PROJECT_DIR/.claude/hooks/create-personal-notes-branch.sh"
```

### Cloud/web sessions (fresh clone every time)

Push your notes file to `origin` exactly as above first. Then wire the environment variable into
your session environment's configuration — which of the two options below applies depends on what
your specific environment offers:

### Option A: your environment has a persistent environment-variable list

Copy [`personal-notes.env.example`](./personal-notes.env.example) into that list, with your own
values substituted:

```
CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>
```

`session-start.sh` reads these directly — nothing else to configure.

### Option B: your environment has a "setup script" (arbitrary commands run on every fresh clone)

Set the same two variables however that setup script can see them (its own env-var mechanism, or
literal `export` lines above the call), then run
[`configure-personal-notes.sh`](./configure-personal-notes.sh), e.g.:

```bash
export CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
export CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>   # optional
"$CLAUDE_PROJECT_DIR/.claude/hooks/configure-personal-notes.sh"
```

This seeds the fresh clone's git config from those variables, so `session-start.sh` finds them
exactly as it would for a persistent local clone. It's a no-op if
`CLAUDE_PERSONAL_NOTES_BRANCH` isn't set, so it's safe to include even before you've opted in.

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
  target): `git fetch` finds nothing, so nothing gets written.
- Never merges anything: the hook only ever *reads* the resolved branch via `git show`. It never
  checks it out or merges it into your working branch.
- `create-personal-notes-branch.sh` never touches your current branch or working tree either — it
  does its work in a scratch worktree and refuses to run if the target branch already exists
  locally or on `origin`.
- `CLAUDE.local.md` is gitignored, so populated notes can't accidentally end up in a commit on any
  branch, including this one.
- Safe to re-run: `session-start.sh` only ever overwrites `CLAUDE.local.md`, and does nothing if
  the resolved branch or path isn't reachable (e.g. a fresh clone, or a fork that never created
  it).
- Coexists with your own hooks: Claude Code merges `SessionStart` hook arrays across all settings
  layers by concatenation, not override, so this hook runs alongside — never instead of — any
  `SessionStart` hook you already have configured for yourself.
