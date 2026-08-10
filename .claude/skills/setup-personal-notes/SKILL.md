---
name: setup-personal-notes
description: One-time setup for this repo's personal-notes tooling - the personal-notes branch, its remote/branch/path resolution, the plan-dashboard dependencies, and the CLAUDE.local.md the SessionStart hook writes. Invoke as "/setup-personal-notes". Use when someone is setting up this repo for Claude Code for the first time, when personal notes/PR progress/plan dashboards aren't working, or when another skill's prerequisite check reports the setup is incomplete.
allowed-tools: Bash, Read, Write, AskUserQuestion, mcp__github__get_me, mcp__github__get_label
---

# Set up personal notes

Gets a clone from "I have a fork of this repo and nothing else" to "personal
notes, PR progress and plan dashboards all work," asking one question per
decision that is genuinely the user's, with a default they can accept as-is.

**Fast path first.** Steps 2 to 7 are each conditional on
[`check-setup.sh`](../../hooks/check-setup.sh) reporting that specific thing
missing. If it reports nothing missing, go straight to step 8 — the label check
the script can't perform — and stop there. A re-run on an already-set-up clone
must be a near-no-op, not an interrogation.

**Never guess a value that is the user's to choose.** Where their notes live
and what goes in them are personal; ask, with a default. Everything else
(installing a documented dependency, running a read-only script) is mechanical
— just do it.

## 1. Find out what is actually missing

```bash
source .claude/hooks/resolve-personal-notes-config.sh
bash "${CHECK_SETUP_SCRIPT}" || true
```

The script prints one tab-separated `<check>` / `<status>` / `<detail>` row per
check and exits non-zero if any row is `needs-setup` (`|| true` above keeps a
non-zero exit from ending the block early — the rows are the point, not the
status). Read its own header comment for what each check means.

**If it exited 0:** tell the user everything it covers is already set up, list
what was found (the remote, branch and path in play, and that dependencies and
`CLAUDE.local.md` are in place), then go to step 8 and stop. Do not ask
anything else. Do not re-verify by hand what the script just verified.

**Otherwise:** work through the `needs-setup` rows in the order printed —
they're ordered so an earlier fix can be a prerequisite of a later one — using
the steps below. Skip the step for any check already `ok`.

## 2. `tooling_files` — the fork is missing the tooling itself

Not fixable from here: their default branch predates the `.claude/` tooling.
Tell them to merge the current default branch into their fork and re-run. Stop;
every later step depends on files this clone doesn't have.

## 3. `session_start_hook` / `claude_local_md_ignored` — a broken checkout

Both come from committed files (`.claude/settings.json`, `.gitignore`), so a
failure means this checkout diverges from the default branch rather than
anything personal being unconfigured. Say which one is off and what restores it
(`git checkout <default-branch> -- .claude/settings.json` / `.gitignore`); ask
before changing a tracked file, since the divergence may be deliberate.

## 4. `notes_branch` — where the notes should live

The decision the whole setup turns on: **is the notes remote their own fork, or
a shared upstream they can't push to?** The script can't tell (a git remote URL
doesn't say who owns it), so establish it here.

Read the `notes_remote_url` row, and get the user's GitHub login:

```
mcp__github__get_me
```

If the URL's owner is that login, the resolved remote is already their own fork
— go straight to creating the branch. If it isn't (the common case: `origin` is
the shared upstream), the notes would be pushed somewhere they don't own. Ask
where their notes should live, offering `https://github.com/<their-login>/<repo>`
as the default, and note that they may also use a remote name already in the
clone, or a different repository entirely.

Apply the answer, if it differs from what resolved, before creating anything:

```bash
git config claude.personalNotesRemote <chosen-remote-or-url>
```

**Then tell them plainly that `git config` alone is not enough for sessions
that clone fresh every time** (Claude Code on the web, and any cloud
environment): the clone — and this config with it — is gone next session. For
those, the same value has to be set as a persistent environment variable at the
environment level, which no command can do from inside the session. Give them
the exact lines to paste, and only the ones that differ from the defaults:

```
CLAUDE_PERSONAL_NOTES_REMOTE=<chosen-remote-or-url>
CLAUDE_PERSONAL_NOTES_BRANCH=<branch, only if not claude/personal-notes>
CLAUDE_PERSONAL_NOTES_PATH=<path, only if not .claude/personal/cram-notes.md>
```

Point them at their environment's own docs for where that list lives (for
Claude Code on the web: <https://code.claude.com/docs/en/claude-code-on-the-web>),
and at [`personal-notes.env.example`](../../hooks/personal-notes.env.example)
and [`configure-personal-notes.sh`](../../hooks/configure-personal-notes.sh)
for the two shapes that wiring takes. This is the one part of the setup that
stays the user's to finish — say so, rather than leaving them to discover it
next session when their notes have vanished.

Only the remote is worth asking about by default. The branch and path have
working defaults; mention that both are overridable (same three-way precedence,
via `claude.personalNotesBranch` / `claude.personalNotesPath`) and only ask if
they want something else — for example, a distinct branch name so several
people sharing one remote don't collide.

Then create it:

```bash
"${PROJECT_ROOT}/.claude/hooks/create-personal-notes-branch.sh"
```

It creates the branch with an empty notes file, on the resolved remote, in a
scratch worktree — the current branch and working tree are untouched. It
refuses to touch a branch that already exists anywhere it can see, so it can
never overwrite notes that are already there.

## 5. `notes_file` — the notes file, and what starts in it

Offer to seed the new notes with
[`starter-notes.md`](./starter-notes.md) — working conventions for pull
requests, review comments and progress tracking — defaulting to yes, and making
clear it's a starting point they own and can edit or discard. If they decline,
an empty file is a perfectly good state; `create-personal-notes-branch.sh`
already made one.

To seed it:

```bash
"${PROJECT_ROOT}/.claude/hooks/write-personal-notes-file.sh" \
  --source "${STARTER_NOTES_FILE}" \
  --destination "${NOTES_PATH}" \
  --message "Initialize personal notes from the starter template"
```

That helper is idempotent — it pushes nothing if the content already matches.

## 6. `dashboard_dependencies` — what the dashboards need

Mechanical, not a preference. Say what's missing and install it:

```bash
pip install -r "${PLAN_DASHBOARD_REQUIREMENTS_FILE}"
```

If the install fails (no network, a managed environment, a read-only
interpreter), don't retry blindly — report the failure and the command, and
carry on with the rest of the setup. Everything except plan dashboards works
without these.

## 7. `claude_local_md` — make this session pick it up

The `SessionStart` hook writes `CLAUDE.local.md`, so it has already run for
this session, before any of the above existed. Run it now so the notes are
live immediately instead of next session:

```bash
"${PROJECT_ROOT}/.claude/hooks/session-start.sh"
```

Its output is the same summary a real session start prints. Note for the user
that the notes it just wrote are not yet in this conversation's context — this
session will pick them up on its next start, while everything else is usable
right away.

## 8. The pull request labels this tooling applies

`check-setup.sh` can't see this one — labels live behind the GitHub API, which
is reachable only from a session — so it is checked here instead.

A fresh fork starts with GitHub's own default labels and none of the ones this
tooling uses. That matters in two different ways, and only the first is
cosmetic:

- `merged` is *read* by `build_dashboard.py`, which treats it as proof a pull
  request landed even when GitHub's merge API never recorded it. Without the
  label, such an item reads as closed-unmerged on every dashboard.
- `bug` and `in-review` are *applied* by the conventions in
  [`starter-notes.md`](./starter-notes.md) ("bug-fix PRs must always carry the
  `bug` label"). Applying a label a repository doesn't have fails, so a session
  following those notes hits an error at the worst moment — mid-way through
  opening a pull request.

Check all three against the repository the user opens pull requests against —
their fork, normally this clone's `origin`, and the same repository step 4
settled on if that differed:

```
mcp__github__get_label   # per label: merged, bug, in-review
```

A `404` means it's missing. If any are, say which and offer to create them —
ask first, since this writes to their repository.

There is no create-label tool in the GitHub MCP server, so create them through
the `gh` CLI when it's available:

```bash
gh label create <name> --repo <owner>/<repo> --description "<what it means>"
```

If `gh` isn't installed, don't pretend it is: give the user the exact names and
point them at `https://github.com/<owner>/<repo>/labels`, and say which of the
three are missing. Missing labels don't block anything else in this setup —
report and carry on.

## 9. Confirm, by re-running the check

```bash
bash "${CHECK_SETUP_SCRIPT}" || true
```

Report what the second run says, not what you expect it to say. If a row is
still `needs-setup`, say which and why — a setup that half-worked and was
reported as finished is worse than one that reported the truth.

Then tell them what they can now do, briefly: ask for notes edits in any
session, track a PR's progress, and — if the dependencies installed —
`/plan-create` and `/plan-dashboard`, pointing at
[`example-walkthrough.md`](../plan-dashboard/example-walkthrough.md) for the
worked example.

## What this skill must never do

- **Never invent where someone's notes live.** A wrong remote pushes personal
  notes to a repository they didn't choose. Ask, always, unless the resolved
  remote is already provably their own.
- **Never claim the environment-variable half is done.** No command inside a
  session can persist an environment variable for the next fresh clone.
- **Never touch an existing personal-notes branch's contents.** Every write
  here is either creating a branch that provably doesn't exist yet, or a
  no-op-if-unchanged push of a file the user just asked for.
- **Never create labels in someone's repository unasked.** Labels are visible
  to everyone who sees the repository; report what's missing and offer.
