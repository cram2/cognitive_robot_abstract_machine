# Personal Claude Code notes hook

A `SessionStart` hook that writes your own workflow preferences into `CLAUDE.local.md` — which
Claude Code already loads as project memory, and which is gitignored — from a branch on a remote
you own. Your notes (e.g. "always open my PRs as drafts") follow you across sessions and are never
committed to a shared branch.

> **Rather see it in action than read about it?**
> [`example-walkthrough.md`](../skills/plan-dashboard/example-walkthrough.md) is a short worked
> example of the whole thing in use, from a plan-mode idea to a published dashboard. Or just run
> `/setup-personal-notes` in a session and come back here when you want the details.

Three kinds of content, stored the same way and never merged anywhere:

| Section in `CLAUDE.local.md` | Holds | Stored on the notes branch at | Saved by |
| --- | --- | --- | --- |
| Personal notes | Your standing preferences | `.claude/personal/cram-notes.md` | [`save-personal-notes.sh`](./save-personal-notes.sh) |
| PR progress | Plan, progress and next steps for the current branch's PR | `.claude/personal/pr-progress/<branch>.md` | [`save-pr-progress.sh`](./save-pr-progress.sh) |
| Plan | A multi-PR initiative's manifest and roadmap | `.claude/personal/plans/<plan-id>/` | [`save-plan.sh`](./save-plan.sh) |

The PR-progress section appears on any branch that isn't the default branch, a detached `HEAD`, or
the notes branch itself, as an empty scaffold even before anything has been saved.

Your local Claude Code settings ride along on the same branch too, as a whole file rather than a
section — see [Syncing your local Claude Code settings](#syncing-your-local-claude-code-settings).
Every run prints a summary of what it found and wrote, so a session never has to infer it.

## Quick start

1. Run `/setup-personal-notes` in any Claude Code session on this repo.
2. Answer the one question it asks — which remote your notes live on — or accept the default.
3. Done. Every session from now on writes `CLAUDE.local.md` automatically.

It is safe to re-run: on a clone that's already set up it reports what it found and asks nothing.
You don't have to run it first either — `/plan-create`, `/plan-dashboard`, `/plan-item-kickoff` and
`/plan-item-resolve` each offer it if something is missing.

To do the same by hand:

```bash
"$CLAUDE_PROJECT_DIR/.claude/hooks/create-personal-notes-branch.sh"           # create the branch
"$CLAUDE_PROJECT_DIR/.claude/hooks/check-setup.sh"                            # inspect, change nothing
"$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh" && cat CLAUDE.local.md   # verify
```

`check-setup.sh` prints one row per check and exits non-zero if anything still needs doing.

Every session start prints its own summary, so neither of the two things below has to be
remembered. Its `setup:` line runs `check-setup.sh` and names any check that still needs setup;
its `plan:` line distinguishes *no plans are tracked here* from *plans exist and no item tracks
this branch* — the second being the cue to add the item before starting work, not after. Both
lines appear only once a personal-notes branch exists, so a clone that uses none of this stays
silent.

## Editing your notes

- **Ask Claude** — *"add \<X\> to my personal notes"*, *"edit my personal notes"*. Nothing else to
  explain: the header the hook writes into `CLAUDE.local.md` is always in context and names the
  branch, the path, and the script that pushes changes back.
- **By hand** — edit `CLAUDE.local.md` between that section's `BEGIN-`/`END-` markers, then run the
  save script from the table above.

Only content between the markers is ever saved. Headers and markers are regenerated every session,
so editing them has no effect.

## Syncing your local Claude Code settings

The same branch can carry your personal `.claude/settings.local.json` — the file Claude Code reads
as this project's local settings (permission rules, environment variables, anything else in its
settings schema). Store it on the notes branch at `.claude/personal/settings.local.json` and each
session start copies it into the project root, so permissions you'd otherwise re-grant in every
fresh clone follow you around. It is gitignored; the committed `.claude/settings.json` stays the
shared, team-wide one.

It gets no header or markers — strict JSON has nowhere to put them — so the file is copied whole.

**Your local edits are never overwritten.** Claude Code writes to that same file whenever you grant
a permission with "don't ask again", so the hook writes only when the file is missing or still
identical to what it last synced (tracked in the gitignored `.claude/.personal-settings-sync-hash`).
Otherwise it keeps your version and says so in its summary. To push your edits up and let syncing
resume:

```bash
"$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-settings.sh"
```

Or ask Claude — *"save my Claude settings"* — the same way as your notes.

## Configuration

Three independent settings, each resolved as **git config**, then **environment variable**, then
**default**:

| Setting | git config | Environment variable | Default |
| --- | --- | --- | --- |
| Remote | `claude.personalNotesRemote` | `CLAUDE_PERSONAL_NOTES_REMOTE` | `origin` |
| Branch | `claude.personalNotesBranch` | `CLAUDE_PERSONAL_NOTES_BRANCH` | `claude/personal-notes` |
| Path | `claude.personalNotesPath` | `CLAUDE_PERSONAL_NOTES_PATH` | `.claude/personal/cram-notes.md` |

Override only what you need:

- **Remote** — when your notes don't live on this clone's `origin` (commonly: `origin` is the shared
  upstream and your fork is a differently-named remote, or isn't a remote at all). Takes a remote
  name (`myfork`) or a raw URL (`https://github.com/<you>/<repo>`); the URL form needs no
  `git remote add` first, so it works in a clone that has never heard of your fork.
- **Branch** — when several people share one remote and you don't want to collide on the default.
- **Path** — rarely needed.

### Where to put them

- **A persistent local clone** → `git config <key> <value>`, once per clone. Never committed.
- **A fresh clone every session** (cloud/web) → git config won't survive, so use the environment:
  - Your environment has a **persistent environment-variable list**: paste the variables in, per
    [`personal-notes.env.example`](./personal-notes.env.example). Nothing else to configure.
  - Your environment has a **setup script**: export the variables there, then call
    [`configure-personal-notes.sh`](./configure-personal-notes.sh) — it seeds the fresh clone's git
    config from them, and is a no-op if none are set.

For Claude Code on the web, see <https://code.claude.com/docs/en/claude-code-on-the-web> for where
either of those lives.

### Fallback: your branch's own upstream

If the resolved remote doesn't have the notes branch, the hook tries one more: the remote your
currently checked-out branch already tracks. That covers the common case of a fork added under some
name other than `origin`, with no configuration at all.

Reads use the fallback; `create-personal-notes-branch.sh` never creates there. `save-personal-notes.sh`
writes back to whichever remote actually served the notes, and the header always names it.

## Plan dashboards (multi-PR initiatives)

When one PR's progress note isn't enough — a stacked refactor, a multi-wave programme, anything
you'd otherwise write up as a one-off master-roadmap doc — a **plan** is a `plan.yaml` (waves,
tracks, and items with branch, PR number, status and dependencies) plus a sibling `roadmap.md` for
the narrative that doesn't belong in structured data.

- Worked example → [`example-walkthrough.md`](../skills/plan-dashboard/example-walkthrough.md).
- Field reference → [`plan-schema.md`](../skills/plan-dashboard/plan-schema.md).
- Create one → `/plan-create <plan-id>`.
- Publish or refresh → `/plan-dashboard [<plan-id>]`; no argument publishes the master index of
  every plan. It cross-checks every item against live GitHub PR/CI/review state, so a manifest
  can't silently go stale the way a hand-maintained roadmap doc could.
- Start or unblock one item → `/plan-item-kickoff <plan-id> <item-id>`,
  `/plan-item-resolve <plan-id> <item-id>`. Kickoff opens the item's branch and draft PR and
  marks it `in_progress` as soon as its plan is approved — via
  [`plan_item_bootstrap.py`](./plan_item_bootstrap.py), which you can also run by hand — so the
  manifest never says `not_started` while the work is underway.
- Recheck one for updates, without rereading it →
  [`plan-updates-since.sh`](./plan-updates-since.sh) `<plan-id> [--since <sha>]`. Every
  `session-start.sh` run stamps the notes-branch commit it just fetched (gitignored, at
  `.claude/.plan-state-sync-sha`), so this can diff the plan's directory from that stamp and print
  only the tracking-issue comments newer than it. Needs no Claude Code session: it prefers the `gh`
  CLI when installed, otherwise `GH_TOKEN`/`GITHUB_TOKEN` with `curl`.

**Auto-discovery.** If your branch is an item in some plan, that plan's `plan.yaml` and `roadmap.md`
are pulled into `CLAUDE.local.md` too, via a generated branch-to-plan index that `save-plan.sh`
regenerates from every manifest on each save, so it can't drift.

**Labels the dashboard reads**, all applied by this repo's convention rather than by GitHub itself:

- `merged` — the changes landed but GitHub's merge API never recorded it (branch pushed directly,
  PR then closed by hand). Treated exactly like a real merge.
- `bug` — marks the item with a `bug` chip in the dashboard's "What to do next" sidebar, in
  whichever action group it already belongs to, and is what the sidebar's "Bug fixes only" filter
  keeps. Being a bug fix is a property of the work, not a next action of its own, so it never
  moves an item into a group or out of one.
- `in-review` — recognized so it doesn't read as an unknown label; no script acts on it yet.

Any other label is preserved but not interpreted.

## Safety

- Does nothing until you create the notes branch: `git fetch` finds nothing, so `CLAUDE.local.md` is
  never written.
- Never merges and never checks anything out — the hook and `plan-updates-since.sh` only read the
  branch off `FETCH_HEAD`.
- Never touches your current branch or working tree: every script that writes works in a scratch
  worktree.
- `create-personal-notes-branch.sh` refuses to run if the branch already exists anywhere it can see,
  so it can't overwrite notes you already have. The save scripts are no-ops when nothing changed.
- Each save script extracts only its own markers, so neither the sync headers nor another section's
  content can leak into what it pushes.
- PR progress and plans can't be merged into a PR by construction: they're only ever written to the
  notes branch, never to a file tracked on your branch.
- Synced settings never silently replace local ones: `.claude/settings.local.json` is written only
  when it's missing or unchanged since the last sync, so "don't ask again" grants survive until you
  run `save-personal-settings.sh` yourself.
- `CLAUDE.local.md`, `.claude/settings.local.json` and the recheck stamp
  `.claude/.plan-state-sync-sha` are all gitignored.
- Always operates on this repo's project root, resolved from the scripts' own location on disk —
  not the caller's cwd, which a `SessionStart` hook can't rely on.
- Coexists with your own `SessionStart` hooks: Claude Code concatenates hook arrays across settings
  layers rather than overriding them.
