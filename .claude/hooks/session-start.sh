#!/bin/bash
set -euo pipefail

# Generic personal Claude Code notes hook.
#
# Populates CLAUDE.local.md (gitignored, never committed on any branch) from a
# personal branch on a remote (default: `origin`). No-op in effect for anyone
# who never creates that branch (default or overridden), and collision-free
# for multiple contributors sharing one remote if each overrides the branch
# name via local config instead of relying on the shared default.
#
# Always writes to the project root, never wherever this happens to be
# invoked from: a SessionStart hook's cwd isn't guaranteed to be the project
# root, so a bare `CLAUDE.local.md` could silently land in the wrong place
# (and Claude Code only auto-loads it from the root) - see CLAUDE_LOCAL_MD in
# ./resolve-personal-notes-config.sh for how that's resolved deterministically.
#
# Works out of the box, zero config: it looks for a branch named
# `claude/personal-notes` on `origin` and, if found, reads
# `.claude/personal/cram-notes.md` off it into CLAUDE.local.md.
# ./create-personal-notes-branch.sh creates that branch (with an empty notes
# file) for anyone who doesn't have it yet.
#
# Override the remote/branch/path per clone, locally (never committed):
#   git config claude.personalNotesRemote <remote-name-or-url>   # optional
#   git config claude.personalNotesBranch <your-branch-name>
#   git config claude.personalNotesPath   <path-on-that-branch>   # optional
#
# The remote defaults to `origin`, but only matters when your own notes live
# somewhere other than the clone's `origin` - e.g. some session environments
# name the upstream repo `origin` and your own fork something else. Set it to
# either a remote already configured in this clone (by name) or a raw git URL
# (`https://github.com/<you>/<repo>`) - `git fetch`/`git push` accept both, and
# a URL needs no `git remote add` first, so it works even in a clone that's
# never heard of your fork. See ./README.md for when to use which form.
#
# git config is per-clone, so it's the wrong mechanism anywhere sessions start
# from a fresh clone every time (e.g. cloud/web sessions) - there's no
# persistent .git/config for it to live in. For that case, override via
# persistent environment variables instead (configured once at the environment
# level, outside the repo, so they survive every fresh clone):
#   CLAUDE_PERSONAL_NOTES_REMOTE=<remote-name-or-url>   # optional
#   CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
#   CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>   # optional
# See ./README.md for exactly how to wire these into a cloud environment.
# Precedence: git config > environment variable > the zero-config default, so
# a local or environment-level override always wins over it.
#
# Safe to re-run: it only ever overwrites CLAUDE.local.md, and does nothing if
# the configured (or default) branch or path isn't reachable (e.g. a fresh
# clone, or a fork that never created it).
#
# Remote fallback: if NOTES_REMOTE doesn't have the branch, this also tries
# the current branch's own upstream remote (if it has one, and it differs
# from NOTES_REMOTE) before giving up - see fetch_personal_notes_branch in
# ./resolve-personal-notes-config.sh. Covers a clone whose checked-out branch
# already tracks your fork under some other remote name, with no config
# needed. The written header always names whichever remote actually served
# the notes, so it's clear which one was used.
#
# Editing your notes: the written CLAUDE.local.md starts with a short header
# (see below) naming the resolved branch/path and pointing at
# ./save-personal-notes.sh. Since Claude Code loads CLAUDE.local.md as project
# memory every session, that header is always in context - so asking Claude to
# "edit my personal notes" needs no other setup: it edits the notes between
# the BEGIN-PERSONAL-NOTES/END-PERSONAL-NOTES markers, then runs the save
# script to push the change back.
#
# PR progress: on any branch with a sensible "current PR" (i.e. not the
# default branch, a detached HEAD, or the personal-notes branch itself - see
# pr_progress_path in ./resolve-personal-notes-config.sh), CLAUDE.local.md
# also gets a second section for that branch's plan/progress/next-steps,
# keyed to the branch name and stored on the personal-notes branch just like
# the notes above - so it is never committed to the PR branch itself, and
# survives session restarts automatically. Always present (as a scaffold, if
# nothing's been saved yet) on such a branch, so the agent is nudged to
# initialize and maintain it from the start. See ./save-pr-progress.sh.
#
# How this script gets invoked (see ../settings.json): Claude Code registers it
# as a SessionStart hook via `$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh`.
# CLAUDE_PROJECT_DIR is an env var Claude Code itself injects into every hook
# command's environment, resolving to this project's root - so that path is
# correct regardless of Claude Code's own cwd when it runs the hook.
#
# Coexistence with your own settings: Claude Code merges the `hooks` arrays
# across all settings layers (managed > CLI args > .claude/settings.local.json
# > .claude/settings.json (this repo's, committed) > ~/.claude/settings.json)
# by concatenation, not override. So this SessionStart hook runs alongside -
# never instead of - any SessionStart hook you already have configured for
# yourself. settings.json is strict JSON with no comment support, which is why
# this explanation lives here instead of there.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

fetch_personal_notes_branch || exit 0

# FETCH_HEAD, not "${ACTIVE_NOTES_REMOTE}/${NOTES_BRANCH}": a URL-form remote
# creates no remote-tracking ref, but FETCH_HEAD always points at what was
# just fetched, whether the serving remote was a name or a raw URL.

OUTPUT_FILE="$(mktemp)"
WROTE_ANYTHING=0

if git cat-file -e "FETCH_HEAD:${NOTES_PATH}" 2>/dev/null; then
  cat <<HEADER >> "${OUTPUT_FILE}"
<!--
Personal notes, synced from '${NOTES_BRANCH}' (${NOTES_PATH}) on remote
'${ACTIVE_NOTES_REMOTE}' by session-start.sh.
To edit: change the notes between the markers below, then run
  "\$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
to push the change back. This header and the markers are regenerated every
session from git config/environment/default (plus a same-branch-upstream
fallback) - editing them has no effect; only content between the markers is
ever saved.
-->
<!-- BEGIN-PERSONAL-NOTES -->
HEADER
  git show "FETCH_HEAD:${NOTES_PATH}" >> "${OUTPUT_FILE}"
  echo "<!-- END-PERSONAL-NOTES -->" >> "${OUTPUT_FILE}"
  WROTE_ANYTHING=1
fi

PROGRESS_PATH="$(pr_progress_path || true)"
if [ -n "${PROGRESS_PATH}" ]; then
  [ "${WROTE_ANYTHING}" = "1" ] && printf '\n' >> "${OUTPUT_FILE}"
  cat <<PROGRESS_HEADER >> "${OUTPUT_FILE}"
<!--
PR progress for branch '$(git rev-parse --abbrev-ref HEAD)', synced from
'${NOTES_BRANCH}' (${PROGRESS_PATH}) on remote '${ACTIVE_NOTES_REMOTE}' by
session-start.sh. Maintain the current plan, what's done, and what's next
here throughout work on this PR. It is never merged: it lives only on
'${NOTES_BRANCH}', never on this branch. A stale file left behind after the
PR merges is harmless (just unread from then on) - delete it directly on
'${NOTES_BRANCH}' if you want to tidy it up.
To edit: change the notes between the markers below, then run
  "\$CLAUDE_PROJECT_DIR/.claude/hooks/save-pr-progress.sh"
to push the change back. This header and the markers are regenerated every
session - editing them has no effect; only content between the markers is
ever saved.
-->
<!-- BEGIN-PR-PROGRESS -->
PROGRESS_HEADER
  if git cat-file -e "FETCH_HEAD:${PROGRESS_PATH}" 2>/dev/null; then
    git show "FETCH_HEAD:${PROGRESS_PATH}" >> "${OUTPUT_FILE}"
  else
    cat <<'SCAFFOLD' >> "${OUTPUT_FILE}"
No progress recorded yet for this branch. Initialize it now: a short plan,
what's done so far, and what's next. Keep it current as you work.
SCAFFOLD
  fi
  echo "<!-- END-PR-PROGRESS -->" >> "${OUTPUT_FILE}"
  WROTE_ANYTHING=1
fi

if [ "${WROTE_ANYTHING}" = "1" ]; then
  mv "${OUTPUT_FILE}" "${CLAUDE_LOCAL_MD}"
else
  rm -f "${OUTPUT_FILE}"
fi
