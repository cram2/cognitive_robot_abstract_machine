#!/bin/bash
set -euo pipefail

# Personal-only Claude Code notes for abdelrhmanbassiouny's fork.
#
# These notes (draft-PR-by-default, bug-fix labeling, etc.) live on the
# `claude/personal-notes` branch only and must never reach `main`. This hook
# fetches that branch's file straight from the remote (without checking it out
# or merging it into the working branch) and writes it to CLAUDE.local.md,
# which Claude Code loads automatically as project memory but which stays out
# of git history for every other branch (see .gitignore).
#
# Safe to re-run: it only ever overwrites CLAUDE.local.md, and does nothing if
# the personal-notes branch or file isn't reachable (e.g. a fresh clone that
# doesn't have it, or a fork that never created it).

NOTES_BRANCH="origin/claude/personal-notes"
NOTES_PATH=".claude/personal/cram-notes.md"

git fetch origin claude/personal-notes --quiet 2>/dev/null || exit 0

if git cat-file -e "${NOTES_BRANCH}:${NOTES_PATH}" 2>/dev/null; then
  git show "${NOTES_BRANCH}:${NOTES_PATH}" > CLAUDE.local.md
fi
