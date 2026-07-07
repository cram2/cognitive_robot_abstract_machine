#!/bin/bash
set -euo pipefail

# Generic, opt-in personal Claude Code notes hook.
#
# Populates CLAUDE.local.md (gitignored, never committed on any branch) from a
# personal branch on `origin` that each contributor names for themselves via
# local git config, so this stays a complete no-op for anyone who hasn't opted
# in, and collision-free for multiple contributors sharing one origin remote
# (each points at their own branch name instead of one hardcoded literal).
#
# Opt in once, locally, per clone (never committed):
#   git config claude.personalNotesBranch <your-branch-name>
#   git config claude.personalNotesPath   <path-on-that-branch>   # optional
#
# With claude.personalNotesBranch unset, this exits immediately: no fetch, no
# write, no effect for anyone who hasn't configured it.
#
# Safe to re-run: it only ever overwrites CLAUDE.local.md, and does nothing if
# the configured branch or path isn't reachable (e.g. a fresh clone, or a fork
# that never created it).

NOTES_BRANCH="$(git config --get claude.personalNotesBranch || true)"
[ -n "${NOTES_BRANCH}" ] || exit 0

NOTES_PATH="$(git config --get claude.personalNotesPath || true)"
NOTES_PATH="${NOTES_PATH:-.claude/personal/cram-notes.md}"

git fetch origin "${NOTES_BRANCH}" --quiet 2>/dev/null || exit 0

if git cat-file -e "origin/${NOTES_BRANCH}:${NOTES_PATH}" 2>/dev/null; then
  git show "origin/${NOTES_BRANCH}:${NOTES_PATH}" > CLAUDE.local.md
fi
