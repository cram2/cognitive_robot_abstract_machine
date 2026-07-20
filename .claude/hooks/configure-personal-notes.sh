#!/bin/bash
set -euo pipefail

# Generic environment-setup script for the personal-notes SessionStart hook
# (./session-start.sh). Paste this into your cloud/web session environment's
# "setup script" (a field that runs arbitrary commands on every fresh clone,
# as opposed to a persistent environment-variable list) if that's what your
# environment offers instead of persistent environment variables.
#
# It reads the same three variable names as ./personal-notes.env.example
# (CLAUDE_PERSONAL_NOTES_REMOTE, CLAUDE_PERSONAL_NOTES_BRANCH,
# CLAUDE_PERSONAL_NOTES_PATH) and seeds them into this fresh clone's git
# config, so session-start.sh finds them there exactly as it would for a
# persistent local clone where you ran `git config` yourself once. Set the
# variables however your environment lets you (its own
# persistent-environment-variable feature, or literal `export` lines pasted
# above the call to this script) - this script only turns them into git
# config, it does not define them itself, so it stays generic: nothing here
# is specific to any one contributor's branch name or fork.
#
# No-op (exit 0, no git config written) if none of the three are set, so it's
# always safe to include in a setup script even before you've opted in. Each
# of the three is otherwise independent: set only CLAUDE_PERSONAL_NOTES_REMOTE
# if you just need to point at a fork that isn't this clone's `origin`, while
# keeping the default branch/path.
#
# Safe to re-run: `git config` simply overwrites the previous value.

NOTES_REMOTE="${CLAUDE_PERSONAL_NOTES_REMOTE:-}"
NOTES_BRANCH="${CLAUDE_PERSONAL_NOTES_BRANCH:-}"
NOTES_PATH="${CLAUDE_PERSONAL_NOTES_PATH:-}"

if [ -z "${NOTES_REMOTE}" ] && [ -z "${NOTES_BRANCH}" ] && [ -z "${NOTES_PATH}" ]; then
  exit 0
fi

[ -z "${NOTES_REMOTE}" ] || git config claude.personalNotesRemote "${NOTES_REMOTE}"
[ -z "${NOTES_BRANCH}" ] || git config claude.personalNotesBranch "${NOTES_BRANCH}"
[ -z "${NOTES_PATH}" ] || git config claude.personalNotesPath "${NOTES_PATH}"
