#!/bin/bash
set -euo pipefail

# Persists this clone's `.claude/settings.local.json` back onto the personal-notes
# branch (at `.claude/personal/settings.local.json`), so the next session's
# session-start.sh syncs the updated settings into every clone. This is the write
# half of the loop session-start.sh's settings block reads from - the counterpart
# of ./save-personal-notes.sh for Claude Code's own local settings rather than for
# your notes.
#
# Usage (from anywhere, after editing .claude/settings.local.json):
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-settings.sh"
#
# Run this whenever settings changed locally - including when Claude Code itself
# appended a permission rule after an "always allow". Until it is run, those edits
# stay local: session-start.sh deliberately refuses to overwrite settings that
# differ from the last synced content, so an unsaved edit is kept rather than
# silently replaced (see personal_settings_are_locally_modified in
# ./resolve-personal-notes-config.sh). Saving them makes them the new baseline, so
# syncing resumes from the next session on.
#
# The whole file is pushed verbatim - it is strict JSON, so unlike the notes and
# PR-progress files there are no markers to extract, and nothing else is ever
# written into it.
#
# Resolves the remote/branch exactly like the other hook scripts (git config >
# environment variable > the zero-config default, plus the same-branch-upstream
# fallback), by delegating the commit-and-push itself to
# ./write-personal-notes-file.sh.
#
# Safe to re-run: a no-op if the branch already carries identical settings. Does
# its work in a scratch worktree, so it never touches your current branch or
# working tree. Fails with a clear message if the personal-notes branch doesn't
# exist yet on any resolved remote - run ./create-personal-notes-branch.sh first
# in that case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

if [ ! -f "${LOCAL_SETTINGS_JSON}" ]; then
  echo "No ${LOCAL_SETTINGS_RELATIVE_PATH} at the project root (${LOCAL_SETTINGS_JSON}) - nothing to save." >&2
  exit 1
fi

"${SCRIPT_DIR}/write-personal-notes-file.sh" \
  --source "${LOCAL_SETTINGS_JSON}" \
  --destination "${PERSONAL_SETTINGS_PATH}" \
  --message "Update personal Claude Code settings"

# Stamped after the push (including its no-op path, where the branch already
# matches): either way the local content is now what the branch carries, which is
# exactly what the stamp records.
record_personal_settings_sync
