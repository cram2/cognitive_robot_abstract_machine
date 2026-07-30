#!/bin/bash
set -euo pipefail

# Persists edits made to CLAUDE.local.md back onto the personal-notes branch
# on its remote (default: `origin`), so the next session's session-start.sh
# reads the updated notes. This is the write half of the loop
# session-start.sh's own header (see ./session-start.sh) points Claude at when
# asked to edit your notes: edit CLAUDE.local.md below its header, then run
# this script.
#
# Usage (from anywhere, after editing CLAUDE.local.md):
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
#
# Always reads/writes the project root's CLAUDE.local.md specifically (see
# CLAUDE_LOCAL_MD in ./resolve-personal-notes-config.sh), regardless of the
# invoking cwd - not whatever CLAUDE.local.md a bare relative path might
# happen to resolve to from wherever this is run.
#
# Resolves the remote/branch/path exactly like session-start.sh (git config >
# environment variable > the zero-config default, plus the same
# same-branch-upstream fallback - see fetch_personal_notes_branch in
# ./resolve-personal-notes-config.sh), so it always writes back to wherever
# the notes were actually read from, even when that was the fallback remote
# rather than the resolved one. The remote may be a configured remote's name
# or a raw git URL - see ./README.md.
#
# Extracts only the content between the BEGIN-PERSONAL-NOTES/END-PERSONAL-NOTES
# markers before pushing - never the header above them, and never the separate
# PR-progress section session-start.sh may have appended below them (see
# ./save-pr-progress.sh for that one) - so only your actual notes content ever
# ends up committed to the notes branch.
#
# Safe to re-run: a no-op if CLAUDE.local.md's content (after stripping the
# header) already matches what's on the branch. Does its work in a scratch
# worktree, so it never touches your current branch or working tree. Fails
# with a clear message if the target branch doesn't exist yet on any resolved
# remote - run ./create-personal-notes-branch.sh first in that case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

if [ ! -f "${CLAUDE_LOCAL_MD}" ]; then
  echo "No CLAUDE.local.md at the project root (${CLAUDE_LOCAL_MD}) - nothing to save." >&2
  exit 1
fi

if ! fetch_personal_notes_branch; then
  echo "Branch '${NOTES_BRANCH}' doesn't exist yet (tried: ${ATTEMPTED_NOTES_REMOTES})." >&2
  echo "Run ./create-personal-notes-branch.sh first, then re-run this script." >&2
  exit 1
fi

CONTENT_FILE="$(mktemp)"
SCRATCH_DIR="$(mktemp -d)"
cleanup() {
  git worktree remove --force "${SCRATCH_DIR}" 2>/dev/null || rm -rf "${SCRATCH_DIR}"
  git branch -D __save-personal-notes-tmp > /dev/null 2>&1 || true
  rm -f "${CONTENT_FILE}"
}
trap cleanup EXIT

if grep -q '^<!-- BEGIN-PERSONAL-NOTES -->$' "${CLAUDE_LOCAL_MD}"; then
  awk '/^<!-- BEGIN-PERSONAL-NOTES -->$/{flag=1; next} /^<!-- END-PERSONAL-NOTES -->$/{flag=0} flag' \
    "${CLAUDE_LOCAL_MD}" > "${CONTENT_FILE}"
else
  cp "${CLAUDE_LOCAL_MD}" "${CONTENT_FILE}"
fi

git branch -D __save-personal-notes-tmp > /dev/null 2>&1 || true
# FETCH_HEAD, not "${ACTIVE_NOTES_REMOTE}/${NOTES_BRANCH}": a URL-form remote
# creates no remote-tracking ref, but FETCH_HEAD always points at what was
# just fetched, whether the serving remote was a name or a raw URL. It's
# shared across worktrees (unlike HEAD/index), so this scratch worktree sees
# the fetch above correctly.
git worktree add -b __save-personal-notes-tmp "${SCRATCH_DIR}" FETCH_HEAD --quiet

mkdir -p "${SCRATCH_DIR}/$(dirname "${NOTES_PATH}")"
cp "${CONTENT_FILE}" "${SCRATCH_DIR}/${NOTES_PATH}"
git -C "${SCRATCH_DIR}" add "${NOTES_PATH}"

if git -C "${SCRATCH_DIR}" diff --cached --quiet -- "${NOTES_PATH}"; then
  echo "No changes to save - '${NOTES_PATH}' on '${NOTES_BRANCH}' (remote '${ACTIVE_NOTES_REMOTE}') is already up to date."
  exit 0
fi

git -C "${SCRATCH_DIR}" commit --quiet -m "Update personal notes"
# Push back to ACTIVE_NOTES_REMOTE (wherever the branch was actually read
# from above), not unconditionally to NOTES_REMOTE, so a save always lands on
# the same remote the notes came from - see fetch_personal_notes_branch.
git -C "${SCRATCH_DIR}" push "${ACTIVE_NOTES_REMOTE}" "HEAD:${NOTES_BRANCH}"

echo "Saved '${NOTES_PATH}' back to '${NOTES_BRANCH}' on '${ACTIVE_NOTES_REMOTE}'."
