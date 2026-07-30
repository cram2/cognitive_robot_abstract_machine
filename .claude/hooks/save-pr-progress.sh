#!/bin/bash
set -euo pipefail

# Persists edits made to CLAUDE.local.md's PR-progress section back onto the
# personal-notes branch, at a path keyed to the branch currently checked out
# (see pr_progress_path in ./resolve-personal-notes-config.sh) - so the
# plan/progress/next-steps for *this* PR survives session restarts, without
# ever being committed to the PR branch itself. This is the write half of the
# loop session-start.sh's own PR-progress header points Claude at when asked
# to update the plan: edit CLAUDE.local.md between the BEGIN-PR-PROGRESS /
# END-PR-PROGRESS markers, then run this script.
#
# Usage (from anywhere, after editing CLAUDE.local.md):
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/save-pr-progress.sh"
#
# Always reads the project root's CLAUDE.local.md specifically (see
# CLAUDE_LOCAL_MD in ./resolve-personal-notes-config.sh), regardless of the
# invoking cwd - not whatever CLAUDE.local.md a bare relative path might
# happen to resolve to from wherever this is run.
#
# Resolves the remote/branch exactly like session-start.sh and
# save-personal-notes.sh (git config > environment variable > the zero-config
# default, plus the same-branch-upstream fallback - see
# fetch_personal_notes_branch in ./resolve-personal-notes-config.sh). The path
# is never configured directly - it's always derived from the branch you
# currently have checked out, so it's impossible to accidentally save one
# PR's progress under another PR's key.
#
# Refuses to run on a branch with no sensible "current PR" (the default
# branch, a detached HEAD, or the personal-notes branch itself), or if
# CLAUDE.local.md has no PR-progress section to extract - run
# session-start.sh first in that case.
#
# Safe to re-run: a no-op if the extracted content already matches what's on
# the branch. Does its work in a scratch worktree, so it never touches your
# current branch or working tree - and never touches the PR branch itself, or
# anything that could be merged: the progress file lives only on the
# personal-notes branch. Fails with a clear message if that branch doesn't
# exist yet on any resolved remote - run ./create-personal-notes-branch.sh
# first in that case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

PROGRESS_PATH="$(pr_progress_path || true)"
if [ -z "${PROGRESS_PATH}" ]; then
  echo "Not on a PR branch (or on the personal-notes branch itself) - nothing to save." >&2
  echo "Switch to the branch whose progress you want to save, then re-run this script." >&2
  exit 1
fi

if [ ! -f "${CLAUDE_LOCAL_MD}" ]; then
  echo "No CLAUDE.local.md at the project root (${CLAUDE_LOCAL_MD}) - nothing to save." >&2
  exit 1
fi

if ! grep -q '^<!-- BEGIN-PR-PROGRESS -->$' "${CLAUDE_LOCAL_MD}"; then
  echo "CLAUDE.local.md has no PR-progress section to extract." >&2
  echo "Run session-start.sh first, then edit the section it writes." >&2
  exit 1
fi

if ! fetch_personal_notes_branch; then
  echo "Branch '${NOTES_BRANCH}' doesn't exist yet (tried: ${ATTEMPTED_NOTES_REMOTES})." >&2
  echo "Run ./create-personal-notes-branch.sh first, then re-run this script." >&2
  exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
CONTENT_FILE="$(mktemp)"
SCRATCH_DIR="$(mktemp -d)"
cleanup() {
  git worktree remove --force "${SCRATCH_DIR}" 2>/dev/null || rm -rf "${SCRATCH_DIR}"
  git branch -D __save-pr-progress-tmp > /dev/null 2>&1 || true
  rm -f "${CONTENT_FILE}"
}
trap cleanup EXIT

awk '/^<!-- BEGIN-PR-PROGRESS -->$/{flag=1; next} /^<!-- END-PR-PROGRESS -->$/{flag=0} flag' \
  "${CLAUDE_LOCAL_MD}" > "${CONTENT_FILE}"

git branch -D __save-pr-progress-tmp > /dev/null 2>&1 || true
# FETCH_HEAD: see the note on FETCH_HEAD vs. "<remote>/<branch>" refs in
# save-personal-notes.sh - a URL-form remote has no tracking ref, and
# FETCH_HEAD is shared across worktrees so this scratch worktree sees the
# fetch above correctly regardless.
git worktree add -b __save-pr-progress-tmp "${SCRATCH_DIR}" FETCH_HEAD --quiet

mkdir -p "${SCRATCH_DIR}/$(dirname "${PROGRESS_PATH}")"
cp "${CONTENT_FILE}" "${SCRATCH_DIR}/${PROGRESS_PATH}"
git -C "${SCRATCH_DIR}" add "${PROGRESS_PATH}"

if git -C "${SCRATCH_DIR}" diff --cached --quiet -- "${PROGRESS_PATH}"; then
  echo "No changes to save - '${PROGRESS_PATH}' on '${NOTES_BRANCH}' (remote '${ACTIVE_NOTES_REMOTE}') is already up to date."
  exit 0
fi

git -C "${SCRATCH_DIR}" commit --quiet -m "Update PR progress for ${CURRENT_BRANCH}"
git -C "${SCRATCH_DIR}" push "${ACTIVE_NOTES_REMOTE}" "HEAD:${NOTES_BRANCH}"

echo "Saved '${PROGRESS_PATH}' back to '${NOTES_BRANCH}' on '${ACTIVE_NOTES_REMOTE}'."
