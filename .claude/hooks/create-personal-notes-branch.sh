#!/bin/bash
set -euo pipefail

# One-time helper: creates the personal-notes branch (default:
# `claude/personal-notes`) on a remote (default: `origin`) with a single
# empty notes file at the default path (`.claude/personal/cram-notes.md`),
# and pushes it - so session-start.sh has something to read out of the box,
# with no config needed on your end.
#
# Usage (from anywhere - always operates on this repo specifically, see
# resolve-personal-notes-config.sh):
#   ./.claude/hooks/create-personal-notes-branch.sh
#
# Resolves the remote/branch/path the same way session-start.sh does (git
# config > environment variable > default), so it always targets whatever a
# fresh session would actually read from. Override via the same variable
# names it and configure-personal-notes.sh read:
#   CLAUDE_PERSONAL_NOTES_REMOTE=<remote-name-or-url> \
#     CLAUDE_PERSONAL_NOTES_BRANCH=<branch> CLAUDE_PERSONAL_NOTES_PATH=<path> \
#     ./.claude/hooks/create-personal-notes-branch.sh
# See ./README.md for when a remote name vs. a raw URL is the right choice.
#
# Safe to re-run: refuses to touch a branch that already exists locally, on
# the resolved remote, or on the current branch's own upstream remote (see
# current_branch_upstream_remote in ./resolve-personal-notes-config.sh) - so
# it never overwrites existing notes, and never creates a second, divergent
# copy of a branch that fetch_personal_notes_branch's fallback would already
# have found on read. It always creates on the resolved remote specifically,
# though, never on the fallback one - creation is a deliberate act, so it
# targets exactly what you configured (or the zero-config default), with no
# surprises. Does its work in a scratch worktree, so it never touches your
# current branch or working tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

if git show-ref --verify --quiet "refs/heads/${NOTES_BRANCH}"; then
  echo "Branch '${NOTES_BRANCH}' already exists locally - not touching it." >&2
  exit 1
fi

if git ls-remote --exit-code --heads "${NOTES_REMOTE}" "${NOTES_BRANCH}" > /dev/null 2>&1; then
  echo "Branch '${NOTES_BRANCH}' already exists on '${NOTES_REMOTE}' - not touching it." >&2
  exit 1
fi

UPSTREAM_REMOTE="$(current_branch_upstream_remote)"
if [ -n "${UPSTREAM_REMOTE}" ] && [ "${UPSTREAM_REMOTE}" != "${NOTES_REMOTE}" ] \
    && git ls-remote --exit-code --heads "${UPSTREAM_REMOTE}" "${NOTES_BRANCH}" > /dev/null 2>&1; then
  echo "Branch '${NOTES_BRANCH}' already exists on '${UPSTREAM_REMOTE}' (this branch's upstream" >&2
  echo "remote) - not creating a second copy on '${NOTES_REMOTE}'. session-start.sh and" >&2
  echo "save-personal-notes.sh already fall back to '${UPSTREAM_REMOTE}' automatically." >&2
  exit 1
fi

SCRATCH_DIR="$(mktemp -d)"
trap 'git worktree remove --force "${SCRATCH_DIR}" 2>/dev/null || rm -rf "${SCRATCH_DIR}"' EXIT

git worktree add --orphan -b "${NOTES_BRANCH}" "${SCRATCH_DIR}" --quiet
git -C "${SCRATCH_DIR}" rm -rf --quiet . > /dev/null 2>&1 || true

mkdir -p "${SCRATCH_DIR}/$(dirname "${NOTES_PATH}")"
: > "${SCRATCH_DIR}/${NOTES_PATH}"

git -C "${SCRATCH_DIR}" add "${NOTES_PATH}"
git -C "${SCRATCH_DIR}" commit --quiet -m "Initialize personal notes branch"
git -C "${SCRATCH_DIR}" push "${NOTES_REMOTE}" "${NOTES_BRANCH}"

git worktree remove --force "${SCRATCH_DIR}"
git branch -D "${NOTES_BRANCH}" > /dev/null
trap - EXIT

echo "Created and pushed '${NOTES_BRANCH}' with an empty '${NOTES_PATH}'."
echo "Your current branch and working tree are untouched."
