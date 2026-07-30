#!/bin/bash
set -euo pipefail

# Generic "commit and push one file to the personal-notes branch" primitive.
# Every write in this system that isn't already served by a purpose-built
# script (save-personal-notes.sh for CLAUDE.local.md, save-plan.sh for a
# plan's manifest/roadmap/branch-index trio) is exactly this shape: copy one
# already-prepared local file to one destination path, commit, push, done.
# Extracted so callers with that shape - currently
# .claude/skills/plan-dashboard/refresh_dashboard.sh's manifest-correction
# push, and /plan-dashboard's own dashboard-URL-cache update - don't each
# re-embed the worktree dance as a bash snippet of their own.
#
# Usage:
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/write-personal-notes-file.sh" \
#     --source <local-file> \
#     --destination <repo-relative-path> \
#     --message <commit-message>
#
# Resolves the remote/branch exactly like every other hook script here (git
# config > environment variable > the zero-config default, plus the
# same-branch-upstream fallback - see fetch_personal_notes_branch in
# ./resolve-personal-notes-config.sh).
#
# Safe to re-run: a no-op (exit 0, nothing pushed) if --destination's content
# on the branch already matches --source. Does its work in a scratch
# worktree, so it never touches the caller's current branch or working tree.
# Fails with a clear message if the target branch doesn't exist yet on any
# resolved remote - run ./create-personal-notes-branch.sh first in that case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

SOURCE_FILE=""
DESTINATION_PATH=""
COMMIT_MESSAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --source)
      SOURCE_FILE="$2"
      shift 2
      ;;
    --destination)
      DESTINATION_PATH="$2"
      shift 2
      ;;
    --message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${SOURCE_FILE}" ] || [ -z "${DESTINATION_PATH}" ] || [ -z "${COMMIT_MESSAGE}" ]; then
  echo "Usage: ${BASH_SOURCE[0]} --source <local-file> --destination <repo-relative-path> --message <commit-message>" >&2
  exit 1
fi
if [ ! -f "${SOURCE_FILE}" ]; then
  echo "--source file not found: ${SOURCE_FILE}" >&2
  exit 1
fi
case "${DESTINATION_PATH}" in
  /*|*/../*|../*|*/..|..)
    echo "--destination must be a relative path with no '..' component and no" >&2
    echo "leading '/': ${DESTINATION_PATH}" >&2
    exit 1
    ;;
esac

if ! fetch_personal_notes_branch; then
  echo "Branch '${NOTES_BRANCH}' doesn't exist yet (tried: ${ATTEMPTED_NOTES_REMOTES})." >&2
  echo "Run ./create-personal-notes-branch.sh first, then re-run this script." >&2
  exit 1
fi

SCRATCH_DIR="$(mktemp -d)"
# Suffixed with $$ (this process's PID) so two concurrent invocations never
# race over the same worktree branch name.
SCRATCH_BRANCH="__write-personal-notes-file-tmp-$$"
cleanup() {
  git worktree remove --force "${SCRATCH_DIR}" 2>/dev/null || rm -rf "${SCRATCH_DIR}"
  git branch -D "${SCRATCH_BRANCH}" > /dev/null 2>&1 || true
}
trap cleanup EXIT

git branch -D "${SCRATCH_BRANCH}" > /dev/null 2>&1 || true
# FETCH_HEAD, not "${ACTIVE_NOTES_REMOTE}/${NOTES_BRANCH}": a URL-form remote
# creates no remote-tracking ref, but FETCH_HEAD always points at what was
# just fetched, whether the serving remote was a name or a raw URL.
git worktree add -b "${SCRATCH_BRANCH}" "${SCRATCH_DIR}" FETCH_HEAD --quiet

mkdir -p "${SCRATCH_DIR}/$(dirname "${DESTINATION_PATH}")"
cp "${SOURCE_FILE}" "${SCRATCH_DIR}/${DESTINATION_PATH}"
git -C "${SCRATCH_DIR}" add "${DESTINATION_PATH}"

if git -C "${SCRATCH_DIR}" diff --cached --quiet; then
  echo "No changes to '${DESTINATION_PATH}' - already up to date on '${NOTES_BRANCH}' (remote '${ACTIVE_NOTES_REMOTE}')."
  exit 0
fi

git -C "${SCRATCH_DIR}" commit --quiet -m "${COMMIT_MESSAGE}"
git -C "${SCRATCH_DIR}" push "${ACTIVE_NOTES_REMOTE}" "HEAD:${NOTES_BRANCH}"

echo "Wrote '${DESTINATION_PATH}' back to '${NOTES_BRANCH}' on '${ACTIVE_NOTES_REMOTE}'."
