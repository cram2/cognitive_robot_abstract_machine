#!/bin/bash
set -euo pipefail

# Records your git identity on the personal-notes branch, so every clone
# session-start.sh runs in authors its commits as you - see the "Git identity"
# section of ./session-start.sh's header for what reads it back.
#
# Usage:
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/save-git-identity.sh" \
#     --name "Your Name" \
#     --email "you@example.com"
#
# Both arguments are required, and neither is guessed from the clone's current
# git config. That config is exactly what cannot be trusted here: in a fresh
# session environment it resolves to the agent's own global identity, so a
# script that read it would record the identity this whole mechanism exists to
# stop, and do it silently. Refusing to guess is the reason this is worth a
# script at all.
#
# Writes the file with `git config --file`, the same tool that reads it back,
# so the two can never disagree about the format.
#
# Safe to re-run: delegates the commit and push to
# ./write-personal-notes-file.sh, which is a no-op when the recorded identity
# already matches. Fails with a clear message if the notes branch doesn't exist
# yet - run ./create-personal-notes-branch.sh first in that case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

IDENTITY_NAME=""
IDENTITY_EMAIL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name)
      IDENTITY_NAME="$2"
      shift 2
      ;;
    --email)
      IDENTITY_EMAIL="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${IDENTITY_NAME}" ] || [ -z "${IDENTITY_EMAIL}" ]; then
  echo "Usage: ${BASH_SOURCE[0]} --name \"Your Name\" --email \"you@example.com\"" >&2
  echo "Both are required - neither is guessed from this clone's git config," >&2
  echo "which in a session environment is the agent's identity, not yours." >&2
  exit 1
fi

IDENTITY_FILE="$(mktemp)"
trap 'rm -f "${IDENTITY_FILE}"' EXIT
git config --file "${IDENTITY_FILE}" user.name "${IDENTITY_NAME}"
git config --file "${IDENTITY_FILE}" user.email "${IDENTITY_EMAIL}"

bash "${PROJECT_ROOT}/${WRITE_PERSONAL_NOTES_FILE_SCRIPT}" \
  --source "${IDENTITY_FILE}" \
  --destination "${PERSONAL_GIT_IDENTITY_PATH}" \
  --message "Record git identity for $(format_git_identity "${IDENTITY_NAME}" "${IDENTITY_EMAIL}")"
