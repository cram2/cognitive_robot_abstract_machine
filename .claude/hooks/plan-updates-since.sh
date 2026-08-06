#!/bin/bash
set -euo pipefail

# Implements the recheck-deltas convention (cram-notes.md, personal-notes
# branch): rather than rereading a whole plan.yaml/roadmap.md to answer "what
# changed since I last looked", diff only from the last commit this clone
# saw. Also prints the plan's tracking-issue comments newer than that
# commit's timestamp, since a manifest diff alone misses discussion that
# never touched the manifest.
#
# Usage:
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/plan-updates-since.sh" <plan-id> [--since <sha>]
#
# <plan-id> is required - unlike save-plan.sh, there is no current-branch
# auto-discovery here, since this script is meant to be run from anywhere,
# including a branch that isn't itself a tracked item.
#
# --since <sha> is optional. Without it, the baseline is
# ./resolve-personal-notes-config.sh's PLAN_STATE_SYNC_STAMP - the
# notes-branch commit session-start.sh (or a previous run of this script)
# last recorded. Pass it explicitly to diff from an older or arbitrary point
# instead.
#
# The tracking-issue comment lookup needs no Claude Code session: like
# github-api.sh (see that script's own header comment), it prefers the `gh`
# CLI when installed, otherwise GH_TOKEN/GITHUB_TOKEN with curl. It is not
# sourced from github-api.sh directly - that file is not yet reachable from
# this repository's main branch (see this item's plan.yaml notes) - so the
# credential/request logic below is a small, deliberately temporary copy of
# that same pattern, not a new one.
#
# Every user-facing message string, and the comment-JSON parsing, live in
# plan_updates_since_support.py rather than inline here - so this script
# never carries its own copy of text the test suite also has to check
# against, and the comment-parsing logic is real, testable Python rather
# than an inline `python3 -c` snippet. See that module's own docstring.
#
# Finishes by advancing PLAN_STATE_SYNC_STAMP to the commit it just diffed
# up to, so the next recheck starts from here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

if ! command -v python3 > /dev/null 2>&1; then
  echo "python3 is required (it prints this script's messages and parses" >&2
  echo "GitHub's tracking-issue-comments response)." >&2
  exit 1
fi

SUPPORT_SCRIPT="${SCRIPT_DIR}/plan_updates_since_support.py"
GITHUB_API_BASE_URL="https://api.github.com"

# github_api_token: prints GH_TOKEN, else GITHUB_TOKEN, and fails with a
# message naming both routes (plus `gh`) if neither is set.
github_api_token() {
  local token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [ -z "${token}" ]; then
    echo "No GitHub credentials available. Either install the 'gh' CLI and run" >&2
    echo "'gh auth login', or set GH_TOKEN or GITHUB_TOKEN to a token with access" >&2
    echo "to the repository." >&2
    return 1
  fi
  printf '%s' "${token}"
}

# print_issue_comments_since: prints every comment on
# repository#issue_number created or updated at or after since_timestamp_utc
# (an "YYYY-MM-DDTHH:MM:SSZ" string), formatted by
# plan_updates_since_support.py's format_issue_comments.
print_issue_comments_since() {
  local repository="$1" issue_number="$2" since_timestamp_utc="$3"
  local path="repos/${repository}/issues/${issue_number}/comments?since=${since_timestamp_utc}&per_page=100"
  local response

  if command -v gh > /dev/null 2>&1; then
    response="$(gh api --paginate "${path}")"
  else
    local token
    token="$(github_api_token)" || return 1
    response="$(curl -sS \
      -H "Authorization: Bearer ${token}" \
      -H "Accept: application/vnd.github+json" \
      "${GITHUB_API_BASE_URL}/${path}")"
  fi

  printf '%s' "${response}" | python3 "${SUPPORT_SCRIPT}" print-comments
}

PLAN_ID=""
SINCE_SHA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --since)
      SINCE_SHA="$2"
      shift 2
      ;;
    -*)
      echo "Unrecognized argument: $1" >&2
      exit 1
      ;;
    *)
      if [ -n "${PLAN_ID}" ]; then
        echo "Unexpected extra argument: $1" >&2
        exit 1
      fi
      PLAN_ID="$1"
      shift
      ;;
  esac
done

if [ -z "${PLAN_ID}" ]; then
  echo "Usage: ${BASH_SOURCE[0]} <plan-id> [--since <sha>]" >&2
  exit 1
fi

if ! fetch_personal_notes_branch; then
  echo "Branch '${NOTES_BRANCH}' doesn't exist yet (tried: ${ATTEMPTED_NOTES_REMOTES})." >&2
  echo "Run ./create-personal-notes-branch.sh first, then re-run this script." >&2
  exit 1
fi

MANIFEST_PATH="$(plan_manifest_path "${PLAN_ID}")"
if ! git cat-file -e "FETCH_HEAD:${MANIFEST_PATH}" 2>/dev/null; then
  echo "No such plan '${PLAN_ID}' on '${NOTES_BRANCH}' (${MANIFEST_PATH} not found)." >&2
  exit 1
fi

if [ -z "${SINCE_SHA}" ]; then
  SINCE_SHA="$(last_recorded_plan_state_sha || true)"
fi
if [ -z "${SINCE_SHA}" ]; then
  echo "No baseline SHA known - pass --since <sha> explicitly, or run" >&2
  echo "session-start.sh once first so it can record one." >&2
  exit 1
fi

NEW_SHA="$(git rev-parse FETCH_HEAD)"
PLAN_DIRECTORY="$(plan_directory_path "${PLAN_ID}")"

echo "=== Changes to ${PLAN_DIRECTORY} (${SINCE_SHA}..${NEW_SHA}) ==="
DELTA="$(git diff "${SINCE_SHA}" "${NEW_SHA}" -- "${PLAN_DIRECTORY}")"
if [ -z "${DELTA}" ]; then
  python3 "${SUPPORT_SCRIPT}" print-no-changes-message
else
  printf '%s\n' "${DELTA}"
fi
echo

TRACKING_ISSUE="$(git show "FETCH_HEAD:${MANIFEST_PATH}" \
  | grep -oE '^tracking_issue:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+$' || true)"

if [ -z "${TRACKING_ISSUE}" ]; then
  echo "=== Tracking issue ==="
  python3 "${SUPPORT_SCRIPT}" print-no-tracking-issue-message
else
  DEFAULT_REPOSITORY="$(git show "FETCH_HEAD:${MANIFEST_PATH}" \
    | grep -oE '^default_repository:[[:space:]]*.+$' | head -1 \
    | sed -E 's/^default_repository:[[:space:]]*//' || true)"
  if [ -z "${DEFAULT_REPOSITORY}" ]; then
    python3 "${SUPPORT_SCRIPT}" print-no-default-repository-message \
      "${PLAN_ID}" "${TRACKING_ISSUE}"
    exit 1
  fi

  SINCE_TIMESTAMP_UTC="$(date -u -d "@$(git show -s --format=%ct "${SINCE_SHA}")" '+%Y-%m-%dT%H:%M:%SZ')"
  echo "=== Tracking issue #${TRACKING_ISSUE} comments since ${SINCE_TIMESTAMP_UTC} ==="
  print_issue_comments_since "${DEFAULT_REPOSITORY}" "${TRACKING_ISSUE}" "${SINCE_TIMESTAMP_UTC}"
fi

record_plan_state_sync_stamp
echo
echo "Recorded '${NEW_SHA}' as the new baseline for '${PLAN_ID}' (was ${SINCE_SHA})."
