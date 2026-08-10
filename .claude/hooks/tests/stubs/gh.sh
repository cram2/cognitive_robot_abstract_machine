#!/bin/bash
set -uo pipefail

# Test stub standing in for the `gh` CLI, so plan-updates-since.sh's tests can
# exercise its preferred backend for fetching tracking-issue comments without
# reaching GitHub. Copied into place as an executable named `gh`, earlier on
# PATH than any real one - see stub_bin fixture in
# test_plan_updates_since_sh.py.
#
# Recognizes only `gh api --paginate repos/<owner>/<repo>/issues/<n>/comments?...`,
# the one call plan-updates-since.sh makes through this backend:
#   STUB_GH_ISSUE_COMMENTS_JSON - the JSON body to print
#   STUB_GH_CALL_LOG            - file the invocation is appended to, so a
#                                 test can assert the exact call made
#
# Exits 64 on an invocation it doesn't recognize, rather than a plausible-
# looking success: a test must fail loudly if plan-updates-since.sh changes
# the call it makes.

if [ -n "${STUB_GH_CALL_LOG:-}" ]; then
  printf '%s\n' "$*" >> "${STUB_GH_CALL_LOG}"
fi

if [ "${1:-}" = "api" ] && [ "${2:-}" = "--paginate" ]; then
  case "${3:-}" in
    repos/*/issues/*/comments\?*)
      printf '%s' "${STUB_GH_ISSUE_COMMENTS_JSON:-[]}"
      exit 0
      ;;
  esac
fi

echo "stub gh: unexpected invocation: $*" >&2
exit 64
