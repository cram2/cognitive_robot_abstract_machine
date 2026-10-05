#!/bin/bash
set -uo pipefail

# Test stub standing in for the `gh` CLI, so every caller that shells out to it can be
# exercised without reaching GitHub. Copied into place as an executable named `gh`,
# earlier on PATH than any real one - see the stub_bin fixture in
# test_plan_updates_since_sh.py and the stubbed_gh fixture in test_upstream_reviews.py.
#
# One stub rather than one per suite: the two recognized invocations are disjoint, and a
# second copy is what drifts when the contract moves.
#
# `gh api graphql --input -`, the one call upstream_reviews' transport makes:
#   STUB_GH_GRAPHQL_JSON - the JSON body to print
#   STUB_GH_EXIT_CODE    - the exit code to return, defaulting to 0
#   STUB_GH_CALL_LOG     - file the request body is appended to, so a test can
#                          assert the exact query and variables sent
#
# `gh api --paginate repos/<owner>/<repo>/issues/<n>/comments?...`, the one call
# plan-updates-since.sh makes through this backend:
#   STUB_GH_ISSUE_COMMENTS_JSON - the JSON body to print
#   STUB_GH_CALL_LOG            - file the invocation is appended to, so a
#                                 test can assert the exact call made
#
# Exits 64 on an invocation it doesn't recognize, rather than a plausible-looking
# success: a test must fail loudly if a caller changes the call it makes.

if [ "${1:-}" = "api" ] && [ "${2:-}" = "graphql" ] && [ "${3:-}" = "--input" ]; then
  REQUEST_BODY="$(cat)"
  if [ -n "${STUB_GH_CALL_LOG:-}" ]; then
    printf '%s\n' "${REQUEST_BODY}" >> "${STUB_GH_CALL_LOG}"
  fi
  EXIT_CODE="${STUB_GH_EXIT_CODE:-0}"
  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "stub gh: simulated failure" >&2
    exit "${EXIT_CODE}"
  fi
  printf '%s' "${STUB_GH_GRAPHQL_JSON:-{\}}"
  exit 0
fi

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
