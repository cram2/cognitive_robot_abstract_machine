#!/bin/bash
set -uo pipefail

# Test stub standing in for `curl`, so plan-updates-since.sh's tests can
# exercise its token fallback - the path taken when `gh` isn't installed -
# without reaching GitHub. Copied into place as an executable named `curl`,
# earlier on PATH than the real one; see stub_bin fixture in
# test_plan_updates_since_sh.py.
#
# Recognizes only a GET against .../issues/<n>/comments, the one request
# shape plan-updates-since.sh makes through this backend:
#   STUB_CURL_ISSUE_COMMENTS_JSON - the JSON body to print
#   STUB_CURL_CALL_LOG            - file the invocation is appended to
#
# Exits 64 on an unrecognized invocation for the same reason gh.sh does: a
# changed call must fail a test rather than pass by accident.

if [ -n "${STUB_CURL_CALL_LOG:-}" ]; then
  printf '%s\n' "$*" >> "${STUB_CURL_CALL_LOG}"
fi

REQUEST_URL=""
for argument in "$@"; do
  case "${argument}" in
    https://*) REQUEST_URL="${argument}" ;;
  esac
done

case "${REQUEST_URL}" in
  */issues/*/comments\?*)
    printf '%s' "${STUB_CURL_ISSUE_COMMENTS_JSON:-[]}"
    exit 0
    ;;
esac

echo "stub curl: unexpected URL: ${REQUEST_URL}" >&2
exit 64
