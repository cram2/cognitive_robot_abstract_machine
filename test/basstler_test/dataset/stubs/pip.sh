#!/bin/bash
set -uo pipefail

# Test stub standing in for `pip`, so session-start.sh's tests can exercise
# the install it runs without installing anything into the machine running
# them. Copied into place as an executable named `pip`, earlier on PATH than
# the real one; see the stub_bin fixture in conftest.py.
#
# Recognizes only `install <specifier>...`, the one invocation
# install_dependencies makes:
#   STUB_PIP_CALL_LOG - file the invocation is appended to
#   STUB_PIP_STATUS   - exit status to report, default 0
#
# Exits 64 on an unrecognized invocation for the same reason gh.sh does: a
# changed call must fail a test rather than pass by accident.

if [ -n "${STUB_PIP_CALL_LOG:-}" ]; then
  printf '%s\n' "$*" >> "${STUB_PIP_CALL_LOG}"
fi

if [ "${1:-}" != "install" ] || [ -z "${2:-}" ]; then
  echo "pip stub: unrecognized invocation: $*" >&2
  exit 64
fi

shift
STATUS="${STUB_PIP_STATUS:-0}"
if [ "${STATUS}" = "0" ]; then
  echo "Successfully installed $*"
  exit 0
fi

echo "ERROR: could not install $*" >&2
exit "${STATUS}"
