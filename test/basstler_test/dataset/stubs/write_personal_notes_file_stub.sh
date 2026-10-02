#!/bin/bash
set -euo pipefail

# Stand-in for write-personal-notes-file.sh used by test_refresh_dashboard_sh.py:
# records its own invocation arguments to a file instead of pushing anything to a
# real personal-notes branch.

printf '%s\n' "$@" > write_personal_notes_file_invocation.txt
