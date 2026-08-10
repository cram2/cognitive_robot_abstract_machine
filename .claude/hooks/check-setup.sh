#!/bin/bash
set -euo pipefail

# Reports whether this clone has everything the .claude/ agent tooling needs -
# personal notes, PR progress, plan dashboards - so a session (or a person) can
# tell in one call what is already set up and what still needs doing.
#
# Usage (from anywhere - always inspects this repo specifically, see
# resolve-personal-notes-config.sh):
#   ./.claude/hooks/check-setup.sh
#
# Read-only: it fetches (into FETCH_HEAD) and reads git config, but never
# writes config, branches, files, or remotes. Running it can't change the
# answer it gives.
#
# Output is one tab-separated "<check>\t<status>\t<detail>" row per check, in
# the order the checks have to be fixed in (a later one can depend on an
# earlier one being satisfied). Three statuses:
#   ok           - nothing to do
#   needs-setup  - something is missing; <detail> says what
#   info         - context for whoever is reading, never a pass/fail
# Exit code is 0 when no row is needs-setup, 1 otherwise - so a caller can
# take the fast path ("everything is already fine") on the exit code alone,
# without parsing anything.
#
# TSV rather than JSON for the same reason the plan branch index is TSV (see
# plan_id_for_branch in ./resolve-personal-notes-config.sh): a tab can't occur
# inside any value here, and it needs nothing beyond the shell itself - this
# script must not gain a dependency just to describe whether dependencies are
# installed.
#
# The one thing it deliberately does not check is GitHub API access, which
# /plan-dashboard needs for live pull request state: that is reachable only
# through a session's MCP tools, not from a shell. The same goes for the pull
# request labels the dashboard reads. .claude/skills/setup-personal-notes
# covers both as their own steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

EXIT_CODE=0

# report: prints one TSV row, and remembers that the overall run failed if the
# status is needs-setup, so the exit code never has to be tracked by hand at
# each call site.
report() {
  local check="$1" status="$2" detail="$3"
  [ "${status}" != "needs-setup" ] || EXIT_CODE=1
  printf '%s\t%s\t%s\n' "${check}" "${status}" "${detail}"
}

# resolution_source: prints where a resolved setting actually came from, given
# the git config key and environment variable name that feed it, mirroring the
# precedence resolve-personal-notes-config.sh applies (git config >
# environment variable > built-in default).
resolution_source() {
  local config_key="$1" environment_variable="$2"
  if [ -n "$(git config --get "${config_key}" || true)" ]; then
    printf 'git config %s\n' "${config_key}"
  elif [ -n "${!environment_variable:-}" ]; then
    printf 'environment variable %s\n' "${environment_variable}"
  else
    printf 'built-in default\n'
  fi
}

# %% the tooling itself

# Everything below assumes this checkout actually carries the agent tooling.
# A fork that predates it would otherwise fail later, one confusing missing
# file at a time, instead of here with a single clear answer.
MISSING_TOOLING=""
for tooling_path in \
    "${BUILD_DASHBOARD_SCRIPT}" \
    "${REFRESH_DASHBOARD_SCRIPT}" \
    "${PLAN_DASHBOARD_REQUIREMENTS_FILE}" \
    "${PLAN_SCHEMA_DOCUMENT}"; do
  [ -f "${tooling_path}" ] || MISSING_TOOLING="${MISSING_TOOLING} ${tooling_path}"
done
if [ -n "${MISSING_TOOLING}" ]; then
  report tooling_files needs-setup \
    "this checkout is missing:${MISSING_TOOLING} - merge the plan-dashboard tooling into your fork's default branch first"
else
  report tooling_files ok "plan-dashboard scripts, schema reference and requirements are all present"
fi

# %% session-start wiring

if grep -q 'session-start.sh' "${PROJECT_ROOT}/.claude/settings.json" 2>/dev/null; then
  report session_start_hook ok "registered in .claude/settings.json"
else
  report session_start_hook needs-setup \
    ".claude/settings.json does not register .claude/hooks/session-start.sh - CLAUDE.local.md will never be populated"
fi

if git check-ignore --quiet CLAUDE.local.md 2>/dev/null; then
  report claude_local_md_ignored ok "CLAUDE.local.md is gitignored, so notes can never be committed"
else
  report claude_local_md_ignored needs-setup \
    "CLAUDE.local.md is not gitignored here - personal notes could end up in a commit"
fi

# %% where the notes live

report notes_remote info "${NOTES_REMOTE} (from $(resolution_source claude.personalNotesRemote CLAUDE_PERSONAL_NOTES_REMOTE))"
# The URL matters to a caller that can tell whose repository it is (a session
# with GitHub access can, this script can't): a remote name says nothing about
# whether it points at the reader's own fork or at a shared upstream they
# cannot push notes to.
report notes_remote_url info "$(git remote get-url "${NOTES_REMOTE}" 2>/dev/null || printf '%s\n' "${NOTES_REMOTE}")"
report notes_branch_name info "${NOTES_BRANCH} (from $(resolution_source claude.personalNotesBranch CLAUDE_PERSONAL_NOTES_BRANCH))"
report notes_path info "${NOTES_PATH} (from $(resolution_source claude.personalNotesPath CLAUDE_PERSONAL_NOTES_PATH))"

# %% the personal-notes branch and its contents

if fetch_personal_notes_branch; then
  report notes_branch ok "'${NOTES_BRANCH}' found on '${ACTIVE_NOTES_REMOTE}'"

  if git cat-file -e "FETCH_HEAD:${NOTES_PATH}" 2>/dev/null; then
    report notes_file ok "'${NOTES_PATH}' exists on '${NOTES_BRANCH}'"
  else
    report notes_file needs-setup \
      "'${NOTES_BRANCH}' exists but has no '${NOTES_PATH}' - session-start.sh will write no notes"
  fi
else
  report notes_branch needs-setup \
    "no '${NOTES_BRANCH}' branch on any of: ${ATTEMPTED_NOTES_REMOTES} - run ./create-personal-notes-branch.sh (after pointing the remote at your own fork if it isn't already)"
  report notes_file needs-setup "not checked - the branch that would hold it doesn't exist yet"
fi

# %% plan-dashboard dependencies

# Derived from requirements.txt itself rather than a second hand-written list
# of import names, which would silently go stale the moment that file changes.
# Distribution names are what requirements.txt states, so they're what gets
# looked up - no pyyaml/yaml-style mapping to maintain anywhere.
if ! command -v python3 > /dev/null 2>&1; then
  report dashboard_dependencies needs-setup "python3 is not on PATH, so the plan-dashboard scripts cannot run at all"
elif [ ! -f "${PLAN_DASHBOARD_REQUIREMENTS_FILE}" ]; then
  report dashboard_dependencies needs-setup "cannot check: ${PLAN_DASHBOARD_REQUIREMENTS_FILE} is missing"
else
  MISSING_DEPENDENCIES="$(python3 - "${PLAN_DASHBOARD_REQUIREMENTS_FILE}" <<'PYTHON'
import re
import sys
from importlib.metadata import PackageNotFoundError, distribution

missing = []
for line in open(sys.argv[1], encoding="utf-8"):
    requirement = line.split("#", 1)[0].strip()
    if not requirement:
        continue
    name = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]
    try:
        distribution(name)
    except PackageNotFoundError:
        missing.append(name)
print(" ".join(missing))
PYTHON
)"
  if [ -z "${MISSING_DEPENDENCIES}" ]; then
    report dashboard_dependencies ok "every requirement in ${PLAN_DASHBOARD_REQUIREMENTS_FILE} is installed"
  else
    report dashboard_dependencies needs-setup \
      "not installed:${MISSING_DEPENDENCIES// / } - run: pip install -r ${PLAN_DASHBOARD_REQUIREMENTS_FILE}"
  fi
fi

# %% the result of it all working

# Last, because it is the outcome of everything above rather than a separate
# thing to configure: if the notes branch resolves and session-start.sh has
# run, this file exists and the session is already reading it.
if [ -f "${CLAUDE_LOCAL_MD}" ]; then
  report claude_local_md ok "populated at ${CLAUDE_LOCAL_MD}"
else
  report claude_local_md needs-setup \
    "not written yet - run ./session-start.sh once the checks above pass, or start a fresh session"
fi

exit "${EXIT_CODE}"
