#!/bin/bash
set -euo pipefail

# Orchestrates one plan's refresh: sync_manifest_status.py's auto-correction,
# pushing that correction back to the personal-notes branch if it changed
# anything, then build_dashboard.py's render - the exact sequence
# .claude/skills/plan-dashboard/SKILL.md step 2 previously spelled out as an
# embedded bash snippet for a session to improvise from. Extracted so that
# sequence is real, tested-by-construction code (it just calls the two
# scripts .claude/skills/plan-dashboard/tests/ already covers) rather than
# prose a session re-derives - and re-risks getting subtly wrong - on every
# run.
#
# Usage:
#   refresh_dashboard.sh \
#     --plan-id <plan-id> \
#     --plan <plan.yaml path> \
#     --roadmap <roadmap.md path> \
#     --pr-data <pr_data.json path> \
#     --output <dashboard.html output path> \
#     [--tracking-url <url>]
#
# Prints one JSON summary to stdout, merging sync_manifest_status.py's own
# {"corrected": [...]} with build_dashboard.py's status/drift/ready summary,
# so the calling skill has everything step 4's report needs from one place.
#
# Requires PyYAML, Jinja2, and the markdown package - see requirements.txt
# next to this script.

# Locating resolve-personal-notes-config.sh is the one irreducible hardcoded
# path here: it's the file that defines every other shared path constant
# used below (BUILD_DASHBOARD_SCRIPT, WRITE_PERSONAL_NOTES_FILE_SCRIPT, ...),
# so it can't itself be referenced through one of them. Sourcing it also
# `cd`s to the project root, so every one of its path constants below can be
# used exactly as written - no further path arithmetic needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../hooks/resolve-personal-notes-config.sh"

PLAN_ID=""
PLAN_FILE=""
ROADMAP_FILE=""
PULL_REQUEST_DATA_FILE=""
OUTPUT_FILE=""
TRACKING_URL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --plan-id)
      PLAN_ID="$2"
      shift 2
      ;;
    --plan)
      PLAN_FILE="$2"
      shift 2
      ;;
    --roadmap)
      ROADMAP_FILE="$2"
      shift 2
      ;;
    --pr-data)
      PULL_REQUEST_DATA_FILE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --tracking-url)
      TRACKING_URL="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${PLAN_ID}" ] || [ -z "${PLAN_FILE}" ] || [ -z "${ROADMAP_FILE}" ] \
    || [ -z "${PULL_REQUEST_DATA_FILE}" ] || [ -z "${OUTPUT_FILE}" ]; then
  echo "Usage: ${BASH_SOURCE[0]} --plan-id <id> --plan <plan.yaml> --roadmap <roadmap.md> --pr-data <pr_data.json> --output <dashboard.html> [--tracking-url <url>]" >&2
  exit 1
fi

SYNC_SUMMARY="$(python3 "${SYNC_MANIFEST_STATUS_SCRIPT}" \
  --plan "${PLAN_FILE}" \
  --pr-data "${PULL_REQUEST_DATA_FILE}")"

CORRECTED_COUNT="$(python3 "${REFRESH_DASHBOARD_SUPPORT_SCRIPT}" count-corrected "${SYNC_SUMMARY}")"

if [ "${CORRECTED_COUNT}" != "0" ]; then
  DESTINATION_PATH="$(plan_manifest_path "${PLAN_ID}")"
  "${WRITE_PERSONAL_NOTES_FILE_SCRIPT}" \
    --source "${PLAN_FILE}" \
    --destination "${DESTINATION_PATH}" \
    --message "Auto-sync ${PLAN_ID}: ${CORRECTED_COUNT} item(s) to done (merged on GitHub)"
fi

BUILD_ARGUMENTS=(
  --plan "${PLAN_FILE}"
  --roadmap "${ROADMAP_FILE}"
  --pr-data "${PULL_REQUEST_DATA_FILE}"
  --output "${OUTPUT_FILE}"
)
if [ -n "${TRACKING_URL}" ]; then
  BUILD_ARGUMENTS+=(--tracking-url "${TRACKING_URL}")
fi
BUILD_SUMMARY="$(python3 "${BUILD_DASHBOARD_SCRIPT}" "${BUILD_ARGUMENTS[@]}")"

python3 "${REFRESH_DASHBOARD_SUPPORT_SCRIPT}" merge-summaries "${SYNC_SUMMARY}" "${BUILD_SUMMARY}"
