# Sourced (not executed) by session-start.sh, create-personal-notes-branch.sh,
# save-personal-notes.sh and save-pr-progress.sh, so all four resolve the
# personal-notes remote, branch and path with the exact same precedence: git
# config > environment variable > the zero-config default. See ./README.md.

# CLAUDE_LOCAL_MD: the one, deterministic path to CLAUDE.local.md, always the
# project root regardless of the caller's current working directory. Derived
# from this file's own location on disk (${BASH_SOURCE[0]}, which - inside a
# sourced file - is that file's own path, not the sourcing script's) rather
# than $CLAUDE_PROJECT_DIR or the caller's cwd: a SessionStart hook's cwd
# isn't guaranteed to be the project root (see session-start.sh), and these
# scripts are also run directly, outside any hook, where nothing guarantees
# $CLAUDE_PROJECT_DIR is set at all. This file always lives at
# <project-root>/.claude/hooks/resolve-personal-notes-config.sh, so two
# levels up from its own directory is always the project root, unconditionally.
HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HOOKS_DIR}/../.." && pwd)"
CLAUDE_LOCAL_MD="${PROJECT_ROOT}/CLAUDE.local.md"

# Every caller also does `git` operations (config, fetch, worktree, branch)
# that assume they're running inside this repo. Move there explicitly instead
# of trusting the invoking cwd to already be inside it (or inside it at all) -
# git itself would otherwise auto-discover a *different* repo if run from
# inside some other one, or fail outright if run from outside any repo.
cd "${PROJECT_ROOT}"

NOTES_REMOTE="$(git config --get claude.personalNotesRemote || true)"
NOTES_REMOTE="${NOTES_REMOTE:-${CLAUDE_PERSONAL_NOTES_REMOTE:-origin}}"

NOTES_BRANCH="$(git config --get claude.personalNotesBranch || true)"
NOTES_BRANCH="${NOTES_BRANCH:-${CLAUDE_PERSONAL_NOTES_BRANCH:-claude/personal-notes}}"

NOTES_PATH="$(git config --get claude.personalNotesPath || true)"
NOTES_PATH="${NOTES_PATH:-${CLAUDE_PERSONAL_NOTES_PATH:-.claude/personal/cram-notes.md}}"

# NOTES_REMOTE may be either a configured remote's name (e.g. "origin") or a
# raw git URL (e.g. "https://github.com/<you>/<repo>") - `git fetch`/`git
# push` accept both interchangeably, and a URL needs no `git remote add`
# first. Use a URL whenever your own fork isn't the clone's "origin" (for
# example, some session environments name the upstream repo "origin" and your
# fork something else) - the URL form works without depending on that
# session-specific remote name/alias existing at all.

# PERSONAL_SETTINGS_PATH / LOCAL_SETTINGS_RELATIVE_PATH / LOCAL_SETTINGS_JSON /
# PERSONAL_SETTINGS_SYNC_STAMP: the personal Claude Code settings round trip -
# `.claude/personal/settings.local.json` on the personal-notes branch, synced into
# this clone's `.claude/settings.local.json` (the file Claude Code itself reads as
# local settings, and which is gitignored). Fixed convention, never overridden: the
# destination is dictated by Claude Code, and the source is per-contributor already
# by virtue of living on that contributor's own notes branch - same reasoning as
# PLANS_DIR and the pr-progress directory below.
PERSONAL_SETTINGS_PATH=".claude/personal/settings.local.json"
LOCAL_SETTINGS_RELATIVE_PATH=".claude/settings.local.json"
LOCAL_SETTINGS_JSON="${PROJECT_ROOT}/${LOCAL_SETTINGS_RELATIVE_PATH}"
# The stamp records the hash of the settings content last synced into - or saved
# out of - LOCAL_SETTINGS_JSON, which is what makes "has this been edited since?"
# answerable at all: without it, a session start cannot tell a file it wrote itself
# last time from one Claude Code (or a human) has since added rules to.
PERSONAL_SETTINGS_SYNC_STAMP="${PROJECT_ROOT}/.claude/.personal-settings-sync-hash"

# personal_settings_are_locally_modified: returns 0 if this clone's local settings
# exist and differ from what was last synced or saved (so overwriting them would
# lose an edit - typically permission rules Claude Code itself appended after an
# "always allow"), 1 otherwise. Settings that exist but were never synced count as
# modified: nothing recorded them, so nothing may claim them.
personal_settings_are_locally_modified() {
  [ -f "${LOCAL_SETTINGS_JSON}" ] || return 1
  [ -f "${PERSONAL_SETTINGS_SYNC_STAMP}" ] || return 0
  [ "$(git hash-object "${LOCAL_SETTINGS_JSON}")" \
    != "$(cat "${PERSONAL_SETTINGS_SYNC_STAMP}")" ]
}

# record_personal_settings_sync: stamps the local settings' current content as the
# synced baseline, so the next session start may update them in place.
record_personal_settings_sync() {
  git hash-object "${LOCAL_SETTINGS_JSON}" > "${PERSONAL_SETTINGS_SYNC_STAMP}"
}

# current_branch_upstream_remote: prints the remote name the current branch
# tracks (e.g. "abdel-direct" for a branch whose upstream is
# "abdel-direct/some-branch"), or nothing if it has no upstream (detached
# HEAD, or a branch that was never pushed with -u/--set-upstream). Shared by
# fetch_personal_notes_branch below and by create-personal-notes-branch.sh's
# existence check, so both apply the exact same fallback remote.
current_branch_upstream_remote() {
  git rev-parse --abbrev-ref --symbolic-full-name @{upstream} 2>/dev/null | cut -d/ -f1
}

# fetch_personal_notes_branch: fetches NOTES_BRANCH from NOTES_REMOTE. If that
# fails (remote unreachable, or the branch just isn't there), falls back once
# to the current branch's own upstream remote (current_branch_upstream_remote
# above) - if it has one, and it differs from NOTES_REMOTE - before giving up.
# This covers the common case of a clone whose checked-out branch already
# tracks a contributor's own fork under some other remote name/URL, without
# requiring NOTES_REMOTE to be configured explicitly for it.
#
# On success: sets ACTIVE_NOTES_REMOTE to whichever remote actually served
# the branch (NOTES_REMOTE or the upstream fallback), leaves the fetched
# commit in FETCH_HEAD (see the note on FETCH_HEAD vs. "<remote>/<branch>"
# refs in session-start.sh), and returns 0.
# On failure: sets ATTEMPTED_NOTES_REMOTES to a human-readable, comma
# separated list of every remote that was tried (for callers that want to
# report it), and returns 1.
#
# Read-only fallback: this never affects where create-personal-notes-branch.sh
# creates the branch, or (by itself) where save-personal-notes.sh pushes an
# edit back to - callers that push should push back to ACTIVE_NOTES_REMOTE,
# i.e. wherever the branch was actually read from, not unconditionally to
# NOTES_REMOTE, so a save always lands on the same remote the notes came from.
fetch_personal_notes_branch() {
  ATTEMPTED_NOTES_REMOTES="${NOTES_REMOTE}"
  if git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet 2>/dev/null; then
    ACTIVE_NOTES_REMOTE="${NOTES_REMOTE}"
    return 0
  fi

  local upstream_remote
  upstream_remote="$(current_branch_upstream_remote)"
  if [ -n "${upstream_remote}" ] && [ "${upstream_remote}" != "${NOTES_REMOTE}" ]; then
    ATTEMPTED_NOTES_REMOTES="${ATTEMPTED_NOTES_REMOTES}, ${upstream_remote}"
    if git fetch "${upstream_remote}" "${NOTES_BRANCH}" --quiet 2>/dev/null; then
      ACTIVE_NOTES_REMOTE="${upstream_remote}"
      return 0
    fi
  fi

  return 1
}

# default_branch_name: prints the repo's actual default branch name, with no
# network access - resolved from origin's local HEAD ref
# (refs/remotes/origin/HEAD, set by a normal `git clone` or `git remote
# set-head`) when available, otherwise whichever of main/master actually
# exists as a local or origin-tracking branch, otherwise "main". Used by
# pr_progress_path below so a repo whose default branch is neither main nor
# master (e.g. "develop") is still recognized, instead of being silently
# treated as an ordinary per-branch PR-progress branch.
default_branch_name() {
  local remote_head candidate
  remote_head="$(git symbolic-ref -q refs/remotes/origin/HEAD 2>/dev/null)"
  if [ -n "${remote_head}" ]; then
    printf '%s\n' "${remote_head#refs/remotes/origin/}"
    return 0
  fi
  for candidate in main master; do
    if git show-ref --verify --quiet "refs/heads/${candidate}" \
        || git show-ref --verify --quiet "refs/remotes/origin/${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf 'main\n'
}

# pr_progress_path: prints the deterministic per-branch PR-progress file path
# (.claude/personal/pr-progress/<branch>.md) for whichever branch is currently
# checked out, and returns 0. Returns 1 (prints nothing) if there's no
# sensible "current PR" to track progress for: detached HEAD, the repo's
# default branch (see default_branch_name above), or the personal-notes
# branch itself. The directory is a fixed convention, independent of
# NOTES_PATH - PR progress is inherently plural/keyed, unlike the single
# personal-notes file, so it isn't tied to wherever NOTES_PATH happens to be
# overridden to.
#
# Shared by session-start.sh and save-pr-progress.sh so both agree on exactly
# the same key for exactly the same branch - there is no other place this
# path is computed, so it can never drift between reading and writing it.
pr_progress_path() {
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  case "${branch}" in
    HEAD|"$(default_branch_name)"|"${NOTES_BRANCH}"|"") return 1 ;;
  esac
  printf '.claude/personal/pr-progress/%s.md\n' "${branch}"
}

# PERSONAL_GIT_IDENTITY_PATH: where the human contributor's git identity is
# recorded on the notes branch, so a fresh clone can be given one instead of
# inheriting whatever the environment's global git config happens to be. Fixed
# convention, never overridden - same reasoning as the plan paths below.
#
# Stored in git's own config format and read back with `git config --file`
# rather than parsed here: the format already has a parser, and writing it
# with the same tool that reads it means the two can never disagree.
PERSONAL_GIT_IDENTITY_PATH=".claude/personal/git-identity"

# format_git_identity: prints a name and email in the one form every message
# about an identity uses, so the same pair can't be rendered two ways in two
# different reports.
format_git_identity() {
  printf '%s <%s>\n' "$1" "$2"
}

# effective_git_identity: prints "<name><TAB><email>" for the identity a commit
# made here right now would actually carry, and returns 0. Returns 1 (prints
# nothing) if git cannot determine one at all.
#
# Resolved via `git var GIT_AUTHOR_IDENT`, which applies git's real precedence -
# GIT_AUTHOR_NAME/GIT_AUTHOR_EMAIL, then repository-local config, then global.
# `git config --get user.name` deliberately not used: it reports the global
# value even in a clone whose commits are correctly authored from the
# environment, which is the one wrong answer a check about commit authorship
# must never give.
effective_git_identity() {
  local author_identity
  author_identity="$(git var GIT_AUTHOR_IDENT 2>/dev/null)" || return 1
  # GIT_AUTHOR_IDENT is "<name> <<email>> <timestamp> <timezone>"; the trailing
  # two fields are when the commit would be made, not who by.
  printf '%s\n' "${author_identity}" \
    | sed -E 's/^(.*) <(.*)> [0-9]+ [-+][0-9]{4}$/\1\t\2/'
}

# repository_local_git_identity: prints "<name><TAB><email>" for the identity
# configured in this clone's own config, and returns 0. Returns 1 (prints
# nothing) unless both halves are set - half an identity cannot author a commit,
# so it is not an identity.
repository_local_git_identity() {
  local name email
  name="$(git config --local --get user.name || true)"
  email="$(git config --local --get user.email || true)"
  [ -n "${name}" ] && [ -n "${email}" ] || return 1
  printf '%s\t%s\n' "${name}" "${email}"
}

# recorded_git_identity_exists / recorded_git_identity: whether the notes branch
# carries a git identity at all, and what it records. Caller must have already
# fetched NOTES_BRANCH successfully (see fetch_personal_notes_branch) - these
# read FETCH_HEAD directly rather than fetching again themselves.
#
# Two functions rather than one for the same reason as plan_branch_index_exists
# above: "nothing is recorded yet" and "what is recorded cannot be used" are
# different answers needing different advice, and a single failing lookup
# collapses them into one.
recorded_git_identity_exists() {
  git cat-file -e "FETCH_HEAD:${PERSONAL_GIT_IDENTITY_PATH}" 2>/dev/null
}

recorded_git_identity() {
  recorded_git_identity_exists || return 1
  local identity_file name email
  identity_file="$(mktemp)"
  git show "FETCH_HEAD:${PERSONAL_GIT_IDENTITY_PATH}" > "${identity_file}"
  name="$(git config --file "${identity_file}" --get user.name || true)"
  email="$(git config --file "${identity_file}" --get user.email || true)"
  rm -f "${identity_file}"
  [ -n "${name}" ] && [ -n "${email}" ] || return 1
  printf '%s\t%s\n' "${name}" "${email}"
}

# PLANS_DIR / PLAN_MANIFEST_FILENAME / PLAN_ROADMAP_FILENAME: the one,
# shared definition of where a plan's files live, so no caller re-derives
# these path fragments itself (session-start.sh and save-plan.sh both used
# to build ".claude/personal/plans/<id>/plan.yaml" inline - two independent
# copies of the same literal that could silently drift apart). Fixed
# convention, never overridden - plan storage is plural/generated data, not
# a per-clone preference like NOTES_PATH.
PLANS_DIR=".claude/personal/plans"
PLAN_MANIFEST_FILENAME="plan.yaml"
PLAN_ROADMAP_FILENAME="roadmap.md"

# plan_directory_path / plan_manifest_path / plan_roadmap_path: the
# deterministic per-plan paths for the given plan id. Shared by
# session-start.sh and save-plan.sh so both agree on exactly the same
# layout - see pr_progress_path above for the same reasoning applied to
# PR-progress files.
plan_directory_path() {
  printf '%s/%s\n' "${PLANS_DIR}" "$1"
}
plan_manifest_path() {
  printf '%s/%s/%s\n' "${PLANS_DIR}" "$1" "${PLAN_MANIFEST_FILENAME}"
}
plan_roadmap_path() {
  printf '%s/%s/%s\n' "${PLANS_DIR}" "$1" "${PLAN_ROADMAP_FILENAME}"
}

# PLAN_BRANCH_INDEX_PATH: the generated reverse index mapping every plan
# item's branch to the plan id that tracks it (see
# .claude/skills/plan-dashboard/plan-schema.md for the
# full plan-dashboard schema this feeds).
PLAN_BRANCH_INDEX_PATH="${PLANS_DIR}/_generated/branch-index.tsv"

# DASHBOARD_URL_CACHE_PATH: the generated cache mapping each plan id (plus
# "_index" for the master index) to the Artifact URL its dashboard is
# published at, so /plan-dashboard updates that page instead of minting a
# second one. Named here rather than typed into plan-dashboard/SKILL.md,
# same defined-once reasoning as PLAN_BRANCH_INDEX_PATH above.
DASHBOARD_URL_CACHE_PATH="${PLANS_DIR}/_generated/dashboard-urls.yaml"

# PLAN_DASHBOARD_DIRECTORY / *_SCRIPT / *_FILE / *_DOC: the canonical
# location of every script, hook, requirements file, and reference doc the
# plan-dashboard/plan-item-*/CI tooling invokes or reads - defined once,
# here, so refresh_dashboard.sh, every plan-*/SKILL.md, and
# .github/workflows/ci.yml source this file and use these variables instead
# of each carrying its own separately-typed literal path (exactly the drift
# risk a reviewer flagged after those paths had already been duplicated
# across all of them). Relative to the project root, which sourcing this
# file already `cd`s into (see PROJECT_ROOT above) - so every caller can
# use these directly, with no further path arithmetic of its own.
PLAN_DASHBOARD_DIRECTORY=".claude/skills/plan-dashboard"
# build_dashboard.py: renders one plan's dashboard HTML from its manifest
# and live GitHub data - see the script's own module docstring.
BUILD_DASHBOARD_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/build_dashboard.py"
# build_index.py: renders the master index page listing every plan.
BUILD_INDEX_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/build_index.py"
# sync_manifest_status.py: auto-corrects a plan.yaml's item statuses to
# "done" wherever GitHub confirms the item's pull request is merged.
SYNC_MANIFEST_STATUS_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/sync_manifest_status.py"
# check_dependency_readiness.py: classifies one item's dependencies as
# ready or not-ready to build on - see dependency-readiness.md below.
CHECK_DEPENDENCY_READINESS_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/check_dependency_readiness.py"
# refresh_dashboard.sh: orchestrates sync_manifest_status.py, the
# conditional push of its correction, then build_dashboard.py - the whole
# refresh sequence /plan-dashboard runs for one plan.
REFRESH_DASHBOARD_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/refresh_dashboard.sh"
# refresh_dashboard_support.py: the JSON-plumbing helpers
# refresh_dashboard.sh calls between its two script calls.
REFRESH_DASHBOARD_SUPPORT_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/refresh_dashboard_support.py"
# record_dashboard_url.py: writes one key's published Artifact URL into
# DASHBOARD_URL_CACHE_PATH, resolving that URL from the account's live
# Artifact listing so a URL nobody published cannot be recorded.
RECORD_DASHBOARD_URL_SCRIPT="${PLAN_DASHBOARD_DIRECTORY}/record_dashboard_url.py"
# requirements.txt: the PyYAML/Jinja2/markdown dependencies every script
# above needs - installed by both CI and a session running them directly.
PLAN_DASHBOARD_REQUIREMENTS_FILE="${PLAN_DASHBOARD_DIRECTORY}/requirements.txt"
# tests/: the pytest suite covering every script above - the exact
# directory CI and a session both run against.
PLAN_DASHBOARD_TESTS_DIRECTORY="${PLAN_DASHBOARD_DIRECTORY}/tests"
# hooks/tests/: the pytest suite covering plan_manifest_tools.py (the one
# hook-directory script with non-trivial logic worth testing the same way).
HOOKS_TESTS_DIRECTORY=".claude/hooks/tests"

# STACK_DIRECTORY / *_SCRIPT / *_CONFIG_FILE / *_TESTS_DIRECTORY: the
# stacked-PR fork-staging/cram2-review tooling's canonical location, same
# defined-once reasoning as the PLAN_DASHBOARD_* block above - so the
# setup-stacked-prs skill and the shared pr_state module reference these
# instead of retyping the literal paths.
STACK_DIRECTORY=".claude/stack"
# stack.py: read-only stacked-PR status tool (status/check/next/restack-plan)
# - see its own module docstring and STACK_DIRECTORY/README.md.
STACK_SCRIPT="${STACK_DIRECTORY}/stack.py"
# stack.toml: the committed defaults stack.py's load_configuration layers a
# personal-notes .claude/personal/stack.toml override on top of.
STACK_CONFIG_FILE="${STACK_DIRECTORY}/stack.toml"
# tests/: the pytest suite covering stack.py, including its personal-notes
# config-layering behaviour (via the hooks tests' ScratchRepository).
STACK_TESTS_DIRECTORY="${STACK_DIRECTORY}/tests"

# plan-schema.md: the full plan.yaml field reference every plan-* skill
# reads before drafting or interpreting a manifest. On main, next to the
# tooling that enforces it, so every clone has it with no setup - unlike the
# plan *data* it describes, which lives only on the personal-notes branch.
PLAN_SCHEMA_DOCUMENT="${PLAN_DASHBOARD_DIRECTORY}/plan-schema.md"
# dependency-readiness.md: the shared bulk-fetch-and-check procedure
# plan-item-kickoff and plan-item-resolve both reference instead of each
# restating it.
DEPENDENCY_READINESS_DOCUMENT="${PLAN_DASHBOARD_DIRECTORY}/dependency-readiness.md"
# pr-data-fetching.md: the shared "how to bulk-fetch pull request state
# into pr_data.json" procedure - referenced by dependency-readiness.md and
# every plan-*/SKILL.md that assembles pr_data.json, instead of each
# restating the GitHub API calls involved.
PULL_REQUEST_DATA_FETCHING_DOCUMENT="${PLAN_DASHBOARD_DIRECTORY}/pr-data-fetching.md"
# write-personal-notes-file.sh: generic commit-and-push-one-file-to-the
# personal-notes-branch helper, used by refresh_dashboard.sh (the manifest
# auto-sync correction) and plan-dashboard/SKILL.md (the dashboard-URL
# cache) alike.
WRITE_PERSONAL_NOTES_FILE_SCRIPT=".claude/hooks/write-personal-notes-file.sh"

# SETUP_PERSONAL_NOTES_DIRECTORY / *_DOCUMENT / STARTER_NOTES_FILE /
# CHECK_SETUP_SCRIPT: the one-time-setup half of this system - the skill a
# person runs first (/setup-personal-notes), the starter notes it offers, the
# shared "is this clone set up yet?" procedure every other skill defers to
# instead of restating it, and the read-only inspection script all of them
# call. Same defined-once reasoning as every path above.
SETUP_PERSONAL_NOTES_DIRECTORY=".claude/skills/setup-personal-notes"
# check-setup.sh: reports, as TSV, which parts of the setup are already done -
# the single source of truth for that question, so no caller re-implements
# "is the notes branch there?" with its own git plumbing.
CHECK_SETUP_SCRIPT=".claude/hooks/check-setup.sh"
# prerequisite-check.md: the shared "run check-setup.sh, offer
# /setup-personal-notes if it fails" procedure that plan-create,
# plan-dashboard, plan-item-kickoff and plan-item-resolve each reference in
# one line rather than each spelling it out.
SETUP_PREREQUISITE_DOCUMENT="${SETUP_PERSONAL_NOTES_DIRECTORY}/prerequisite-check.md"
# starter-notes.md: the default content /setup-personal-notes offers to seed a
# brand-new notes file with, so a first session starts from working
# conventions instead of an empty file.
STARTER_NOTES_FILE="${SETUP_PERSONAL_NOTES_DIRECTORY}/starter-notes.md"

# ADD_PLAN_ITEM_DIRECTORY / SCOPE_DECISION_DOCUMENT /
# CHECK_SCOPE_OVERLAP_SCRIPT / ADD_PLAN_ITEM_TESTS_DIRECTORY: the
# where-does-this-work-belong half of the system - the skill someone runs when
# describing new work (/add-plan-item), the shared scope rule all four plan
# skills defer to instead of each restating it, the script that gathers that
# rule's evidence, and its pytest suite. Same defined-once reasoning as every
# path above.
ADD_PLAN_ITEM_DIRECTORY=".claude/skills/add-plan-item"
# scope-decision.md: the shared "is this new work, or a change to work already
# in flight?" rule that plan-create, plan-item-kickoff, plan-item-resolve and
# add-plan-item each reference in a line rather than each spelling it out.
SCOPE_DECISION_DOCUMENT="${ADD_PLAN_ITEM_DIRECTORY}/scope-decision.md"
# check_scope_overlap.py: reports which of the work's paths the base branch
# lacks, and which unlanded branches already touch them - see the script's own
# module docstring.
CHECK_SCOPE_OVERLAP_SCRIPT="${ADD_PLAN_ITEM_DIRECTORY}/check_scope_overlap.py"
# tests/: the pytest suite covering check_scope_overlap.py - the exact
# directory CI and a session both run against.
ADD_PLAN_ITEM_TESTS_DIRECTORY="${ADD_PLAN_ITEM_DIRECTORY}/tests"

# SAVE_PLAN_SCRIPT: same reasoning as the block above, extended to
# save-plan.sh - unlike the other hook scripts in this directory (which are
# always run directly by a human, once, per hooks/README.md's own setup
# instructions), save-plan.sh is invoked from plan-create/SKILL.md's own
# bootstrap step, i.e. a real caller this codebase controls - the same
# duplication risk, just for a hook script instead of a plan-dashboard one.
SAVE_PLAN_SCRIPT=".claude/hooks/save-plan.sh"

# PLAN_ITEM_BOOTSTRAP_SCRIPT: same reasoning again, for the script that opens
# an item's branch and draft pull request and records its manifest entry -
# invoked from plan-item-kickoff/SKILL.md and add-plan-item/SKILL.md, so it is
# a path this codebase controls rather than one a human types once.
PLAN_ITEM_BOOTSTRAP_SCRIPT=".claude/hooks/plan_item_bootstrap.py"

# PLAN_ITEM_MODE_SCRIPT / PLAN_ITEM_MODES_CONFIG_FILE /
# PERSONAL_PLAN_ITEM_MODES_PATH: the script that resolves whether a plan-item
# skill asks, plans, or implements on its own, plus the two files it layers -
# committed defaults in this repository, per-user overrides on the
# personal-notes branch. Same committed-defaults/personal-override split as
# STACK_CONFIG_FILE above. Invoked from plan-item-kickoff/SKILL.md and
# plan-item-resolve/SKILL.md via execution-modes.md, so these are paths this
# codebase controls rather than ones a human types once.
PLAN_ITEM_MODE_SCRIPT=".claude/hooks/plan_item_mode.py"
PLAN_ITEM_MODES_CONFIG_FILE=".claude/hooks/plan-item-modes.toml"
PERSONAL_PLAN_ITEM_MODES_PATH=".claude/personal/plan-item-modes.toml"

# EXECUTION_MODES_DOCUMENT: the shared "which mode is in force, what it
# obliges, and when auto mode still asks" procedure that plan-item-kickoff and
# plan-item-resolve both reference instead of each restating it - same
# reasoning as DEPENDENCY_READINESS_DOCUMENT above.
EXECUTION_MODES_DOCUMENT="${PLAN_DASHBOARD_DIRECTORY}/execution-modes.md"

# PLAN_ITEM_GATHERING_DOCUMENT: the shared "what is already known and already
# decided about this item?" procedure - the setup check, resolving the item off
# the notes branch, the tracking-issue subscription, the full roadmap read, the
# dependency chain and the standing conventions. plan-item-kickoff and
# plan-item-resolve both run it in full and then add only what their own
# situation needs, instead of each carrying its own copy.
PLAN_ITEM_GATHERING_DOCUMENT="${PLAN_DASHBOARD_DIRECTORY}/plan-item-gathering.md"

# GITHUB_LIST_PULL_REQUESTS_TOOL / GITHUB_PULL_REQUEST_READ_TOOL: the two
# MCP tools every pr_data.json-gathering procedure in this system calls
# (see pr-data-fetching.md), named once here so every doc references the
# same constant instead of retyping the literal identifier. Documentation
# aliases only, not live substitutions: Claude Code's tool-calling
# mechanism has no notion of a shell-expanded tool name, so an actual call
# always still has to type the literal name below - but a session that has
# sourced this file can read `${GITHUB_LIST_PULL_REQUESTS_TOOL}` in a doc
# and know exactly which tool that refers to, the same way it already does
# for every script path above.
GITHUB_LIST_PULL_REQUESTS_TOOL="mcp__github__list_pull_requests"
GITHUB_PULL_REQUEST_READ_TOOL="mcp__github__pull_request_read"

# plan_id_for_branch: prints the plan id that tracks the given branch, per
# PLAN_BRANCH_INDEX_PATH on FETCH_HEAD, and returns 0. Returns 1 (prints
# nothing) if the index doesn't exist yet, or the branch isn't in it. Caller
# must have already fetched NOTES_BRANCH successfully (see
# fetch_personal_notes_branch) - this reads FETCH_HEAD directly rather than
# fetching again itself, so session-start.sh and save-plan.sh each fetch
# exactly once per run.
#
# The index is tab-separated values (TSV): one "<branch><TAB><plan-id>" line
# per branch, generated fresh in full by ./save-plan.sh on every run (never
# hand-edited or incrementally patched). TSV rather than a hand-rolled
# YAML-lookalike matched by fixed-string grep: it's an unambiguous, widely
# understood interchange format - a tab can never appear inside a branch
# name or plan id, so a field-based match can't misfire the way a
# substring/prefix match on a YAML-shaped string could - while still
# needing nothing beyond `awk`, which every session-start environment
# already has (see the module docstring: session-start.sh must not gain a
# hard dependency on python3/PyYAML just to check whether the current
# branch belongs to a plan).
plan_id_for_branch() {
  local branch="$1"
  git cat-file -e "FETCH_HEAD:${PLAN_BRANCH_INDEX_PATH}" 2>/dev/null || return 1
  git show "FETCH_HEAD:${PLAN_BRANCH_INDEX_PATH}" 2>/dev/null \
    | awk -F'\t' -v branch="${branch}" '$1 == branch { print $2; exit }'
}

# branch_can_hold_plan_item: whether a plan item could ever track the given
# branch. False for a detached HEAD, the repo's default branch, and the
# personal-notes branch: none of the three is per-change work, so telling a
# session "no item tracks this branch" there is noise rather than a prompt to
# record one - work done from them is typically a personal-notes edit that
# never becomes a pull request at all.
#
# Deliberately its own copy of the three cases pr_progress_path excludes,
# rather than a shared helper: the two answer different questions and are
# expected to diverge. A branch whose pull request targets the notes branch
# still wants PR progress tracked, but still never wants a plan item.
branch_can_hold_plan_item() {
  local branch="$1"
  case "${branch}" in
    HEAD|"$(default_branch_name)"|"${NOTES_BRANCH}"|"") return 1 ;;
  esac
  return 0
}

# plan_branch_index_exists / tracked_plan_count: whether any plan is tracked on
# the notes branch at all, and how many distinct ones there are. Same FETCH_HEAD
# precondition as plan_id_for_branch above.
#
# These exist so a caller can tell apart the two situations plan_id_for_branch
# collapses into a single "no": nobody tracks plans here, versus plans exist and
# this branch is in none of them. Only the second is worth a word to a session,
# and only these two functions know which is which, since the index path is
# theirs alone to read.
plan_branch_index_exists() {
  git cat-file -e "FETCH_HEAD:${PLAN_BRANCH_INDEX_PATH}" 2>/dev/null
}

tracked_plan_count() {
  plan_branch_index_exists || { printf '0\n'; return 0; }
  git show "FETCH_HEAD:${PLAN_BRANCH_INDEX_PATH}" 2>/dev/null \
    | awk -F'\t' 'NF >= 2 { seen[$2] = 1 } END { print length(seen) }'
}

# PLAN_STATE_SYNC_STAMP: gitignored file recording the personal-notes commit
# SHA that was FETCH_HEAD the last time this clone read plan state (either
# session-start.sh's own auto-discovery, or ./plan-updates-since.sh). This is
# the "last-seen SHA" the recheck-deltas convention in cram-notes.md is built
# around: a session that wants to know what changed since it last looked
# diffs from this stamp instead of rereading whole files - see
# ./plan-updates-since.sh, which is also what advances it.
PLAN_STATE_SYNC_STAMP="${PROJECT_ROOT}/.claude/.plan-state-sync-sha"

# record_plan_state_sync_stamp: stamps FETCH_HEAD as the notes-branch commit
# this clone has now read plan state at. Caller must have already fetched
# NOTES_BRANCH successfully (see fetch_personal_notes_branch) - reads
# FETCH_HEAD directly rather than fetching again itself, same reasoning as
# plan_id_for_branch above.
record_plan_state_sync_stamp() {
  git rev-parse FETCH_HEAD > "${PLAN_STATE_SYNC_STAMP}"
}

# last_recorded_plan_state_sha: prints the SHA record_plan_state_sync_stamp
# last recorded, and returns 0. Returns 1 (prints nothing) if nothing has
# been recorded yet - a fresh clone, or one whose session-start.sh predates
# this stamp.
last_recorded_plan_state_sha() {
  [ -f "${PLAN_STATE_SYNC_STAMP}" ] || return 1
  cat "${PLAN_STATE_SYNC_STAMP}"
}
