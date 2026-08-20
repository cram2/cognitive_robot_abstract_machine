#!/bin/bash
# The wording of session-start.sh's summary lines, defined once.
#
# Sourced by ./session-start.sh, which prints these, and called directly by the
# tests that assert on them - so a reworded message changes both sides at once
# instead of drifting apart from a second copy typed into an assertion.
#
# One function per outcome rather than one template string per outcome: the
# arguments are then named and positional in the same place the sentence is
# written, and a caller that passes the wrong number fails loudly here rather
# than rendering a half-substituted line.
#
# Deliberately holds no logic. Deciding *which* message applies is
# session-start.sh's business; this file only says how each one reads.

# %% the plan line

# plan_line_not_applicable: for a branch no plan item could ever track - the
# default branch, the notes branch, a detached HEAD.
plan_line_not_applicable() {
  printf 'not applicable (this branch never holds a plan item)'
}

# plan_line_no_plans_tracked: plans are not in use on the notes branch at all.
plan_line_no_plans_tracked() {
  local notes_branch="$1"
  printf "no plans tracked on '%s' yet" "${notes_branch}"
}

# plan_line_no_item_tracks_branch: plans are in use, and none holds an item for
# this branch. Even-handed on purpose: belonging to no plan is an ordinary
# state for most branches and must not read as a reprimand.
plan_line_no_item_tracks_branch() {
  local branch="$1"
  local tracked_plan_count="$2"
  printf "no item tracks branch '%s' (%s plan(s) tracked) - if this session's work belongs to one of them, add its item before starting; if it belongs to none, there is nothing to do" \
    "${branch}" "${tracked_plan_count}"
}

# plan_line_manifest_missing: the index names a plan whose manifest is not on
# the notes branch, so the two have drifted apart.
plan_line_manifest_missing() {
  local plan_id="$1"
  local manifest_path="$2"
  local notes_branch="$3"
  printf "'%s' tracks this branch, but %s is missing on '%s'" \
    "${plan_id}" "${manifest_path}" "${notes_branch}"
}

# plan_line_tracked: the branch is a tracked item of a plan that resolved.
plan_line_tracked() {
  local plan_id="$1"
  local tracking_issue="$2"
  printf "'%s' (tracking issue: %s)" "${plan_id}" "${tracking_issue}"
}

# %% the git identity line

# git_identity_line_not_recorded: the notes branch carries no identity at all,
# so there is nothing to write into this clone.
git_identity_line_not_recorded() {
  local notes_branch="$1"
  local identity_path="$2"
  printf "not recorded on '%s' (%s) - run ./save-git-identity.sh to record one" \
    "${notes_branch}" "${identity_path}"
}

# git_identity_line_incomplete: an identity is recorded but only half of it,
# and half an identity cannot author a commit.
git_identity_line_incomplete() {
  local identity_path="$1"
  local notes_branch="$2"
  printf "%s on '%s' needs both user.name and user.email - nothing written" \
    "${identity_path}" "${notes_branch}"
}

# git_identity_line_already_set: this clone has an identity of its own, which
# the hook only ever fills a gap around rather than overriding.
git_identity_line_already_set() {
  local identity="$1"
  printf 'already set in this clone: %s - left unchanged' "${identity}"
}

# git_identity_line_written: the recorded identity was written into this
# clone's repository-local config.
git_identity_line_written() {
  local notes_branch="$1"
  local identity_path="$2"
  local identity="$3"
  printf "set from '%s' (%s): %s" "${notes_branch}" "${identity_path}" "${identity}"
}

# %% the setup line

# setup_line_not_checked: check-setup.sh is not in this checkout, so there is
# no verdict to report rather than a passing one.
setup_line_not_checked() {
  local check_setup_script="$1"
  printf 'not checked - %s is not in this checkout' "${check_setup_script}"
}

# setup_line_ok: every check passed.
setup_line_ok() {
  printf 'ok'
}

# setup_line_needs_setup: the heading above the indented needs-setup rows,
# which check-setup.sh itself words.
setup_line_needs_setup() {
  local needs_setup_count="$1"
  printf '%s check(s) need setup - run /setup-personal-notes:' "${needs_setup_count}"
}
