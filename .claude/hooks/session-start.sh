#!/bin/bash
set -euo pipefail

# Generic personal Claude Code notes hook.
#
# Populates CLAUDE.local.md (gitignored, never committed on any branch) from a
# personal branch on a remote (default: `origin`). No-op in effect for anyone
# who never creates that branch (default or overridden), and collision-free
# for multiple contributors sharing one remote if each overrides the branch
# name via local config instead of relying on the shared default.
#
# Always writes to the project root, never wherever this happens to be
# invoked from: a SessionStart hook's cwd isn't guaranteed to be the project
# root, so a bare `CLAUDE.local.md` could silently land in the wrong place
# (and Claude Code only auto-loads it from the root) - see CLAUDE_LOCAL_MD in
# ./resolve-personal-notes-config.sh for how that's resolved deterministically.
#
# Works out of the box, zero config: it looks for a branch named
# `claude/personal-notes` on `origin` and, if found, reads
# `.claude/personal/cram-notes.md` off it into CLAUDE.local.md.
# ./create-personal-notes-branch.sh creates that branch (with an empty notes
# file) for anyone who doesn't have it yet.
#
# Override the remote/branch/path per clone, locally (never committed):
#   git config claude.personalNotesRemote <remote-name-or-url>   # optional
#   git config claude.personalNotesBranch <your-branch-name>
#   git config claude.personalNotesPath   <path-on-that-branch>   # optional
#
# The remote defaults to `origin`, but only matters when your own notes live
# somewhere other than the clone's `origin` - e.g. some session environments
# name the upstream repo `origin` and your own fork something else. Set it to
# either a remote already configured in this clone (by name) or a raw git URL
# (`https://github.com/<you>/<repo>`) - `git fetch`/`git push` accept both, and
# a URL needs no `git remote add` first, so it works even in a clone that's
# never heard of your fork. See ./README.md for when to use which form.
#
# git config is per-clone, so it's the wrong mechanism anywhere sessions start
# from a fresh clone every time (e.g. cloud/web sessions) - there's no
# persistent .git/config for it to live in. For that case, override via
# persistent environment variables instead (configured once at the environment
# level, outside the repo, so they survive every fresh clone):
#   CLAUDE_PERSONAL_NOTES_REMOTE=<remote-name-or-url>   # optional
#   CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
#   CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>   # optional
# See ./README.md for exactly how to wire these into a cloud environment.
# Precedence: git config > environment variable > the zero-config default, so
# a local or environment-level override always wins over it.
#
# Safe to re-run: it only ever overwrites CLAUDE.local.md, and does nothing if
# the configured (or default) branch or path isn't reachable (e.g. a fresh
# clone, or a fork that never created it).
#
# Remote fallback: if NOTES_REMOTE doesn't have the branch, this also tries
# the current branch's own upstream remote (if it has one, and it differs
# from NOTES_REMOTE) before giving up - see fetch_personal_notes_branch in
# ./resolve-personal-notes-config.sh. Covers a clone whose checked-out branch
# already tracks your fork under some other remote name, with no config
# needed. The written header always names whichever remote actually served
# the notes, so it's clear which one was used.
#
# Editing your notes: the written CLAUDE.local.md starts with a short header
# (see below) naming the resolved branch/path and pointing at
# ./save-personal-notes.sh. Since Claude Code loads CLAUDE.local.md as project
# memory every session, that header is always in context - so asking Claude to
# "edit my personal notes" needs no other setup: it edits the notes between
# the BEGIN-PERSONAL-NOTES/END-PERSONAL-NOTES markers, then runs the save
# script to push the change back.
#
# Personal settings: the same branch may also carry a
# `.claude/personal/settings.local.json`, which is copied verbatim into this
# clone's `.claude/settings.local.json` - the file Claude Code reads as local
# settings, so personal permission rules, env vars and the like follow you into
# every clone the same way your notes do. It is never merged (gitignored, exactly
# like CLAUDE.local.md), and local edits to it are never overwritten - see the
# settings block near the bottom of this script and ./save-personal-settings.sh.
#
# PR progress: on any branch with a sensible "current PR" (i.e. not the
# default branch, a detached HEAD, or the personal-notes branch itself - see
# pr_progress_path in ./resolve-personal-notes-config.sh), CLAUDE.local.md
# also gets a second section for that branch's plan/progress/next-steps,
# keyed to the branch name and stored on the personal-notes branch just like
# the notes above - so it is never committed to the PR branch itself, and
# survives session restarts automatically. Always present (as a scaffold, if
# nothing's been saved yet) on such a branch, so the agent is nudged to
# initialize and maintain it from the start. See ./save-pr-progress.sh.
#
# Plan auto-discovery: if the current branch appears as an item in some
# multi-PR/multi-session plan (see .claude/skills/plan-dashboard/plan-schema.md
# and .claude/skills/plan-dashboard/SKILL.md), CLAUDE.local.md also gets
# that plan's manifest
# (plan.yaml) and narrative (roadmap.md) pulled in - so a session picks up
# the wider initiative its branch belongs to without anyone having to ask it
# to go read a roadmap doc by hand. Looked up via the generated
# branch->plan-id reverse index (plan_id_for_branch in
# ./resolve-personal-notes-config.sh), never hand-maintained, so it can't
# drift out of sync with the plans it's derived from. Unlike PR progress
# above, there is no scaffold for a branch with no plan - most branches
# don't belong to one, and that's normal. See ./save-plan.sh to push edits
# back (regenerating the reverse index too).
#
# If the plan has a `tracking_issue` set, the written header also reminds a
# session to always comment there when it makes a structural change (new
# phases, deferring a track, etc.) in addition to editing the manifest
# directly - any session may make structural changes, there is no
# designated steward - and to subscribe to the tracking issue itself while
# actively working an item, so another session's structural change reaches
# it in real time - see plan-schema.md's "Proposing structural changes"
# section for the full convention.
#
# Recheck stamp: every run also records the personal-notes commit this
# clone just fetched (gitignored, see PLAN_STATE_SYNC_STAMP in
# ./resolve-personal-notes-config.sh), regardless of whether this branch
# tracks a plan. ./plan-updates-since.sh diffs from that stamp instead of a
# session rereading whole plan files to answer "what changed since I last
# looked" - see that script and cram-notes.md's recheck-deltas convention.
#
# How this script gets invoked (see ../settings.json): Claude Code registers it
# as a SessionStart hook via `$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh`.
# CLAUDE_PROJECT_DIR is an env var Claude Code itself injects into every hook
# command's environment, resolving to this project's root - so that path is
# correct regardless of Claude Code's own cwd when it runs the hook.
#
# Coexistence with your own settings: Claude Code merges the `hooks` arrays
# across all settings layers (managed > CLI args > .claude/settings.local.json
# > .claude/settings.json (this repo's, committed) > ~/.claude/settings.json)
# by concatenation, not override. So this SessionStart hook runs alongside -
# never instead of - any SessionStart hook you already have configured for
# yourself. settings.json is strict JSON with no comment support, which is why
# this explanation lives here instead of there.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

fetch_personal_notes_branch || exit 0

# FETCH_HEAD, not "${ACTIVE_NOTES_REMOTE}/${NOTES_BRANCH}": a URL-form remote
# creates no remote-tracking ref, but FETCH_HEAD always points at what was
# just fetched, whether the serving remote was a name or a raw URL.

# Stamp this run's baseline unconditionally - not only when the current
# branch turns out to track a plan below. This is the whole branch's tip,
# fetched regardless, so it's just as valid a "last time I looked" baseline
# for a session working on a plan more broadly (e.g. a plan-item-kickoff
# session on a branch that isn't itself a tracked item) as for one on a
# tracked item's own branch. See ./plan-updates-since.sh, the recheck tool
# this stamp exists for.
record_plan_state_sync_stamp

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
# Sanitized copy for embedding into this script's <!-- ... --> HTML comment
# headers: a branch name containing "-->" would otherwise let it break out
# of the comment. Lookups (pr_progress_path, plan_id_for_branch) still use
# the unsanitized ${CURRENT_BRANCH}.
CURRENT_BRANCH_FOR_COMMENT="${CURRENT_BRANCH//-->/}"

OUTPUT_FILE="$(mktemp)"
trap 'rm -f "${OUTPUT_FILE}"' EXIT
WROTE_ANYTHING=0

# SUMMARY_*: what this run actually found/wrote, printed as a deterministic
# report at the end (see the bottom of this script) instead of leaving a
# session to describe secondhand, in its own prose, what the hook did.
SUMMARY_NOTES="not found"
SUMMARY_PROGRESS="not applicable (no current PR on this branch)"
SUMMARY_PLAN="none"

if git cat-file -e "FETCH_HEAD:${NOTES_PATH}" 2>/dev/null; then
  cat <<HEADER >> "${OUTPUT_FILE}"
<!--
Personal notes, synced from '${NOTES_BRANCH}' (${NOTES_PATH}) on remote
'${ACTIVE_NOTES_REMOTE}' by session-start.sh.
To edit: change the notes between the markers below, then run
  "\$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
to push the change back. This header and the markers are regenerated every
session from git config/environment/default (plus a same-branch-upstream
fallback) - editing them has no effect; only content between the markers is
ever saved.
-->
<!-- BEGIN-PERSONAL-NOTES -->
HEADER
  git show "FETCH_HEAD:${NOTES_PATH}" >> "${OUTPUT_FILE}"
  echo "<!-- END-PERSONAL-NOTES -->" >> "${OUTPUT_FILE}"
  WROTE_ANYTHING=1
  SUMMARY_NOTES="loaded from '${NOTES_BRANCH}' (${NOTES_PATH}) on '${ACTIVE_NOTES_REMOTE}'"
fi

PROGRESS_PATH="$(pr_progress_path || true)"
if [ -n "${PROGRESS_PATH}" ]; then
  [ "${WROTE_ANYTHING}" = "1" ] && printf '\n' >> "${OUTPUT_FILE}"
  cat <<PROGRESS_HEADER >> "${OUTPUT_FILE}"
<!--
PR progress for branch '${CURRENT_BRANCH_FOR_COMMENT}', synced from
'${NOTES_BRANCH}' (${PROGRESS_PATH}) on remote '${ACTIVE_NOTES_REMOTE}' by
session-start.sh. Maintain the current plan, what's done, and what's next
here throughout work on this PR. It is never merged: it lives only on
'${NOTES_BRANCH}', never on this branch. A stale file left behind after the
PR merges is harmless (just unread from then on) - delete it directly on
'${NOTES_BRANCH}' if you want to tidy it up.
To edit: change the notes between the markers below, then run
  "\$CLAUDE_PROJECT_DIR/.claude/hooks/save-pr-progress.sh"
to push the change back. This header and the markers are regenerated every
session - editing them has no effect; only content between the markers is
ever saved.
-->
<!-- BEGIN-PR-PROGRESS -->
PROGRESS_HEADER
  if git cat-file -e "FETCH_HEAD:${PROGRESS_PATH}" 2>/dev/null; then
    git show "FETCH_HEAD:${PROGRESS_PATH}" >> "${OUTPUT_FILE}"
    SUMMARY_PROGRESS="loaded for branch '${CURRENT_BRANCH}' (${PROGRESS_PATH})"
  else
    cat <<'SCAFFOLD' >> "${OUTPUT_FILE}"
No progress recorded yet for this branch. Initialize it now: a short plan,
what's done so far, and what's next. Keep it current as you work.
SCAFFOLD
    SUMMARY_PROGRESS="no saved progress yet for branch '${CURRENT_BRANCH}' - scaffold written"
  fi
  echo "<!-- END-PR-PROGRESS -->" >> "${OUTPUT_FILE}"
  WROTE_ANYTHING=1
fi

PLAN_ID="$(plan_id_for_branch "${CURRENT_BRANCH}" || true)"
if [ -n "${PLAN_ID}" ]; then
  PLAN_MANIFEST_PATH="$(plan_manifest_path "${PLAN_ID}")"
  PLAN_ROADMAP_PATH="$(plan_roadmap_path "${PLAN_ID}")"
  if git cat-file -e "FETCH_HEAD:${PLAN_MANIFEST_PATH}" 2>/dev/null; then
    [ "${WROTE_ANYTHING}" = "1" ] && printf '\n' >> "${OUTPUT_FILE}"
    # TRACKING_ISSUE: a plain top-level scalar, so grep/sed suffices here too -
    # same dependency-free reasoning as plan_id_for_branch above. Empty if
    # the plan has no tracking_issue set (nothing to extract, not an error).
    # Named for the mailbox's role, not necessarily a literal GitHub Issue -
    # see plan-schema.md's PR-fallback note for repos with Issues disabled.
    TRACKING_ISSUE="$(git show "FETCH_HEAD:${PLAN_MANIFEST_PATH}" 2>/dev/null \
      | grep -oE '^tracking_issue:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+$')"
    if [ -n "${TRACKING_ISSUE}" ]; then
      TRACKING_ISSUE_NOTE="Structural changes (a new wave/phase, deferring a track, splitting an
item, reprioritizing) can be made directly to the manifest by any session -
there is no designated steward gatekeeping them. Before making one, ask the
user in this session (e.g. via AskUserQuestion) rather than deciding
unilaterally - a structural change is the user's call, not something to
infer and apply silently just because editing the manifest directly is
technically allowed. Once they confirm, make the edit and always also leave
a comment on the tracking issue (#${TRACKING_ISSUE}) describing it, since
the user reviews structural changes there and it is the shared record other
sessions working this plan can check - see plan-schema.md's 'Proposing
structural changes' section. If this session is actively working an item in
this plan, also subscribe to the tracking issue itself (in addition to your
own item's PR) so a structural change another session makes reaches you
while you're still working, not just next session start."
    else
      TRACKING_ISSUE_NOTE="This plan has no tracking_issue set, so there is no coordination
mailbox for structural changes yet - edit the manifest directly as usual."
    fi
    cat <<PLAN_HEADER >> "${OUTPUT_FILE}"
<!--
Plan manifest for '${PLAN_ID}', synced from '${NOTES_BRANCH}'
(${PLAN_MANIFEST_PATH}) on remote '${ACTIVE_NOTES_REMOTE}' by
session-start.sh. This branch is tracked as an item in this plan - see
.claude/skills/plan-dashboard/plan-schema.md for the schema and
.claude/skills/plan-dashboard/SKILL.md for how it's used and refreshed.
To edit: change the manifest between the markers below, then run
  "\$CLAUDE_PROJECT_DIR/.claude/hooks/save-plan.sh"
to push the change back (this also regenerates the branch index), then run
/plan-dashboard ${PLAN_ID} to refresh its dashboard - save-plan.sh can't call
the Artifact tool itself. This header and the markers are regenerated every
session - editing them has no effect; only content between the markers is
ever saved.

${TRACKING_ISSUE_NOTE}
-->
<!-- BEGIN-PLAN-MANIFEST: ${PLAN_ID} -->
PLAN_HEADER
    git show "FETCH_HEAD:${PLAN_MANIFEST_PATH}" >> "${OUTPUT_FILE}"
    echo "<!-- END-PLAN-MANIFEST -->" >> "${OUTPUT_FILE}"

    printf '\n' >> "${OUTPUT_FILE}"
    cat <<ROADMAP_HEADER >> "${OUTPUT_FILE}"
<!--
Plan roadmap (narrative) for '${PLAN_ID}' - the "why", history, and design
decisions behind the manifest above. Same edit/save mechanism: change
between the markers below, then run save-plan.sh.
-->
<!-- BEGIN-PLAN-ROADMAP: ${PLAN_ID} -->
ROADMAP_HEADER
    if git cat-file -e "FETCH_HEAD:${PLAN_ROADMAP_PATH}" 2>/dev/null; then
      git show "FETCH_HEAD:${PLAN_ROADMAP_PATH}" >> "${OUTPUT_FILE}"
    fi
    echo "<!-- END-PLAN-ROADMAP -->" >> "${OUTPUT_FILE}"
    WROTE_ANYTHING=1
    SUMMARY_PLAN="'${PLAN_ID}' (tracking issue: ${TRACKING_ISSUE:-none})"
  fi
fi

if [ "${WROTE_ANYTHING}" = "1" ]; then
  mv "${OUTPUT_FILE}" "${CLAUDE_LOCAL_MD}"
else
  rm -f "${OUTPUT_FILE}"
fi

# Personal settings: unlike everything above, these don't go into CLAUDE.local.md -
# they are copied verbatim to .claude/settings.local.json, the file Claude Code
# itself reads as this project's local settings (see PERSONAL_SETTINGS_PATH in
# ./resolve-personal-notes-config.sh). No header or markers: it is strict JSON,
# which has no comment syntax to carry them.
#
# Locally modified settings are never overwritten. Claude Code writes to this same
# file whenever a permission is granted with "don't ask again", so a blind copy
# every session start would silently drop those grants - see
# personal_settings_are_locally_modified. Run ./save-personal-settings.sh to push
# such edits up, which makes them the new baseline and lets syncing resume.
SUMMARY_SETTINGS="none on '${NOTES_BRANCH}' (${PERSONAL_SETTINGS_PATH})"
if git cat-file -e "FETCH_HEAD:${PERSONAL_SETTINGS_PATH}" 2>/dev/null; then
  if personal_settings_are_locally_modified; then
    SUMMARY_SETTINGS="kept local edits to ${LOCAL_SETTINGS_RELATIVE_PATH} - run save-personal-settings.sh to push them"
  else
    mkdir -p "$(dirname "${LOCAL_SETTINGS_JSON}")"
    git show "FETCH_HEAD:${PERSONAL_SETTINGS_PATH}" > "${LOCAL_SETTINGS_JSON}"
    record_personal_settings_sync
    SUMMARY_SETTINGS="synced to ${LOCAL_SETTINGS_RELATIVE_PATH}"
  fi
fi

# Deterministic session-start report: what this run found and wrote, printed
# once by the script itself rather than left for a session to notice and
# describe secondhand from CLAUDE.local.md's content. SessionStart hook
# stdout is surfaced as context for the session, so this is the guaranteed
# confirmation of what got loaded - not something the model has to remember
# to summarize on its own.
cat <<SUMMARY
session-start.sh summary:
  personal notes:  ${SUMMARY_NOTES}
  local settings:  ${SUMMARY_SETTINGS}
  PR progress:     ${SUMMARY_PROGRESS}
  plan:            ${SUMMARY_PLAN}
  plan state SHA:  $(git rev-parse FETCH_HEAD) (run plan-updates-since.sh <plan-id> to recheck from here later)
SUMMARY
