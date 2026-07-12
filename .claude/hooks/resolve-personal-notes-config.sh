# Sourced (not executed) by session-start.sh, create-personal-notes-branch.sh
# and save-personal-notes.sh, so all three resolve the personal-notes remote,
# branch and path with the exact same precedence: git config > environment
# variable > the zero-config default. See ./README.md.

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
