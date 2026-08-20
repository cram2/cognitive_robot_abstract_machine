# Personal Claude Code notes

These are personal workflow preferences, not project conventions. They live on
the personal-notes branch only and are pulled into every session by
`.claude/hooks/session-start.sh`; they are never merged into the default
branch.

Everything below is a starting point, offered by `/setup-personal-notes` — it
is yours now. Edit, delete, or replace any of it: ask Claude to "edit my
personal notes" in any session, or change `CLAUDE.local.md` between the
`BEGIN-PERSONAL-NOTES`/`END-PERSONAL-NOTES` markers and run
`.claude/hooks/save-personal-notes.sh`.

## Pull requests

- Always open pull requests as **drafts**. Never open a PR as ready-for-review
  by default; mark it ready only when explicitly told to.
- Always convert a PR back to **draft** after pushing any commit to it or
  otherwise modifying it, even if it was previously marked ready for review.
  Mark it ready again only when explicitly told to.
- Bug-fix PRs must always carry the **`bug`** label.
- Keep bug-fix PRs focused: one root cause per PR, based off the default
  branch, no unrelated cleanup bundled in.
- Always include a link to the session that created the PR in the PR
  description.
- Keep the PR description up to date: after pushing any change that alters
  what the PR does, update the description to match. Never leave it
  describing an earlier state of the PR.
- Never subscribe to a pull request's activity, and never offer to watch,
  monitor, babysit or autofix one. Opening a PR ends the session's obligation
  to it: push it, report in the chat what you did and what is still
  outstanding, and stop. Ask for a CI failure or a review comment to be
  handled when you want it handled.

## Review comments

- Resolve a review comment thread only once you have genuinely done what it
  asked. If instead you need to ask what to do, or you are not taking an
  action, do not resolve it — reply explaining the situation and asking the
  question.
- Always reply to a PR comment explaining what you did before resolving it.

## Before starting work

- Always fetch, pull, and merge from the original repository you cloned (the
  user-owned repository, whether it is a fork of another or not) before
  investigating problems, reacting to events, or implementing features, so
  you are always working from its latest state.

## PR plan and progress tracking

- For every PR you create, maintain a plan/progress/next-steps note in the
  PR-progress section of `CLAUDE.local.md` (the block between the
  BEGIN-PR-PROGRESS/END-PR-PROGRESS markers, written automatically by
  `session-start.sh`). Initialize it with a short plan as soon as you start
  real work on the PR.
- Keep it current: update it whenever the plan changes, whenever you update
  your task list, and before ending any turn that changed either. Run
  `.claude/hooks/save-pr-progress.sh` whenever you update it.
- Never write this plan into any file tracked on the PR branch itself. It
  must live only in the PR-progress section, which is stored on the
  personal-notes branch and is never merged.

## Plan-mode approval → persistent plans

- The moment a normal Claude Code plan-mode plan is approved, before
  implementing, judge whether the work spans multiple PRs/branches/sessions to
  complete. If it's contained in one PR from this session, just implement it —
  do not invoke anything below for it.
- If it spans multiple PRs/sessions:
  - **No existing plan covers it**: invoke `/plan-create <plan-id>`, handing it
    the just-approved plan-mode markdown directly as source material — it's
    valid input under that skill's "existing freeform doc to migrate" case even
    though it only lives in this conversation, not a file.
  - **An existing plan covers/extends it** (check auto-discovery on the current
    branch, or ask): edit `plan.yaml` directly and run
    `.claude/hooks/save-plan.sh` + `/plan-dashboard <plan-id>`, after asking
    what the change should be rather than deciding unilaterally — see
    `.claude/skills/plan-dashboard/plan-schema.md`'s "Proposing structural
    changes" section.
- This is the moment that decides whether the plan gets captured durably or
  evaporates once the session ends — do not let it pass by default.
