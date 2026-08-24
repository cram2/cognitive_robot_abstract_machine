---
name: plan-item-resolve
description: Gather everything available about one already-underway tracked plan item (its plan.yaml entry, roadmap.md history, the real state of its branch/PR - conflicts, CI, review comments, and any unresolved review threads on its upstream pull request - and any relevant discussion on its plan's tracking issue), then resolve whatever is stalling it in whichever execution mode is in force - presenting a plan for approval, carrying it out directly on the item's existing branch, or asking which. Invoke as "/plan-item-resolve <plan-id> <item-id>". Use when resolving a blocked, in-progress, or deferred item from a plan-dashboard's "Resolve"/"Resume"/"Reconsider" link, or when the user asks to "resolve", "unblock", "resume", or "reconsider" a specific tracked item.
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion, Skill, EnterPlanMode, ExitPlanMode, mcp__github__list_pull_requests, mcp__github__pull_request_read, mcp__github__issue_read, mcp__github__get_file_contents, mcp__github__update_pull_request, mcp__Claude_Code_Remote__subscribe_pr_activity
---

# Plan Item Resolve

Generic, plan-agnostic — nothing here may hardcode a specific plan id,
item, or branch. Unlike `plan-item-kickoff` (for an item that hasn't
started), this skill is for an item that already has real state - a
branch, a PR, prior review, a recorded blocker - and needs that state
understood before deciding what to do next.

**Step 1 is research only — it creates nothing and writes no code.**
Step 2 resolves the item's execution mode, and that mode decides what
happens next: `plan` presents the plan and stops until the user approves it,
`auto` carries it out on the item's existing branch without asking, and
`ask` puts the choice to the user. `${EXECUTION_MODES_DOCUMENT}` states what
each mode means and what it still obliges.

This skill never creates a branch or a pull request — the item already has
both, and an item that has neither belongs to `plan-item-kickoff`. Every
invocation starts fresh in the current session; it does not try to detect or
resume any other session.

## 1. Gather the item's context

Follow `${PLAN_ITEM_GATHERING_DOCUMENT}` end to end — the setup check, the
item's resolution off the personal-notes branch, the tracking-issue
subscription, its recorded state, the full roadmap read, the dependency
chain, and the standing conventions. `plan-item-kickoff` runs the same
procedure, which is why it lives there rather than in either skill.

Then add the part only a resolve needs — **the live state of work that
already exists**, which is where the cause of a stall almost always is:

- If `pull_request_number` is set: fetch the PR (`mcp__github__pull_request_read`,
  `method: "get"`) for its mergeable state and CI status
  (`method: "get_check_runs"`), then its review threads
  (`method: "get_review_comments"`) and plain comments
  (`method: "get_comments"`) — read every one, not just the most recent,
  since an older unresolved thread is exactly the kind of thing this skill
  exists to surface. A failing check or a requested-changes review is
  usually the actual blocker; state exactly which one and why, don't just
  say "CI is failing."
- If the fork PR carries the `in_review_label` from `.claude/stack/stack.toml`
  (`in-review` by default, the recorded signal for "promoted upstream, under
  review"), the branch also has a pull request on the upstream, whose review
  threads none of the calls above can see — a fork PR can look entirely clean
  while the item is in fact stalled on an upstream request for changes. Invoke
  `/upstream-reviews` for the item's `branch` and read every unresolved thread
  it reports. Invoke it even without the label when `notes`/`status` say the
  item is under upstream review; skip it otherwise, since a branch never
  promoted has no upstream PR to read. If the dispatch or the run fails, don't
  let that fail the skill: mention it when presenting the plan (step 5) and
  continue — upstream state is valuable context, not a precondition.
- If the fork PR carries the `in_review_label` from `.claude/stack/stack.toml`
  (`in-review` by default, the recorded signal for "promoted upstream, under
  review"), the branch also has a pull request on the upstream, whose review
  threads none of the calls above can see — a fork PR can look entirely clean
  while the item is in fact stalled on an upstream request for changes. Invoke
  `/upstream-reviews` for the item's `branch` and read every unresolved thread
  it reports. Invoke it even without the label when `notes`/`status` say the
  item is under upstream review; skip it otherwise, since a branch never
  promoted has no upstream PR to read. If the dispatch or the run fails, don't
  let that fail the skill: mention it when drafting the plan (step 3) and
  continue — upstream state is valuable context, not a precondition.
- If the item has no PR yet (e.g. blocked before ever starting): there is
  no PR-side state to check — rely on `blockers`/`notes` and the tracking
  issue instead.
- If the plan has a `tracking_issue`, fetch its comments
  (`mcp__github__issue_read`, `method: "get_comments"`) and read every one
  that mentions this item by id, branch, or title — a structural change
  proposed there (a dependency change, a scope split) can be exactly why
  an item stalled.
- If a branch or PR exists, read what's actually in it
  (`mcp__github__pull_request_read` for the diff/description,
  `mcp__github__get_file_contents` or a local `git fetch` + `git show` for
  the real file contents) before proposing anything — the plan must resolve
  the real, current state, not a guessed one. For sibling items in the same
  track that already landed, read their merged diffs the same way
  `plan-item-kickoff` does, when the resolution involves matching an
  established pattern (e.g. a review comment asking this item to follow what
  a later sibling already settled on).

## 2. Resolve how this item gets resolved

Follow `${EXECUTION_MODES_DOCUMENT}`: run its `resolve --skill resolve`
call, and if the answer is `ask`, put its question — with a recommendation
drawn from what step 1 just turned up, and the reasons behind it.

Do this only once step 1 is done. What decides the recommendation here
is whether the cause of the stall is actually identified: a named failing
check or review comment with an obvious fix argues for going ahead, and a
blocker whose real cause is still a guess argues for planning first.

Pass `--requested <mode>` when the user named a mode in the invocation
itself.

## 3. Draft the plan

Apply `${PLAN_ITEM_GATHERING_DOCUMENT}`'s last section first: anything you
are about to raise as an open question is very often already answered by the
material step 1 gathered — here including the pull request's own review
threads and the tracking issue's discussion.

Also ask whether the item should still exist separately: follow
`${SCOPE_DECISION_DOCUMENT}`. If nothing substantial would remain once the
overlapping edits are removed, folding it into that item is often the
resolution — an item stuck behind its own parent is sometimes stuck because it
was never really a separate item. The same goes when two items turn out to have
built the same thing, which that document's purpose comparison is there to
catch.

Draft a concrete plan to resolve the item: what's actually wrong (cite the
specific failing check, review comment, unresolved upstream review thread,
blocker text, or regressed dependency that's the real cause — never a vague
"something's blocking this"), what changes it requires, in which files, in
what order, and how each part will be verified. Cite where each part of the
plan came from so it can be sanity-checked against the source. Flag
explicitly, never silently paper over:

- Any dependency that regressed or still isn't safe to build on.
- Any conflict between what `blockers`/`notes` says and what the PR's own
  review threads or the tracking issue actually say.
- Any conflict between the fork PR's state and the upstream review: a fork
  PR that is green and out of draft while its upstream pull request has
  unresolved threads is exactly the stall this skill exists to surface.
- Whether upstream review state was read at all, when the item looked
  promoted but `/upstream-reviews` could not be run.
- Anything genuinely unresolved after the check above — say so rather
  than filling the gap with an assumption.

In `plan` mode, present it via `ExitPlanMode` and stop there — nothing below
happens until the user approves it, and whether to carry the approved plan
out in this session or a fresh one is their call. In `auto` mode, don't ask:
the plan is settled the moment it is drafted, and the flags above go into
the record described below instead of into a question.

Either way, write no code in this step — its only output is the plan itself.

## 4. Carry it out — `auto` mode only

Work the plan on the item's existing branch, honoring the standing
conventions step 1 cross-checked: tests first per TDD, commits in the user's
own git identity, no assistant author or co-author trailer.

`${EXECUTION_MODES_DOCUMENT}` states what stays owed while doing it, and the
bar for stopping to ask anyway. Two things are this skill's own, because the
item is already underway rather than being started:

- **The record is an update, not a first draft.** Append what the resolution
  turned out to be to `roadmap.md` — especially a blocker whose recorded
  cause turned out to be wrong — and refresh the PR-progress note and the
  pull request description rather than writing them from scratch. Update the
  item's `blockers`, `notes` and `status` where the resolution changed what
  the item means, then republish the dashboard with `/plan-dashboard`.
- **The pull request goes back to draft after the push**, per the user's own
  convention, unless they marked it ready themselves — in which case the
  item was finished and this skill should not have been resolving it.

Finish by reporting what was wrong, what was changed, and what was decided —
in `auto` mode this report is the user's first look at the resolution.
