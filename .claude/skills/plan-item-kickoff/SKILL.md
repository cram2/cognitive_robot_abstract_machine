---
name: plan-item-kickoff
description: Gather everything available about one tracked plan item (its plan.yaml entry, roadmap.md history/design context, its dependency chain's live GitHub state, and patterns from already-landed sibling items in the same track), propose a concrete implementation plan via plan mode without writing any code, and once it is approved open the item's branch and draft pull request and record its manifest state before implementation starts. Invoke as "/plan-item-kickoff <plan-id> <item-id>". Use when starting work on a specific item from a plan-dashboard's "Start now" link, or when the user asks to "start", "kick off", or "plan out" a specific tracked item.
allowed-tools: Bash, Read, Grep, Glob, AskUserQuestion, Skill, EnterPlanMode, ExitPlanMode, mcp__github__list_pull_requests, mcp__github__pull_request_read, mcp__github__get_file_contents, mcp__Claude_Code_Remote__subscribe_pr_activity
---

# Plan Item Kickoff

Generic, plan-agnostic — nothing here may hardcode a specific plan id, item,
or branch. Gathers everything a session doing this item's work would
actually want before starting, then hands the user a concrete
implementation plan via plan mode. **This skill never writes code, and
creates nothing at all until the user has approved a plan** — steps 1-5 are
research and planning only. Once a plan *is* approved, step 6 opens the
item's branch and draft pull request and records its manifest state, before
any implementation begins. Whether to implement the approved plan in this
session or a fresh one is the user's call, made after they see it; step 6
runs either way, so the item stops reading as `not_started` the moment it
isn't.

## 0. Check the setup is in place, and offer it if not

The item's manifest entry and roadmap live on the personal-notes branch, which
the user may not have set up yet. Follow
`.claude/skills/setup-personal-notes/prerequisite-check.md` before step 1: run
the check, and if it reports anything missing, offer `/setup-personal-notes`
rather than failing on a branch that isn't there.

## 1. Resolve the item

Source the shared config script — it resolves the personal-notes
remote/branch precedence and defines `DEPENDENCY_READINESS_DOCUMENT` (used in
step 2):

```bash
source .claude/hooks/resolve-personal-notes-config.sh
git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet
```

Load `<plan-id>/plan.yaml` + `roadmap.md` off `FETCH_HEAD` (same resolution
`plan-dashboard`'s own step 1 uses — read `resolve-personal-notes-config.sh`
if the precedence is unclear rather than re-deriving it). Find the item by
`id` (or `branch` if `id` is unset) among `items[]`.

If the plan id or item id doesn't resolve, stop and list what's actually
available (every plan id under `plans/*/plan.yaml`, or every item id in the
named plan) rather than guessing which one was meant.

If the plan has a `tracking_issue`, subscribe to it now via
`mcp__Claude_Code_Remote__subscribe_pr_activity` (it takes a plain issue
number the same way it takes a PR number). This session may go on to create
this item's branch and PR without a fresh session ever starting — the
subscription `session-start.sh` sets up for an already-checked-out item
branch never fires in that case — so subscribing here, before gathering
context, is what actually covers a kickoff that turns into an
uninterrupted implementation session. Skip this step entirely if the plan
has no `tracking_issue`. The call is idempotent, so it's safe to run even
if something already subscribed this session. If it errors, don't let that
fail the skill: mention it in passing when presenting the plan (step 5) and
continue — subscribing is a convenience for staying aware of concurrent
structural changes, not a precondition for planning this item.

## 2. Gather the item's own context

- `title`, `notes`, `blockers`, `track` (and that track's own `name` +
  `description`), `wave`.
- `depends_on`: follow `${DEPENDENCY_READINESS_DOCUMENT}`'s bulk-fetch-and-check
  procedure, for `--item <item-id>`. The plan needs to know exactly what
  branch to base new work on, and whether it's actually safe to build on
  yet: flag any dependency the script reports not ready for explicitly in
  the proposed plan's assumptions, instead of quietly proceeding as if it
  were ready.
- Read `roadmap.md` **in full** — do not stop at grepping for this item's
  id/branch/title. A roadmap routinely records decisions, conventions, and
  design rationale in sections that don't name every item individually
  (e.g. "Finalized design decisions", "Decisions locked in", a track's own
  design notes, a prior review round's resolution) — those decisions bind
  this item just as much as a direct mention would, and missing one means
  either proposing a plan that contradicts an already-settled call or
  asking the user something they've already answered. After the full read,
  also grep for the item's id/branch/title specifically, to catch any
  focused mention a full read might skim past. If `roadmap.md` is large
  enough that a full read is genuinely impractical, say so explicitly and
  name which sections you read in full versus grepped — don't silently
  read only part of it and present the plan as if it were comprehensive.

## 3. Gather sibling context from the codebase

For other items in the **same track** that are already `done` (merged),
read what they actually changed — `mcp__github__pull_request_read` for the
diff/description, or `mcp__github__get_file_contents` for the merged
result — to learn the real pattern this item should follow, rather than
inventing a shape from roadmap prose alone. Note file layout, testing
conventions, and any review-driven design decisions recorded in those PRs'
descriptions that this item should also honor (a later sibling in a stack
often encodes a correction the reviewer made on an earlier one).

If the item's own branch already exists (partial work, e.g. from a false
start), read what's actually there via `mcp__github__get_file_contents` or
a local `git fetch` + `git show` before proposing anything — the plan must
build on real state, not restate a fresh start over existing work.

## 4. Cross-check the standing conventions

Read `roadmap.md`'s standing-conventions section (however it's titled in
this plan) and this repository's own `AGENTS.md`. Every step in the
proposed plan must honor both — SOLID, TDD, no abbreviations, dataclasses,
docstring conventions, whatever the repo's own rules are — not just what
the item's own `notes` happen to mention.

## 5. Propose the plan — plan mode, no code

Before drafting the plan or raising any open question with the user, check
whether the question is already answered: re-read the relevant part of
`roadmap.md`, the item's own `notes`, and any cited sibling PR — a design
call, a naming convention, or a scope boundary is very often already
decided somewhere in that material. Only surface something as an open
question if, after that check, it's genuinely still unresolved; asking the
user something the roadmap already answered means the read wasn't thorough
enough. If you do ask, say what you checked and why it still looks open, so
the user can correct you quickly with a pointer if you missed it.

Enter plan mode and present, via `ExitPlanMode`, a concrete implementation
plan: what changes, in which files, in what order, and how each part will
be verified (tests first, per TDD). Cite where each part of the plan came
from (the item's own notes, a specific sibling PR, a `roadmap.md` section)
so the user can sanity-check it against the source instead of just trusting
it. Flag explicitly, never silently paper over:

- Any dependency that isn't actually ready to build on yet (step 2).
- Any conflict between the item's `notes` and what a sibling PR or
  `roadmap.md` actually says.
- Anything the gathered context left genuinely unresolved after the check
  above — say so rather than filling the gap with an assumption.

Do not touch git, create a branch, or write any code in this step — its only
output is the plan itself. Everything below happens after the user approves
it, never before.

## 6. Bootstrap the item — before implementing, not after

The moment a plan is approved, the branch, the draft pull request, the
item's `branch`/`session`/`pull_request_number` fields and its roadmap
section are all derivable, and none of them depends on a line of the
implementation. Doing them at the end instead means the manifest says
`not_started` with no branch for the entire length of the work, which every
dashboard, kickoff and resolve run downstream reads as truth.

So run this first, before the first edit:

Create the branch and its draft pull request yourself, then hand the number
over:

```bash
git checkout -b <branch> <base-branch>
git commit --allow-empty -m "Bootstrap <item-id>"
git push -u origin <branch>
# then create the draft pull request with your GitHub tool, and:
source .claude/hooks/resolve-personal-notes-config.sh
python3 "${PLAN_ITEM_BOOTSTRAP_SCRIPT}" open \
    --plan <plan-id> --item <item-id> \
    --branch <branch> --base <base-branch> \
    --session <this session's url> \
    --pull-request-number <number>
python3 "${PLAN_ITEM_BOOTSTRAP_SCRIPT}" record \
    --plan <plan-id> --item <item-id> \
    --status in_progress --roadmap-section <file>
```

`open` before `record`: the pull request number does not exist until the pull
request does. `open` writes the branch, session and pull request number onto
the item and flips it to `in_progress`; `record` appends the approved plan to
`roadmap.md`. Both print a one-line JSON report led by `status` and
`exit_code`.

**Why a session creates the pull request rather than the script.** The script
can create one — with `--pull-request-title`/`--pull-request-body` instead of
`--pull-request-number`, verified live — but a pull request it creates is
attributed to the app its requests are proxied through rather than to the
person whose work it is, the same authorship problem `AGENTS.md` rules out for
commits. Creating it yourself keeps your identity on it. The creating path is
there for an unattended run whose credential is a real one; if you use it,
`open` publishes the branch too, so the three git commands above are yours to
skip.

The branch name and the base branch are this skill's judgment, not the
script's: the base comes from step 2's dependency readiness, and the branch
from whatever this session is designated to develop on.

**Write the approved plan down in both places it belongs**, rather than
leaving it only in the conversation that produced it:

- **`roadmap.md`**, via `record`'s `--roadmap-section` — the durable record of
  what was decided and *why*, including any assumption or open question the
  plan carries. Not a restatement of the diff.
- **The PR-progress note** — the plan, what is done, and what is next, kept
  current as the work goes. Write it between `CLAUDE.local.md`'s
  `BEGIN-PR-PROGRESS`/`END-PR-PROGRESS` markers and run
  `.claude/hooks/save-pr-progress.sh` to push it. Do this as soon as the
  branch exists, not at the end: a note written afterwards is a summary, and
  the point of it is that another session can pick the work up mid-flight.

Then republish the dashboard yourself:

```
/plan-dashboard <plan-id>
```

Both operations end here rather than doing it, because only a live session
can call the `Artifact` tool — the script's report hands the command back
rather than pretending it ran. Do not skip it: a published dashboard that is
older than the manifest behind it is the exact staleness this step exists to
close.
