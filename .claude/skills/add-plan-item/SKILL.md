---
name: add-plan-item
description: Decide where a newly described piece of work belongs - folded into an unlanded item, as a new item in an existing plan, as a new plan of its own, or tracked nowhere - by running the mechanical scope check against live branch and pull request state, then propose the outcome via plan mode without writing any code. Invoke as "/add-plan-item <description of the work>". Use when someone describes something new to build and asks where it goes, or says "add a plan item", "should this be its own PR", or "is this new work or part of <something>".
allowed-tools: Bash, Read, Grep, Glob, AskUserQuestion, Skill, EnterPlanMode, ExitPlanMode, mcp__github__list_pull_requests, mcp__github__pull_request_read, mcp__github__get_file_contents, mcp__github__issue_read, mcp__Claude_Code_Remote__subscribe_pr_activity
---

# Add Plan Item

Generic, plan-agnostic — nothing here may hardcode a specific plan id, item,
or branch. Takes a description of something to build and decides where it
belongs, before any branch exists to regret. **This skill never writes code,
creates a branch, or pushes anything** — its only output is a proposed plan.

There are exactly four outcomes, and step 5 picks one:

1. **Fold** it into an unlanded item or branch that already owns this work.
2. **New item** in a plan that already exists.
3. **New plan** of its own, via `/plan-create`.
4. **No item at all** — work small enough that tracking it costs more than it
   returns.

## 0. Check the setup is in place, and offer it if not

Every plan's manifest lives on the personal-notes branch, which the user may not
have set up yet. Follow
`.claude/skills/setup-personal-notes/prerequisite-check.md` before step 1: run
the check, and if it reports anything missing, offer `/setup-personal-notes`
rather than failing on a branch that isn't there.

## 1. Pin down what the work actually touches

The check in step 3 needs two things the description rarely states outright:

- **The paths** the work would create or modify.
- **The base branch** it ultimately targets (usually the repository's default
  branch, but a plan may target something else).

Derive the paths by reading the repository — find where comparable work already
lives and name the files this would add or change. Only if they are still
genuinely underdetermined after that, ask via `AskUserQuestion`. Never skip
step 3 for want of paths: an unrun check is exactly how work ends up on its own
branch by default.

## 2. Find the candidate homes

```bash
source .claude/hooks/resolve-personal-notes-config.sh
git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet
```

Read **every** plan's `plan.yaml` under `${PLANS_DIR}` off `FETCH_HEAD`, not
just the one the user has in mind — the right home is often a plan they weren't
thinking of. `${PLAN_SCHEMA_DOCUMENT}` is the field reference.

Then fetch live pull request state once, in bulk, per
`${PULL_REQUEST_DATA_FETCHING_DOCUMENT}`.

Candidates are every unlanded item and every open pull request's branch, plus
every plan whose own `description` covers the kind of work being described.

## 3. Run the scope check

The rule is in `${SCOPE_DECISION_DOCUMENT}` — read it rather than re-deriving
it here. Gather its evidence in one call:

```bash
python3 "${CHECK_SCOPE_OVERLAP_SCRIPT}" \
    --base <base-branch> \
    --path <path> [--path <path> ...] \
    --candidate '<item-id-or-title>=<branch>' [--candidate ... ]
```

Non-empty `paths_absent_from_base` means an unlanded branch introduces those
files, and whichever one does is a candidate owner. Non-empty `shared_paths`
names it. Apply the trigger-to-look test from `${SCOPE_DECISION_DOCUMENT}` to
decide what that means; do not treat the output as the verdict by itself.

## 4. Scan for duplicate intent

Two branches building the same thing under different filenames share no path,
so the script cannot flag them. Read each candidate's `changed_paths` and
compare what those branches are *for* against what is being described. A
match here outranks the path check: same purpose is the same work, whatever
the files are called.

## 5. Choose exactly one outcome

- **Fold** — the work has nothing substantial left once the overlapping edits
  are removed, or a candidate is already building the same thing. Name the
  branch it goes on.
- **New item in an existing plan** — it stands on its own and a plan's
  `description` covers it. Name the plan, the track, the wave, and any real
  `depends_on`.
- **New plan** — it stands on its own, spans several pull requests or sessions,
  and no existing plan covers it. Hand off to `/plan-create` via the `Skill`
  tool rather than drafting a manifest here.
- **No item at all** — one self-contained change, done in this session, that
  nothing else depends on. `plan-create`'s own guidance against over-modelling
  applies: a plan that tracks everything is a plan nobody reads.

If two outcomes remain genuinely open after steps 3 and 4, put the choice to
the user with `AskUserQuestion`, giving the evidence for each rather than
asking them to re-derive it.

## 6. Propose the plan — plan mode, no code

Enter plan mode and present, via `ExitPlanMode`, the chosen outcome and what
follows from it. Cite where each part came from — the script's output, a
specific plan's `description`, a candidate branch's diff — so the user can
check it against the source instead of just trusting it.

**If the outcome is a fold**, the proposal must say which branch the work goes
on *and* what happens to any branch already opened for it — reset onto the
parent, or move the commits over and retire the redundant branch. A bare "fold
it" leaves the session to improvise the part most likely to lose work.

**If the outcome changes a plan**, say how the manifest gets updated: edit
`plan.yaml` directly if this session stewards that plan, otherwise propose it on
the plan's `tracking_issue` — `${PLANS_DIR}/README.md`'s "Proposing structural
changes" section governs. Either way the proposal ends with `${SAVE_PLAN_SCRIPT}`
and republishing the dashboard via `/plan-dashboard <plan-id>` in the same turn,
so no published dashboard is older than the manifest behind it.

**If the outcome is a new item**, `${PLAN_ITEM_BOOTSTRAP_SCRIPT}`'s `record`
operation writes its `plan.yaml` entry and `roadmap.md` section and runs the save
for you — `record` only, never `open`, which creates the branch and pull request
this skill does not.

Flag explicitly, never silently paper over:

- A candidate branch that could not be fetched, so its overlap is unknown
  rather than empty.
- Paths that had to be guessed rather than derived in step 1.
- A duplicate-intent match you are unsure about — say what looks the same and
  let the user settle it.

Do not touch git, create a branch, or write any code in this skill — its only
output is the plan itself.
