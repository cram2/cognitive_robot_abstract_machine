# `plan.yaml` schema reference

The full field-by-field reference for a plan manifest: a generalized
replacement for one-off master-roadmap docs when tracking a
multi-PR/multi-session initiative. A structured `plan.yaml` drives
generation; a sibling `roadmap.md` carries the narrative ("why", history,
design decisions) that doesn't belong in structured data.

This document lives on `main`, alongside the tooling that reads it — it is
generic, plan-agnostic reference material, so every clone has it with no
setup. The **plan data it describes** is the opposite: it is never merged
into `main`, living only on the personal-notes branch (`claude/personal-notes`
by default — see [`resolve-personal-notes-config.sh`](../../hooks/resolve-personal-notes-config.sh)
for how that's resolved), exactly like `cram-notes.md` and `pr-progress/*.md`.

New to the system and want to see it end to end before reading the schema?
Start with [`example-walkthrough.md`](./example-walkthrough.md), a short
worked example from a plan-mode idea to a published dashboard. A complete,
schema-conformant manifest you can read alongside this reference is
committed at [`example/plan.yaml`](./example/plan.yaml), with its narrative
half at [`example/roadmap.md`](./example/roadmap.md).

## Layout

```
.claude/personal/plans/
  _generated/
    branch-index.tsv          (generated - do not hand-edit, see below)
  <plan-id>/
    plan.yaml
    roadmap.md
```

`<plan-id>` is a short kebab-case slug — it's the directory name, the `id`
field inside `plan.yaml`, and the key the generated reverse index uses.

## `plan.yaml` schema (schema_version: 1)

```yaml
schema_version: 1
id: <plan-id>                  # matches the directory name
title: "Human-readable title"
description: >
  One short paragraph. Long-form narrative goes in roadmap.md, not here.
default_repository: <owner>/<repo>   # used by every item unless it sets its own `repository`
                                       # (roadmap.md is a fixed sibling filename, not configurable)
tracking_issue: <int, optional> # see "Proposing structural changes" below

waves:                         # ordered phases, purely organizational
  - id: <wave-id>
    name: "Human-readable name"
    description: "optional"

tracks:                        # a parallel line of work within a wave
  - id: <track-id>
    name: "Human-readable name"
    wave: <wave-id>
    description: "optional"

items:                         # the actual trackable units of work - flat,
                                # not nested under wave/track, so any item can
                                # depend_on any other regardless of nesting
  - id: <item-id>               # defaults to `branch` if omitted; must be
                                 # unique. Prefer the real branch name unless
                                 # it's an auto-generated session slug, in
                                 # which case a readable id is worth the
                                 # divergence.
    title: "What this item does"
    branch: <branch-name>       # the actual git branch, once one exists
    repository: null            # optional, overrides default_repository
    pull_request_number: <int or null>  # the real GitHub PR number, once one exists
    track: <track-id>
    depends_on: [<item-id>, ...]  # structural/stacking dependency, by item id
    status: not_started | in_progress | blocked | deferred | done
    session: <url or omitted>
    notes: "short, freeform"
    blockers: ["freeform reasons", ...]   # optional, defaults to []
```

Validation is not documentation-only: `build_dashboard.py` enforces this
schema on every `/plan-dashboard` run, and `/plan-create` drafts against the
same checks. If this document and that script ever disagree, the script is
the authority.

### Why `status` is deliberately thin

`status` is only ever the session's own manual planning assessment. It
never encodes draft/ready-for-review/merged/closed/CI-green/mergeable —
those are fetched live from GitHub every time `/plan-dashboard` runs, so
they can never go stale in the manifest. The dashboard shows both side by
side and flags any item where the two disagree (e.g. `status: in_progress`
on a PR that's already merged) — this is the actual mechanism that replaces
"a session has to notice a note is stale by accident."

**One exception, corrected automatically.** Most drift needs a person to
interpret it — GitHub's state alone can't say whether a still-open PR means
"blocked", "deferred", or just "in progress", and a mismatched PR number
usually means the manifest was mistyped, not that GitHub is wrong. But
"GitHub confirms this PR is merged" has exactly one correct manifest state:
`done`. There's nothing to interpret, so `/plan-dashboard` doesn't just flag
it — `sync_manifest_status.py` corrects `status` to `done` for exactly that
case, in `plan.yaml` itself, every run, before rendering. Every other kind
of drift is still left as a flag for a human.

### Why items are flat, not nested under wave/track

A track can span or reprioritize across waves, and `depends_on` needs to
reference any other item directly by id. Nesting would either duplicate
items across waves or force awkward cross-links. Tag each item with the
`track` it belongs to instead; look up that track's `wave` for grouping in
the dashboard.

## The generated reverse index

`.claude/personal/plans/_generated/branch-index.tsv` maps every item's
`branch` (across every plan) to that plan's `id`. It is regenerated by
[`save-plan.sh`](../../hooks/save-plan.sh) in the same commit as any
`plan.yaml`/`roadmap.md` push, by scanning every `plans/*/plan.yaml` — never
hand-maintained, so it cannot drift out of sync with the manifests it's
derived from.

`session-start.sh` reads it to auto-load the parent plan's `plan.yaml` +
`roadmap.md` into `CLAUDE.local.md` whenever the checked-out branch appears
in some plan's `items[]`, exactly like the existing per-branch PR-progress
lookup already does for `pr-progress/<branch>.md`.

## Creating a new plan

[`plan-create/SKILL.md`](../plan-create/SKILL.md) (`/plan-create <plan-id>`)
automates this: it gathers the plan's scope (an existing freeform roadmap doc
to migrate, named branches/PRs to cross-check live, or plain conversation),
drafts a schema-conformant `plan.yaml`/`roadmap.md`, validates it against the
same checks `plan-dashboard` runs, asks before assuming anything that's a real
structural judgment call, then runs `save-plan.sh` and `/plan-dashboard`
itself.

## Editing an existing plan

Edit `plans/<plan-id>/plan.yaml` and `roadmap.md` directly (in
`CLAUDE.local.md` if a session already pulled the plan in, or by fetching the
personal-notes branch otherwise), then run

```bash
"$CLAUDE_PROJECT_DIR/.claude/hooks/save-plan.sh" <plan-id>
```

to push both files and regenerate the reverse index in one commit. This
pushes data only — it does not publish the dashboard Artifact itself (that
requires a live Claude session, since only a session can call the
`Artifact` tool). `save-plan.sh` prints a reminder to run
`/plan-dashboard <plan-id>` afterward to actually republish it.

## Proposing structural changes (broadcast via a tracking issue)

Editing `status`, `notes`, or `blockers` on an item you're actively working is
a normal edit — do it directly, as above. **Structural** changes (adding a
wave, deferring a track, splitting an item, reprioritizing) are different:
they're judgment calls about the whole plan, not just the one item a session
happens to be sitting on — but there is no designated steward gatekeeping
them. Any session can make one directly, but only after asking the user in the
session itself (e.g. via `AskUserQuestion`) — a structural change is the
user's call, not something to infer and apply unilaterally just because
editing the manifest directly is technically allowed.

Once confirmed, if a plan has a `tracking_issue` (its number, in the
repository named by `default_repository`), the session **also comments on that
issue describing the change**, in addition to editing `plan.yaml` directly —
it's a coordination mailbox, not a work item, and is subscribable exactly like
a PR (GitHub issue-comment subscription works identically whether the number
is an issue or a PR). This is not a proposal awaiting approval from a
gatekeeping session: the user reviews structural changes there themselves, and
the comment is the shared record every other session working the plan can
check. `/plan-create` creates the tracking issue (titled
`[plan-tracking] <plan-id>`) when bootstrapping a new plan and records its
number as `tracking_issue` in `plan.yaml`.

**Real-time awareness for sessions actively working an item.** A session
working an item in a plan that has a `tracking_issue` should also subscribe to
that issue (in addition to its own item's PR) — not just the session making a
structural change. Since every structural change is posted there, subscribing
turns the tracking issue into a broadcast channel: a change lands in every
actively subscribed session's conversation as it happens, not only picked up
by `session-start.sh`'s auto-discovery on that session's *next* fresh start.
`session-start.sh`'s written header reminds a session of this when it
auto-discovers the plan.

**Fallback when a repo has Issues disabled**: some repos disable Issues
entirely — GitHub returns a `410` on creation attempts. When that happens,
`/plan-create` falls back to an empty-commit, permanently-draft **tracking
PR** instead (same subscribable-mailbox mechanism, just a PR instead of an
issue) and still records its number under the same `tracking_issue` field —
the field name describes the *role* (a tracking mailbox), not literally "must
be a GitHub Issue object." Whoever reads it should check which kind it
actually is (an issue-vs-PR read call distinguishes them) before building a
link, rather than assuming.

If a plan has no `tracking_issue` set, there's no mailbox yet — structural
edits go straight to `plan.yaml` as before (single-session plans, or ones
predating this convention, don't need one).

## Generating a dashboard

See [`plan-dashboard/SKILL.md`](./SKILL.md). Invoke it as
`/plan-dashboard <plan-id>` for one plan's dashboard, or `/plan-dashboard`
with no argument for the master index listing every plan found on the
personal-notes branch.
