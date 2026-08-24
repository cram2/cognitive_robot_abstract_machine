---
name: upstream-reviews
description: Report the review threads a fork branch's upstream pull request has collected, including which are still unresolved, by dispatching the fork's own upstream-reviews Action and reading its job log. Invoke as "/upstream-reviews [<pull-request-number>|<branch>]". Use when the user asks what the upstream reviewers said, whether a promoted branch has outstanding review comments, or to check upstream review state before resolving or resuming work on a branch.
allowed-tools: Bash, Read, mcp__github__actions_run_trigger, mcp__github__actions_list, mcp__github__get_job_logs
---

# Upstream reviews

Reads the review state of a fork branch's pull request on the upstream
repository and reports it. Generic and fork-agnostic — nothing here may
hardcode a repository, owner, or branch name.

**This skill only reads.** It never comments on, reviews, resolves, or
otherwise modifies anything on the upstream. `AGENTS.md` forbids it outright,
and the report is the entire deliverable.

## Why an Action rather than a direct read

Thread resolved-state is exposed only by GitHub's GraphQL API, and GraphQL is
blocked for Claude sessions by the agent proxy's egress policy — for every
repository, including the fork's own. The read therefore happens on the fork's
Actions runner, where `gh` is preinstalled and `GITHUB_TOKEN` authenticates it,
and this session reads the resulting job log, which is plain read-only REST.

## 0. One-time prerequisite

Actions are disabled by default on a freshly created fork. If the dispatch in
step 2 fails because Actions are disabled, say so and point the user at their
fork's Settings → Actions rather than retrying.

`workflow_dispatch` additionally requires the workflow file to exist on the
fork's **default branch** — GitHub will not dispatch a workflow it cannot see
there, even when the file exists on the branch being read. If the dispatch is
rejected for that reason, report it plainly; it means the workflow has not
landed on the fork's default branch yet.

## 1. Resolve the target and the fork

The fork is this checkout's own repository — resolve it rather than assuming:

```bash
python3 .claude/stack/stack.py configuration
```

Take the target from the skill's argument: a bare number is an upstream pull
request number, anything else is a branch name. With no argument, use the
current branch (`git rev-parse --abbrev-ref HEAD`).

## 2. Dispatch the workflow

Call `mcp__github__actions_run_trigger` with `method: "run_workflow"`,
`workflow_id: "upstream-reviews.yml"`, the resolved fork as `owner`/`repo`,
`ref` set to the fork's default branch, and `inputs` carrying either
`pull_request` or `branch` (plus `include_resolved` when the user asked for
resolved threads too).

This dispatch is the only call in the skill that is not a read. It starts a
job and changes no repository content.

## 3. Wait for the run

List runs with `mcp__github__actions_list`, `method: "list_workflow_runs"`,
`resource_id: "upstream-reviews.yml"`, and `workflow_runs_filter.event:
"workflow_dispatch"`. Take the newest run created after the dispatch, and poll
it until its status is `completed`. Runs on this repository have historically
started with no queue delay and the job itself is small, so expect well under a
minute. Do not use a fixed `sleep` loop as a substitute for checking status.

## 4. Read the report

Get the run's jobs with `method: "list_workflow_jobs"`, then call
`mcp__github__get_job_logs` with that `job_id` and `return_content: true`. The
report is the markdown the script printed; the same text is also on the run's
job summary, which the user can read from the GitHub mobile app.

## 5. Present it

Reproduce the report's substance in the session: every unresolved thread with
its author, `file:line`, and comment text. Do not summarize away the comment
bodies — the point of the skill is that the reviewer's actual words reach the
session without the user retyping them.

If the run failed, report the failure and the log's error rather than an empty
result. If the branch has no upstream pull request, the script says so
explicitly — relay that as a clean answer, not an error.
