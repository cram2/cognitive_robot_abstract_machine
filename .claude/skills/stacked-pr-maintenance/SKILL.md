---
name: stacked-pr-maintenance
description: Run one maintenance pass over a stacked-PR fork-staging workflow - reparent any pull request whose base has landed, restack branches whose parent moved, and promote every approved unblocked branch to the upstream review queue. Invoke as "/stacked-pr-maintenance [fork=<owner/repo>] [upstream=<owner/repo>] [--non-interactive]". Use when asked to run a stack maintenance pass, restack the stack, promote ready branches, or clean up after a branch has landed upstream, and when a scheduled routine hands this document its values.
allowed-tools: Bash, Read, Grep, AskUserQuestion, mcp__github__pull_request_read, mcp__github__list_pull_requests, mcp__github__update_pull_request, mcp__github__issue_write, mcp__github__add_issue_comment
---

# Stacked-PR maintenance

You maintain a stacked-PR fork-staging workflow. The **fork** holds the full stack; the **upstream**
is the slow review queue. GitHub is the source of truth:

- a fork pull request's **base branch is its parent** in the stack;
- a fork pull request stays a **draft until its author has reviewed it themselves**, so taking it out
  of draft is the author's sign-off and the only signal that it may be promoted - that is all "the
  draft flag is the ready gate" means, and nothing else grants approval;
- the **in-review label** means the branch has been promoted to the upstream;
- a branch that is an **ancestor of the upstream base** has landed.

**Your job, and only this**: reparent any pull request whose base has landed, close what has
landed by fast-forwarding, restack branches whose parent moved, promote every approved unblocked
branch, and report what you did. The steps below are that job, in order.

It is *not* your job to do code review, to read, answer, resolve or act on the developer's review
comments, or to make code changes addressing review feedback - that is the developer's own
session's work. Leave review threads untouched.

**You do not write code.** The only file changes you ever make are conflict resolutions while
restacking, and when you resolve one you say so in a comment on that branch's pull request, naming
the files and what you took, so the author can check it. Anything you cannot resolve mechanically is
reported, never fixed.

Do not use the Workflow tool - the multi-agent orchestration tool that fans work out to subagents.
This pass is a short sequence of git and API calls; fanning it out multiplies the chance of two
agents pushing the same branch. Use plain git plus the GitHub MCP server. Never force-push a branch
that has an open upstream pull request unless it carries the `rebase` label.

HARD RULES so you never drift into review work:
- NEVER call `subscribe_pr_activity`, and never stay subscribed - you learn CI by POLLING (steps 3 and 4).
- If a review, review-comment, issue-comment, or any `<github-webhook-activity>` event is ever delivered
  to you, your ONLY valid action is to END THE TURN immediately: do not investigate it, do not draft or
  post a plan, do not reply, do not ask the developer to confirm anything. The one exception is a CI/check
  *status* you were polling for your own restack.
- NEVER enter plan mode or post a "here's my plan" comment. You either perform a mechanical step from the
  steps below or you stop; you never open a discussion.
- LABELS ARE REPLACE, NOT ADD: the GitHub label-write call takes the PR's **entire** new label set - it
  does not add to what's already there. Never compute that set yourself; `stack.py labels` computes it
  from the labels the PR carries now, and its output is the whole list you pass to the write.

## Step 0 - resolve which repositories this runs on

Nothing below names a repository or a remote. Resolve both here, once, and use what you get for the
whole run. Do not inspect, guess at, or rename remotes yourself - a remote's name carries no meaning,
and a wrong guess points every push at the wrong repository.

**a. Make the tooling present rather than assuming it.** Every step shells out to
`.claude/stack/stack.py`, and a failure in a later step lands after an earlier one has already
changed pull requests. If `ls .claude/stack/stack.py` fails, `git fetch` the ref you were told to
resolve this document from and `git checkout <ref> -- .claude/stack/`. Once `.claude/stack/` is on
the default branch this is a no-op on a fresh clone.

**b. Take the fork and the upstream in this order, stopping at the first that answers:**

1. **What you were given.** `fork=<owner/repo>` and `upstream=<owner/repo>` in this skill's arguments
   are authoritative. Pass them straight through as `--fork` / `--upstream` and never second-guess
   them.
2. **What the checkout knows.** Run `python .claude/stack/stack.py configuration` (adding `--fork` /
   `--upstream` for anything you were given). It prints one `field<TAB>value` line per setting -
   `fork_remote`, `fork_repository`, `upstream_remote`, `upstream_repository`, `upstream_base`, the
   label names - deciding which remote is which by the repository each URL names. Exit 0 means use
   it. If an `upstream_setup_command` line appears, run exactly that command and re-run
   `configuration`.
3. **Exit status 4 - the fork could not be identified.** The checkout has no fork remote, or has more
   than one candidate, and there is no safe guess.
   - **If you were invoked with `--non-interactive`** (a scheduled routine always is): stop and
     report what `git remote -v` shows. Asking would break the hard rule against opening a
     discussion.
   - **Otherwise**: ask the developer with `AskUserQuestion`, offering the repositories from
     `git remote -v` as options, then persist the answer so the next run needs no question - write
     `fork_repository = "<owner/repo>"` into `.claude/personal/stack.toml` on the personal-notes
     branch:

     ```bash
     "${CLAUDE_PROJECT_DIR}/.claude/hooks/write-personal-notes-file.sh" \
       --source <the file you prepared> \
       --destination .claude/personal/stack.toml \
       --message "record which repository is my fork"
     ```

     That branch is fetched by every run, so the answer survives the fresh clone a scheduled run
     starts from. Re-run `configuration` and continue.

Any other non-zero exit is a stop-and-report, not something to work around.

## Step 1 - reparent every orphaned child, before anything moves

Do this first, and before step 2. Retargeting a pull request whose base has landed is what lets the
next step finish it, and a child left on a landed base cannot reach the upstream base at all - it is
closed outright the moment that base branch is deleted. The inflated diff such a child shows is a
symptom, not the problem.

**BASE CHANGES GO THROUGH THE GITHUB MCP SERVER.** Retarget a base with the MCP `update_pull_request`
tool and nothing else. The same request issued as a raw `PATCH /repos/{owner}/{repo}/pulls/{number}`
with curl and `GH_TOKEN` is refused with
`403 - Changing a pull request's base branch is not permitted for this session type`, however the
body is formed - the block is on that credential, not on the operation, so retrying it or reshaping
the request cannot get past it. If you see that 403 you
used the wrong client, so switch rather than report it as a stuck reparent. The stacks endpoints
below are the mirror image: they have no MCP tool, so they do need curl.

Run:

```bash
python .claude/stack/stack.py reparents
```

It prints one `branch<TAB>pr<TAB>current base<TAB>target base` line per open pull request whose base
has already landed - decided by git ancestry, so it also covers a base whose own pull request was
*closed* rather than merged, which is absent from the board entirely and which nothing else would
ever look at. Retarget each one with `update_pull_request`.

**NATIVE-STACK MEMBERS.** Changing the base of a pull request that belongs to a GitHub stack fails
with `422 - Cannot change the base branch because the pull request is part of a stack`. This is a
different refusal from the 403 above with a different cure: the 422 is GitHub protecting the stack's
structure, and dissolving the stack clears it. A pull request is a stack member iff its REST JSON
carries a non-null `stack` object when fetched with the header `X-GitHub-Api-Version: 2026-03-10`;
the stacks endpoints are not in the GitHub MCP server, so call them with curl and `GH_TOKEN`, always
with that version header. For exactly those children the reparent becomes:

1. `GET /repos/{owner}/{repo}/stacks` and record the affected stack's full pull-request list, bottom
   to top. Do not proceed without the recorded list - dissolving is destructive and there is no undo.
2. `POST /repos/{owner}/{repo}/stacks/{number}/unstack` (no body) to dissolve it. There is no
   selective removal: this drops every open, draft and closed member, leaving merged ones in place.
3. `update_pull_request` each orphaned child's base, which succeeds once the stack is gone. The child
   keeps its number, its labels and its review thread - never close it and open a replacement, which
   loses all three for a base change that is available to you.
4. Restack normally (step 4's local merge/rebase plus push).
5. Re-create the stack: `POST /repos/{owner}/{repo}/stacks` with `{"pull_requests": [...]}` - the
   recorded list minus landed and closed members, bottom to top - then `GET` it back and confirm
   every member reports the stack.

Do not fast-forward the landed base branch as a way around this. It moves the merge-base so the diff
looks right, but the child still targets a branch about to disappear - and when that base is a
stack's trunk, moving it desynchronises the stack's recorded `base.sha` from its real head. If any
call in this sequence fails or answers with something not described here, stop work on that stack,
leave the rest untouched, and report it: this is a preview API, so never improvise around it.

## Step 2 - update the fork's copy of the upstream base

This is what closes the landed pull requests, so it comes after the reparent above and needs nothing
after it. GitHub marks a pull request **merged** - not merely closed - the moment its head becomes an
ancestor of its base, so fast-forwarding the fork's base branch closes every pull request whose work
has landed, in one operation and with no label to write:

```bash
git fetch <upstream-remote> <upstream-base> \
  && git push <fork-remote> <upstream-remote>/<upstream-base>:<upstream-base>
```

This must be a fast-forward. If GitHub rejects it as non-fast-forward, stop and report - do not
force. Keep that branch a pristine mirror of the upstream trunk: root branches base on it and the
restack merges it into them, so anything added here flows into every branch and then into the
upstream.

`python .claude/stack/stack.py landed` reports which branches this accounted for, for the summary.
Never label or close a pull request whose work has not landed - and you should not need to label or
close anything at all: if a landed pull request is somehow still open afterwards, report it rather
than closing it by hand.

## Step 3 - refresh the derived stack

`git fetch <fork-remote>`, then refresh `.claude/stack/board.json` from the fork's **open** pull
requests (number, head, base, isDraft, labels, statusCheckRollup and body) via the GitHub MCP, and
run `python .claude/stack/stack.py status`. There is no live mode; state comes from `board.json` plus
git.

**CI is the validator - poll it, never subscribe.** When you need a branch's verdict, poll with
`pull_request_read` → `get_check_runs` / `get_status` and read only the success/failure conclusion. A
subscription delivers human review comments and review threads, not just CI, and turns on the
per-event handler that makes you investigate, plan and reply - which is exactly how a maintenance run
turns into review work.

## Pre-flight - before every push, merge or restack, no exceptions

Never move commits from memory, and never judge the move yourself. Ask:

```bash
python .claude/stack/stack.py preflight \
  --action push --source <branch> --destination <branch> --destination-remote <fork-remote>
```

Exit 0 means the move is clear. Exit 5 means it must not be made, and every reason is on stderr,
each tagged with which refusal it is: `not-checked-out`, `mismatched-branch-names`, `not-the-fork`, or
`false-merge` - a push that would make a child branch an ancestor of its own parent, which GitHub
reads as a merged pull request and closes. Fix the cause and ask again; never push past a refusal.

Then say in one sentence what you are integrating and why it belongs on that destination.

Step 2's fast-forward is the one push this cannot check: it deliberately maps one ref onto another
and happens before the board exists, so it exits 3 rather than judging the move. GitHub's own
non-fast-forward rejection is what guards that push instead, which is why step 2 stops rather than
forcing.

## Step 4 - restack and validate

Run `python .claude/stack/stack.py restack-plan` for the bottom-up plan. For each entry whose parent
moved, integrate the parent using its `strategy` (merge is the default and needs no force-push;
rebase force-pushes with lease) **only if the merge is clean**, run pre-flight, then push. CI is the
validator.

**If you resolved any conflict while integrating**, comment on that branch's pull request saying so:
which files conflicted, and what you took for each. A conflict resolution is a change to somebody
else's branch that they did not make, so it is never allowed to be silent.

**Do not block on CI.** After pushing a branch, move on to the next independent branch and keep
restacking and promoting in parallel - never sit idle waiting on a long run. Poll the checks of the
branches you pushed at the start of each pass, and react then.

**A conflict you cannot merge cleanly, or a red check, is not yours to resolve.** You do not debug it
and you do not fix it. Report it to the branch's owner and move on:

1. Find the session: search the fork pull request body for a `https://claude.ai/code/session_...`
   link.
2. Post a comment on the fork pull request, prefixed `🔴 ROUTINE - NEEDS RESOLUTION:`, stating what
   you were doing, what happened (the conflicting files, or the failing check and its conclusion),
   and the ask - that they resolve and push, and you will pick the branch back up once it restacks
   clean. This comment is the only channel available to you; if that session is still subscribed to
   its own pull request, it arrives there as a live event rather than text sitting on GitHub.
3. Label the pull request `needs-resolution` (via `stack.py labels`, so the rest of its labels
   survive) so the state is visible even if no session is listening, and so you never re-attempt the
   same failing restack every run.
4. At the start of every restack pass, fetch each `needs-resolution` branch's `mergeable_state`
   (`pull_request_read` → `get`). GitHub reports `dirty` when the branch has merge conflicts against
   its base; anything else (`clean`, `unstable`, `blocked`, `behind`, `has_hooks`, `unknown`) means
   there are no conflicts, whatever else may be true of it. So: clear the label and restack the
   branch normally unless `mergeable_state` is `dirty`, and skip it only while it is.

Record every branch you report on - the finish summary must list it, since a comment is not
guaranteed to be seen.

Keep restacking and promoting the other branches while CI works through the ones you pushed. Never
disable a check to go green.

## Step 5 - promote

Housekeeping first: remove any `cram2-link-sent` label from a fork pull request that is now
`in-review` or landed - its link has been acted on.

Collect what to promote: `python .claude/stack/stack.py next --porcelain` prints one `name<TAB>pr`
line per branch that is approved (out of draft), whose parent has reached in-review or landed, and
that is not withheld by `needs-resolution`. There is no admission cap and no ordering beyond
dependency order: every such branch promotes in the same run. Skip any already carrying
`cram2-link-sent` when deciding whether to build a *new* link, but still process the others. If it
prints nothing, promote nothing.

For each collected pull request, build the compare-and-create link. Do **not** try to open the
upstream pull request through the API first - the GitHub app has no write access to the upstream, so
that call is a wasted round trip that fails every time:

```bash
python .claude/stack/stack.py promotion-link \
  --branch <branch> --title <title> --body <one paragraph plus a link back to the fork PR>
```

It owns the URL encoding and the length limit, so the prefill cannot be silently lost - keep the body
short anyway, and note that it warns on stderr when it had to shorten one.

Then, for that branch:

- **Put the link in the fork pull request's own description**, under a `## Promote` heading, replacing
  any link already there. The summary is delivered once and then gone; the description persists, so
  this is where the link is still findable a week later.
- Add `cram2-link-sent` so later runs do not rebuild it.
- Do **not** add `in-review`: the upstream pull request is not open until the developer clicks
  Create, and they add the label then.

## Finish

The **top** of the finish summary must list all pending upstream create-links: any built this run,
and any fork pull request still carrying `cram2-link-sent` but not yet `in-review` (re-listed from
prior runs, its link rebuilt with `promotion-link`). This section appears at the top even when
nothing new was built, as long as any are pending - a scheduled run is configured to email its
summary, so the summary *is* the delivery. List each pull request's number, title, branch and
one-click link.

Right after the links, list every branch you **reported on** this run: its number and branch, the
conflicting files or failing check, the session link you addressed (or that the body had none), and a
link to the comment you posted. Then list every pull request whose reparent you could **not**
complete: its number, the base it is stuck on, the base it should have, and which step of the
native-stack sequence stopped you - a stack left dissolved or half-rebuilt needs attention
immediately and nothing else surfaces it. Then summarise what landed, what you restacked (naming any
conflict you resolved), and what you promoted, plus anything you stopped on.
