---
name: plan-dashboard
description: Publish a live status dashboard Artifact for a multi-PR/multi-session initiative tracked under .claude/personal/plans/<plan-id>/plan.yaml on the personal-notes branch, cross-checked against live GitHub PR/CI/review state. Invoke as "/plan-dashboard <plan-id>" for one plan, or "/plan-dashboard" with no argument to publish the master index of every plan. Use when the user asks to see, refresh, or generate a plan dashboard, or asks "what's the status of <plan>".
allowed-tools: Bash, Read, Write, Artifact, AskUserQuestion, Skill
---

# Plan Dashboard

Generic, plan-agnostic tooling — nothing in this file may hardcode a
specific plan's id, branches, or PRs. All plan data lives on the
personal-notes branch (`claude/personal-notes` by default — see
`resolve-personal-notes-config.sh` for how it's resolved) at
`.claude/personal/plans/<plan-id>/plan.yaml` + `roadmap.md`, never on
`main`; this skill only reads it. See `plan-schema.md` (next to this file)
for the full `plan.yaml` schema reference — read it if
anything below is unclear about a field's meaning, rather than guessing.

**Everything deterministic — schema validation, live-state classification,
drift detection, HTML rendering — lives in the committed scripts next to
this file (`build_dashboard.py`, `build_index.py`, `render_common.py`), not
in this document.** This file's job is only the parts a script can't do:
resolving what to fetch, calling the GitHub API, and calling the `Artifact`
tool. If you find yourself re-deriving rendering or validation logic in
prose instead of just running the script, stop — that logic already exists,
read the script instead of reinventing it.

The whole point of this dashboard is to catch **drift between a plan's
manually-maintained `status` and GitHub's actual live state** (a session
forgot to update a note after a PR shipped, a PR merged without anyone
updating the manifest, a PR number that no longer resolves). Never trust
`plan.yaml`'s `status` field alone — always cross-check live, every run.
One direction of drift (merged on GitHub but not marked `done`) gets
corrected automatically, in the manifest itself, every run — see step 2;
everything else stays a flag for a human to interpret, since GitHub's state
alone can't tell you *why* an item is blocked, deferred, or still open.

## 0. Check the setup is in place, and offer it if not

Every step below reads the personal-notes branch and runs scripts that need
`${PLAN_DASHBOARD_REQUIREMENTS_FILE}` installed — neither of which a user who
hasn't done the one-time setup has. Follow
`.claude/skills/setup-personal-notes/prerequisite-check.md` before step 1: run
the check, and if it reports anything missing, offer `/setup-personal-notes`
rather than failing later on a missing branch or an `ImportError`.

## 1. Resolve mode and fetch the plan data

Determine mode from the invocation argument: a `<plan-id>` argument means
single-plan mode; no argument means master-index mode.

Source the shared config script — it resolves the personal-notes
remote/branch precedence (`git config claude.personalNotesRemote` → env var
→ default `origin`; branch likewise defaults to `claude/personal-notes`)
into `NOTES_REMOTE`/`NOTES_BRANCH`, and defines every other script/hook path
this document references below (`BUILD_INDEX_SCRIPT`,
`REFRESH_DASHBOARD_SCRIPT`, `WRITE_PERSONAL_NOTES_FILE_SCRIPT`,
`PLAN_DASHBOARD_REQUIREMENTS_FILE`, ...) — read that file if you need the
precise logic or the full list of what it defines. Every bash block in this
document assumes it's already been sourced once per session:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet
```

Work off `FETCH_HEAD`, not `<remote>/<branch>` — a URL-form remote creates
no tracking ref (same reasoning as every hook script here).

**Single-plan mode:**

```bash
git cat-file -e "FETCH_HEAD:.claude/personal/plans/<plan-id>/plan.yaml" || {
  echo "No such plan. Available plans:"
  git ls-tree -r --name-only FETCH_HEAD | grep -E '^\.claude/personal/plans/[^/]+/plan\.yaml$'
  exit 1
}
git show "FETCH_HEAD:.claude/personal/plans/<plan-id>/plan.yaml" > /tmp/plan.yaml
git show "FETCH_HEAD:.claude/personal/plans/<plan-id>/roadmap.md" > /tmp/roadmap.md  # roadmap.md is always the fixed filename, never configurable
```

**Master-index mode:** enumerate every plan instead of one:

```bash
git ls-tree -r --name-only FETCH_HEAD | grep -E '^\.claude/personal/plans/[^/]+/plan\.yaml$'
```

Load each one the same way as above, into its own scratch path.

Also read the generated URL cache, if present (used in step 3, absent on a
plan's first-ever publish — not an error):

```bash
git show "FETCH_HEAD:${DASHBOARD_URL_CACHE_PATH}" 2>/dev/null
```

If a plan has a `tracking_issue`, resolve its `html_url` (via
`mcp__github__issue_read` `get`, falling back to
`mcp__github__pull_request_read` `get` if that 404s — a repo with Issues
disabled stores a PR number under the same field, see
`plan-schema.md`'s "Proposing structural changes" section). Use the
returned `html_url` verbatim; don't construct `/issues/<n>` yourself, since
guessing the path is exactly the kind of assumption that breaks silently
when the fallback applies.

## 2. Cross-check every item's PR against live GitHub state, sync, then run the script

Follow `pr-data-fetching.md`'s procedure (next to this file) to assemble
`/tmp/pr_data.json` for every pull request referenced by any item in this
plan.

Everything from here on is deterministic - **run `refresh_dashboard.sh`**
rather than reproducing its steps by hand:

```bash
bash "${REFRESH_DASHBOARD_SCRIPT}" \
  --plan-id <plan-id> \
  --plan /tmp/plan.yaml \
  --roadmap /tmp/roadmap.md \
  --pr-data /tmp/pr_data.json \
  --output /tmp/dashboard.html \
  --tracking-url "<html_url from step 1, if any>"   # omit if the plan has no tracking_issue
```

It runs `sync_manifest_status.py` first - `status` is a manually-maintained
field for everything except one direction: once GitHub confirms a PR is
merged, the item is done, and that's not a judgment call - unlike every
other kind of drift (a bad PR number, "done" while still open, closed
unmerged against an active status), there's no ambiguity to leave for a
human. If that finds anything to correct, it pushes the patched
`plan.yaml` back to the personal-notes branch resolved in step 1 (via
`.claude/hooks/write-personal-notes-file.sh`) *before* rendering, so the
dashboard reflects corrected data instead of reporting the same "merged but
not marked done" drift forever; if nothing needed correcting, nothing is
pushed. It then runs `build_dashboard.py`, which validates the manifest
(schema version, unique ids, track/wave/depends_on references — the exact
checks `plan-create` must also satisfy), classifies every item's live state
(`merged` | `open_draft` | `open_ready` | `closed_unmerged` | `not_found`),
computes drift, builds the "ready to start"/"blocker may be cleared" lists
(a blocked item whose dependency becomes ready always lands in "blocker may
be cleared", never "ready to start" — it's still blocked, not fresh),
stacks items by dependency depth (indented, capped at 4 levels, wrapping
with a left-edge arrow past the cap). `done` items are hidden by default
behind a sidebar toggle ("Show done / merged items") — done client-side via
CSS custom properties, so a dependent item's indentation dedents to
whatever it would be if its done dependencies weren't dependencies at all,
not just one level shallower. Every not-`done` item gets a dashboard action
button, worded to match its status — "Start now" (`plan-item-kickoff`) for
not-started once every dependency is ready, "Resolve"/"Resume"/"Reconsider"
(`plan-item-resolve`) for blocked/in-progress/deferred respectively — plus
a model dropdown beside it; a page can't spawn a session itself, so these
are copy-to-clipboard affordances (the dropdown's choice is prepended as a
`/model <id>` line ahead of the skill command), not live triggers. An item
whose PR is open and still a draft also gets a "Review" button linking
straight to the PR on GitHub — this plan's convention keeps every PR in
draft until its own author has reviewed it, so a draft PR is exactly the
population still needing that review, and flipping it to ready for review
*is* the record of having done so. The sidebar's "Ready to review" list
narrows that population to what's actually worth reviewing right now: not
blocked, not deferred, and every dependency (if any) already has its own
open PR too (the dependency need not itself be past review — just open —
so a whole reviewable stack can surface before its base has merged). Any
sidebar entry whose PR carries the `bug` label is marked with a `bug` chip
where it already sits: a bug fix is a property of the work rather than an
action of its own, so it is never grouped or ordered separately. A "Bug
fixes only" checkbox — shown only when some entry is a bug fix — filters the
sidebar down to them, client-side, with each group's shown count swapped in
alongside its total the same way the done-items toggle swaps indent levels.
Renders the final HTML — including `roadmap.md` converted to HTML and
shown in a collapsed `<details>` section, and the tracking-issue link if
one was passed. None of that is this document's job to describe further;
read the scripts if you need the specifics.

If `refresh_dashboard.sh` exits non-zero, the manifest failed validation —
its stderr says exactly what's wrong (which field, which value). Report
that to the user instead of trying to patch around it yourself; a broken
manifest is something they need to know about, not paper over. Requires
PyYAML, Jinja2, and the `markdown` package —
`pip install -r "${PLAN_DASHBOARD_REQUIREMENTS_FILE}"` if any are missing.

On success it prints one merged JSON summary on stdout: `sync_manifest_status.py`'s
own `{"corrected": [...]}` plus `build_dashboard.py`'s status counts, drift
count, and drifted/ready/recheck item titles - keep it, you need it for
step 4's report without re-parsing the HTML you just wrote or re-deriving
what got auto-corrected. In master-index mode, run `refresh_dashboard.sh`
for every plan before building that plan's index entry.

**Master-index mode** instead assembles one summary object per plan (`id`,
`title`, `description`, `done`/`total` counts from each plan's own
`build_dashboard.py` run or a quick count over its `items[]`, and
`dashboard_url` from the URL cache — `null` if never published) into a JSON
list, then:

```bash
python3 "${BUILD_INDEX_SCRIPT}" \
  --plans /tmp/plans.json \
  --output /tmp/index.html
```

## 3. Publish, and keep the URL cache in sync

**Before calling `Artifact`, load the `artifact-design` skill** if you're
publishing for the first time and want to sanity-check the visual design —
skip this on a routine re-publish of an already-designed dashboard.

Look up this plan's (or, in index mode, the index's own — use the fixed key
`_index`) existing URL from the cache you read in step 1.
Call `Artifact` with that as `url:` if found (updates the existing page in
place); omit `url` if this is a first-ever publish (mints a new one).

**Never skip the `url:` when the cache has one.** Publishing without it mints
a second artifact for the same plan, and the duplicate cannot be deleted from
a session — it can only stop being referenced.

Favicon: keep a single stable emoji across every redeploy of the **same**
artifact identity. Suggested: 📋 for a per-plan dashboard, 🗂️ for the master
index — pick your own if these clash with something already published, but
keep whichever you pick fixed for that artifact going forward.

Then record the URL — **never by hand-writing the cache YAML yourself.** Every
bulk refresh in this cache's history did exactly that and wrote a URL naming
no artifact the account had ever published; the next run then passed that dead
URL as `url:`, could not update a page that wasn't there, minted a fresh one,
and left the plan with a duplicate dashboard. Writing this file is
deterministic, so it belongs in the script, not in your recollection of what
`Artifact` just returned.

Call `Artifact` with `action: "list"` and write its rows to a scratch JSON
file as `[{"title": ..., "url": ..., "updated": ...}, ...]` — `updated` is the
date each row already carries, and is what separates two artifacts sharing a
title. Then:

```bash
git show "FETCH_HEAD:${DASHBOARD_URL_CACHE_PATH}" > /tmp/dashboard-urls.yaml
python3 "${RECORD_DASHBOARD_URL_SCRIPT}" \
  --key <plan-id or _index> \
  --expected-title "<the plan's own title: from plan.yaml, or the index page's title>" \
  --listing /tmp/artifact_listing.json \
  --cache /tmp/dashboard-urls.yaml \
  --output /tmp/updated-dashboard-urls.yaml
```

It finds the artifact by title and records *that* URL, so no UUID ever passes
through you. When several artifacts share the title — a duplicate already
exists — it takes the most recently updated one, which is the page that was
just published; `--url` overrides that and must name one of them. It exits
non-zero only when the title matches no artifact at all, meaning nothing was
published under it. Report a non-zero exit instead of working around it.

Push the result back only when it says `"changed": true`, with the same helper
`refresh_dashboard.sh` uses internally for the manifest correction:

```bash
bash "${WRITE_PERSONAL_NOTES_FILE_SCRIPT}" \
  --source /tmp/updated-dashboard-urls.yaml \
  --destination "${DASHBOARD_URL_CACHE_PATH}" \
  --message "Record dashboard URL for <plan-id or _index>"
```

It does its work in a scratch worktree and never touches the current branch
or working tree.

## 4. Report back

Summarize for the user from the JSON summaries step 2 printed: item counts by
status, every item `sync_manifest_status.py` auto-corrected to done (by
name, since that's a manifest edit the user didn't ask for line by line),
every remaining drift flag by name, and every item now ready to review by
name — this is the actionable output, don't bury it under a wall of
"here's the dashboard" text. Give the Artifact link(s).

If you're in single-plan mode and the master index already exists (its URL
is in the cache), mention that it wasn't refreshed automatically and the
user can run `/plan-dashboard` with no argument to update it too — don't do
it unprompted, since that republishes a second, separate page.

## Refreshing after a manifest edit

This skill is the second half of the refresh loop
`.claude/hooks/save-plan.sh` starts: that script pushes an edited
`plan.yaml`/`roadmap.md` and regenerates the branch reverse index, but it
cannot call the `Artifact` tool itself (only a live Claude session can) — it
prints a reminder to run `/plan-dashboard <plan-id>` afterward, which is
this skill.
