# Checking whether an item's dependencies are ready to build on

Shared by `plan-item-kickoff` and `plan-item-resolve`'s "gather the item's
own context" steps — both need to answer the exact same question ("is it
actually safe to stack new work on top of item X's dependencies?"), so
this procedure lives here once instead of being restated in each skill.

Source the shared config script if you haven't already this session in
this same bash call — it defines `CHECK_DEPENDENCY_READINESS_SCRIPT`, used
below (this has to stay a literal path: it's the one file that defines
every other constant referenced by name in this system, so nothing "more
shared" exists yet for it to point at):

```bash
source .claude/hooks/resolve-personal-notes-config.sh
```

For the item's `depends_on` list, follow `pr-data-fetching.md`'s procedure
(next to this file) to bulk-fetch every referenced pull request's live
state into `/tmp/pr_data.json`. Then run:

```bash
python3 "${CHECK_DEPENDENCY_READINESS_SCRIPT}" \
  --plan /tmp/plan.yaml \
  --pr-data /tmp/pr_data.json \
  --item <item-id>
```

rather than re-deriving the readiness rule in either skill — it reuses
`build_dashboard.py`'s own `Item.is_ready_to_unblock_dependents()`, so
neither skill can ever silently disagree with the dashboard about what
counts as ready. It prints one JSON list, one entry per `depends_on` entry,
in order: `[{"identifier": ..., "title": ..., "live_state": ...,
"is_ready": <bool>}, ...]`.

A dependency the script reports `"is_ready": false` for (still not-started,
only a draft pull request, or was ready and has since regressed to blocked
or closed unmerged) is a real, common cause of a stall — flag it explicitly
in the proposed plan's assumptions rather than quietly proceeding as if it
were ready. See `plan-schema.md` for why an open, non-draft pull request
already counts as ready even though it hasn't merged yet — this repo's
normal workflow stacks new work on it before it does.
