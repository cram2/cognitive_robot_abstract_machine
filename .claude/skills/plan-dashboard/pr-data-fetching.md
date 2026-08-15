# Fetching live pull request data into pr_data.json

The canonical procedure for gathering the `pr_data.json` shape
`build_dashboard.py`, `sync_manifest_status.py`, and
`check_dependency_readiness.py` all consume - referenced by
`plan-dashboard/SKILL.md` step 2 and `dependency-readiness.md` instead of
each restating it. See any of those three scripts' own `--help` / module
docstring for the exact JSON shape expected.

Source the shared config script first if you haven't already this session -
it defines the tool-name constants referenced below:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
```

For each distinct repository referenced (`items[].repository` if set, else
the plan's `default_repository`), fetch pull request state **once, in
bulk**, rather than one API call per item - with dozens of items per plan
this matters:

1. `${GITHUB_LIST_PULL_REQUESTS_TOOL}` with `state: "all"`, `perPage: 100`,
   paginating (`page`) until a page comes back short of 100.
2. For any pull request number not covered by that result set (older than
   the pagination window), fall back to `${GITHUB_PULL_REQUEST_READ_TOOL}`
   with `method: "get"` for that specific `pullNumber`.

## Fields

Every entry needs `state`, `draft`, `merged_at` and `labels`. If you narrow
the response with `fields`, that is the minimum set to ask for - narrowing
it further is what breaks this data, not what makes it cheaper.

`merged_at` is the only thing that distinguishes a merged pull request from
one closed unmerged: `state` is `"closed"` for both. Dropping it turns every
merged pull request into a phantom "marked done, but pull request #N was
closed without merging" drift flag. A closed entry built without it is
rejected outright (`MissingMergeTimestampError`) rather than silently read as
unmerged, so record `merged_at` for closed pull requests explicitly, `null`
included.

`${GITHUB_LIST_PULL_REQUESTS_TOOL}` omits null-valued fields from its
response entirely, so a genuinely unmerged pull request comes back with no
`merged_at` key at all - indistinguishable, in its output, from a field you
never asked for. Write the `null` in yourself for those; transcribing the
response as-is is what the rejection above catches.

`labels` carries two things this codebase reads. A pull request merged
out-of-band never gets `merged_at` set, and this repo's convention is to add
a `"merged"` label by hand in that case - see `build_dashboard.py`'s
`PullRequestLabel`/`was_merged`. That label is a fallback for a real merge
GitHub never recorded, not a substitute for `merged_at` - only pull requests
that happen to carry it survive a fetch that dropped the timestamp. A
`"bug"` label separately marks the item as a bug fix wherever it already
appears in the sidebar, and is what its "Bug fixes only" filter keeps - see
`Item.is_bug_fix`.
