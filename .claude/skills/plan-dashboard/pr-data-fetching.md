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

Include `labels` even though most callers never look at it: a pull request
merged out-of-band never gets `merged_at` set, and this repo's convention
is to add a `"merged"` label by hand in that case - see
`build_dashboard.py`'s `PullRequestLabel`/`was_merged`.
