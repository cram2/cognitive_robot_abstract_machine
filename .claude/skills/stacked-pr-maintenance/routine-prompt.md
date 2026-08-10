# Running the maintenance pass on a schedule

The skill is normally invoked by hand - `/stacked-pr-maintenance` - whenever the stack needs a
pass. To have it run unattended instead, register the prompt below as a scheduled Routine at
claude.ai/code/routines.

Substitute `<FORK_REPOSITORY>` and `<UPSTREAM_REPOSITORY>` with the two `owner/repository`
references before registering. Naming both is what keeps the run non-interactive: the skill's
step 0 only has to ask when nothing has told it which repository is which, and a scheduled run
has nobody to answer.

```text
/stacked-pr-maintenance fork=<FORK_REPOSITORY> upstream=<UPSTREAM_REPOSITORY> --non-interactive

Do not summarise it back to me, do not ask which step to begin with, and do not wait for
confirmation - run it.
```

## Running the same pass by hand

Nothing about the skill is scheduled-only. From any session:

```text
/stacked-pr-maintenance
```

Invoked with no arguments it resolves the repositories from the checkout, and asks - once - if it
cannot. The answer is written to `.claude/personal/stack.toml` on the personal-notes branch, so
later runs, scheduled or not, never ask again.
