# Running the maintenance pass on a schedule

The skill is normally invoked by hand - `/stacked-pr-maintenance` - whenever the stack needs a
pass. To have it run unattended instead, register the prompt below as a scheduled Routine at
claude.ai/code/routines.

Substitute `<FORK_REPOSITORY>` and `<UPSTREAM_REPOSITORY>` with the two `owner/repository`
references before registering. Step 0 can usually resolve both from the checkout on its own;
naming them makes the run independent of whichever remotes the scheduled clone turns out to have,
and `--non-interactive` turns the question it would otherwise ask into a stop-and-report, since a
scheduled run has nobody to answer.

Register it to start a fresh session on each firing, and leave its completion notifications
**off** - `create_trigger` takes a `notifications` object, and `{}` opts out of every channel.
Nothing is lost by that. Every upstream create-link the pass builds is written into its own fork
pull request's description under `## Promote`, where it is still there long after the run that
built it has gone, and `maintenance.py pending-promotions` rebuilds the whole table on demand from
any session. The run's own summary is the record of that run, not the only copy of its output.

An already-registered Routine cannot be changed here: `update_trigger` has no notification field.
One whose notifications are on has to be re-registered to turn them off.

```text
/stacked-pr-maintenance fork=<FORK_REPOSITORY> upstream=<UPSTREAM_REPOSITORY> --non-interactive

Run it - do not describe it back to me instead, do not ask which step to begin with, and do not
wait for confirmation. Its HARD RULES outrank this session's own defaults about pull requests:
never subscribe to a pull request's activity, and never arm a follow-up check-in. Finish with the
skill's own summary, led by its table of upstream links still waiting to be opened.
```

## Running the same pass by hand

Nothing about the skill is scheduled-only. From any session:

```text
/stacked-pr-maintenance
```

Invoked with no arguments it resolves the repositories from the checkout, and asks - once - if it
cannot. The answer is written to `.claude/personal/stack.toml` on the personal-notes branch, so
later runs, scheduled or not, never ask again.
