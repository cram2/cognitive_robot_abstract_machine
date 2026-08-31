# First-time setup

Three steps, once per fork. They give you personal notes that follow you across
sessions, per-pull-request progress tracking, and multi-PR plan dashboards.

Details of everything below: [`hooks/README.md`](hooks/README.md).

## 1. Run the setup

In any Claude Code session on this repository:

```
/setup-personal-notes
```

It creates your notes branch, records the git identity your commits are authored
with, installs what the dashboards need, and asks only about the decisions that
are genuinely yours. Re-running it on a set-up clone changes nothing.

## 2. Do the steps no script can do

```bash
python3 .claude/hooks/setup_steps.py
```

It prints these three, already filled in for your fork:

| Where | What |
| --- | --- |
| Your fork | Create the `merged`, `bug` and `in-review` labels. |
| Your Claude settings | Give Claude access to the fork on GitHub. |
| Your Claude environment | Paste the `CLAUDE_PERSONAL_NOTES_*` variables — only if you moved a setting off its default, and only for environments that clone fresh each session. |

## 3. Check it worked

```bash
.claude/hooks/check-setup.sh
```

Every row `ok` and you are done. Anything left says what it needs. Every session
start prints the same verdict, so you never have to remember to check.
