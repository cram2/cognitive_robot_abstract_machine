# Prerequisite check: offering setup instead of failing

Every skill that reads or writes the personal-notes branch — `plan-create`,
`plan-dashboard`, `plan-item-kickoff`, `plan-item-resolve` — depends on a
one-time setup the user may simply not have done yet. Without this check, that
shows up as a confusing mid-task failure (a fetch of a branch that doesn't
exist, a missing `plan.yaml`, an `ImportError` from a missing dependency)
somewhere deep in the skill, long after the user asked for something else
entirely.

Instead: check first, and *offer* to fix it. The procedure is here, once, so
each skill references it in a line rather than restating it.

## The procedure

Run this **before** the skill's own first step:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
bash "${CHECK_SETUP_SCRIPT}" || true
```

**Exit code 0 — set up.** Say nothing about setup at all; carry straight on
with the skill. A user who is already set up must never be asked about it.

**Non-zero — something is missing.** Do not start the skill's real work, and do
not attempt the individual fixes inline: `/setup-personal-notes` exists to do
exactly that, and duplicating a piece of it here is how the two drift apart.

1. Tell the user, in one or two sentences, what is missing and why this skill
   needs it — from the `needs-setup` rows' own `<detail>` text, not a guess.
2. Ask, via `AskUserQuestion`, whether to run the setup now, defaulting to
   yes. Running it is not a decision to make for them: it creates a branch on a
   remote and can install dependencies.
3. **If they accept:** invoke `/setup-personal-notes` via the `Skill` tool, let
   it finish, then re-run the check above. If it now exits 0, continue with the
   skill as though nothing had happened. If it still doesn't, report what
   remains unresolved and stop — do not push on into work that will fail later
   for the same reason.
4. **If they decline:** stop, and say plainly what won't work without it. Do
   not half-run the skill hoping the missing piece isn't needed on this
   particular path.

## Why an offer and not an automatic fix

The check is read-only; the setup is not. It writes git config, creates a
branch on a remote, can push a notes file, and can install packages — each a
reasonable thing to want, none a thing to do to someone's clone because they
happened to type `/plan-dashboard`. Ask once, act on the answer.
