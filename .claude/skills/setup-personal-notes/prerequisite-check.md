# Prerequisite check: running setup instead of failing

Every skill that reads or writes the personal-notes branch — `plan-create`,
`plan-dashboard`, `plan-item-kickoff`, `plan-item-resolve`, `add-plan-item` —
depends on a one-time setup the user may simply not have done yet. Without this check, that
shows up as a confusing mid-task failure (a fetch of a branch that doesn't
exist, a missing `plan.yaml`, an `ImportError` from a missing dependency)
somewhere deep in the skill, long after the user asked for something else
entirely.

Instead: check first, and fix it. The procedure is here, once, so each skill
references it in a line rather than restating it.

## The procedure

Run this **before** the skill's own first step:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
bash "${CHECK_SETUP_SCRIPT}" || true
```

**Exit code 0 — set up.** Say nothing about setup at all; carry straight on
with the skill. A user who is already set up must never hear about it.

**Non-zero — something is missing.** Do not start the skill's real work, and do
not attempt the individual fixes inline: `/setup-personal-notes` exists to do
exactly that, and duplicating a piece of it here is how the two drift apart.

1. Tell the user, in one or two sentences, what is missing and that setup is
   running now — from the `needs-setup` rows' own `<detail>` text, not a guess.
   This is a statement, not a question: never ask whether to run the setup.
2. Invoke `/setup-personal-notes` via the `Skill` tool and let it finish. It
   asks its own questions for the decisions that are genuinely the user's —
   which repository their notes live in, what goes in them, whether to create
   labels — and does the mechanical parts without asking.
3. Re-run the check above. If it now exits 0, continue with the skill as though
   nothing had happened. If it still doesn't, report what remains unresolved and
   stop — do not push on into work that will fail later for the same reason.

## Why this runs without being offered first

A user who invokes a planning skill has already said what they want, and the
setup is the only route to it: there is no useful answer to "shall I set this
up?" other than yes, and asking it turns every first invocation into two turns.
So the yes/no gate is gone.

What survives is narrower and stays inside `/setup-personal-notes`: the choices
that have more than one reasonable answer — which remote holds the notes, what
the notes start out saying, whether to create labels in a repository — are still
asked there, because getting one of those wrong writes to a place the user
didn't choose. Running the setup is not that kind of decision; picking its
destination is.
