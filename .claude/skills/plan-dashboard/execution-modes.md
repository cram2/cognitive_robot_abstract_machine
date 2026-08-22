# Deciding how an item gets worked: auto mode, plan mode, or asking

Shared by `plan-item-kickoff` and `plan-item-resolve`. Both gather an item's
context and then have to answer the same question — hand the user a plan to
approve, or go ahead and do the work — so the answer lives here once instead
of being restated in each skill.

## The three modes

- **`plan`** — present the plan via `ExitPlanMode` and stop until the user
  approves it. Nothing is created before then.
- **`auto`** — draft the same plan, record it, and implement it without
  asking. The planning phase still happens and is still written down; what it
  stops doing is blocking. The user reviews the finished draft pull request.
  **The default**, because by the time the mode applies the skill has already
  read the item's recorded state and the progress on its plan and pull
  request, and the escalation rule below sends anything that genuinely
  changes the settled plan back as a question anyway.
- **`ask`** — put the choice between the other two to the user. Pin this to
  be asked every time.

## Resolving which one is in force

Run this after gathering the item's context, not before — in `ask` mode the
question is only worth putting once you can say something useful about the
item:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
python3 "${PLAN_ITEM_MODE_SCRIPT}" resolve --skill <kickoff|resolve>
```

It prints one JSON object: `mode` (`ask` | `auto` | `plan`), `source`
(`invocation_argument` | `personal_setting` | `committed_default`), and the
`setting_key` and `personal_setting_path` behind it. Precedence is invocation
argument > personal setting > committed default, so pass `--requested <mode>`
when the user named a mode in the invocation itself.

Do not infer the mode from anything else — not from how well-specified the
item looks, not from how the user phrased the request. The whole point of the
setting is that its answer does not depend on the run's own judgement.

To pin a mode, `/plan-item-mode <mode> [kickoff|resolve|both]` — or the script
it calls, which takes `--skill` once per skill so pinning both is one push:

```bash
python3 "${PLAN_ITEM_MODE_SCRIPT}" set --skill kickoff --skill resolve --mode <mode>
```

This writes the personal-notes branch, so don't run it unprompted. Do run it
when the user's answer carries a standing preference rather than a one-off
choice — "always", "from now on", "stop asking me" — since asking them again
next time is exactly what they just said not to do.

## When the mode is `ask`

Put it with `AskUserQuestion`, one question, two options in this order:

- **Implement now** — "I have what I need; go straight to the implementation
  and leave you a draft PR to review."
- **Plan first** — "Draft a plan and stop for your approval before touching
  anything."

State a recommendation and *why*, drawn from the material already gathered —
whether the item's `notes` and `roadmap.md` section settle the design calls,
whether every dependency came back ready, whether a sibling PR established the
pattern to follow, whether anything is genuinely still open. A recommendation
with no reason attached is worth less than no recommendation, because the user
cannot check it.

Apply the same discipline the skills already apply to open questions: if the
gathered material answers something, it is not an open question, and it is not
a reason to recommend planning first.

## What `auto` mode still obliges

Autonomy is about not waiting for approval, not about skipping the record.

- **Write the plan down before implementing it** — the roadmap section and the
  PR-progress note, exactly as an approved plan would have been. A plan that
  exists only in the session that ran is what makes the work unreviewable.
- **Keep the pull request a draft**, and keep its description matching what the
  work actually does as it changes.
- **Record decisions as you make them**, in the PR-progress note and the
  description, so the review reads the reasoning rather than reconstructing it.
- **Run the multi-PR-scope judgement anyway.** `cram-notes.md`'s "Plan-mode
  approval → persistent plans" rule fires on plan-mode approval, which `auto`
  never reaches — so make the same call at the moment the plan is settled:
  does this span more than one pull request or session, and does it therefore
  need plan items of its own?

## When `auto` mode still stops and asks

Stop and ask via `AskUserQuestion` when a decision:

- changes the item's scope or contract as recorded in `plan.yaml` or
  `roadmap.md`;
- is not easily revertible; or
- departs from the settled plan in a way the item's own description would not
  lead a reviewer to expect.

Anything below that bar is yours to decide: decide it, record it, and carry
on. Asking about a minor, revertible call defeats the mode the user chose —
but so does silently making a structural one because the mode said not to ask.
