---
name: plan-item-mode
description: Change how /plan-item-kickoff and /plan-item-resolve start work on a tracked item - implementing on their own (auto), stopping for plan approval first (plan), or asking each time (ask) - by pinning the choice in the personal-notes branch so it holds for every future session. Invoke as "/plan-item-mode <auto|plan|ask> [kickoff|resolve|both]". Use whenever the user states a standing preference about being asked, such as "stop asking me, just implement", "always show me a plan first", "go back to asking each time", or "make auto the default from now on".
allowed-tools: Bash, Read
---

# Plan Item Mode

Sets the execution mode the plan-item skills resolve. Every mechanical part is
`plan_item_mode.py`'s; this skill only reads what the user asked for and calls
it. `${EXECUTION_MODES_DOCUMENT}` is where the modes are defined and where what
each one obliges is stated — read it if the user asks what a mode means rather
than restating it here.

## 1. Work out what was asked for

Two things, both usually plain in the request:

- **The mode** — `auto` (implement without asking), `plan` (stop for approval),
  `ask` (put the choice each time). Map how the user actually phrased it:
  "stop asking, just do it" is `auto`; "always let me approve first" is `plan`;
  "ask me each time" is `ask`.
- **Which skill** — `kickoff`, `resolve`, or both. **Default to both** when the
  user didn't distinguish them, since a standing preference about being asked
  is rarely meant for only one half.

If the phrasing genuinely doesn't settle the mode, ask with
`AskUserQuestion` rather than guessing — this write persists, so a wrong guess
changes every future session until someone notices.

## 2. Set it

One call, however many skills — the settings file is rewritten whole, so
naming both in one run is one commit on the notes branch instead of two:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
python3 "${PLAN_ITEM_MODE_SCRIPT}" set \
    --skill kickoff --skill resolve --mode <mode>
```

Drop whichever `--skill` the user didn't mean. The report names the skills it
pinned, the mode, and the file it wrote.

## 3. Confirm it from the file, not from the write

```bash
python3 "${PLAN_ITEM_MODE_SCRIPT}" resolve --skill kickoff
python3 "${PLAN_ITEM_MODE_SCRIPT}" resolve --skill resolve
```

Report the `mode` and `source` these come back with. `source` is the part worth
showing: `personal_setting` means the pin is in force, and anything else means
the write didn't take effect the way it looks like it did.

Tell the user the setting lives on the personal-notes branch, so it follows
them into every clone and every future session — and that a single run can
still depart from it, since `/plan-item-kickoff` and `/plan-item-resolve` take
a mode argument that outranks the pinned one.
