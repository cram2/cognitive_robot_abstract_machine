# Scope decision: is this new work, or a change to work already in flight?

Every plan skill hits this question from a different direction — `plan-create`
when drafting an item, `plan-item-kickoff` before opening a branch,
`plan-item-resolve` when an item is stuck, `add-plan-item` when someone
describes something new to build — and all four need the same answer, applied
the same way. The procedure is here, once, so each skill references it in a
line rather than restating it.

Getting it wrong is not hypothetical. In one stack, three separate branches had
to be folded back into their parents after the fact, and two sessions
independently built the same artifact under two different filenames without
either noticing. In each case the split reflected the order the work was
thought of, not anything about the work.

## The question

Prefer the change. Work is genuinely new only if it still stands on its own
once the work before it lands. If instead it *modifies* what an unlanded item
introduces, it is that item's work, and stacking it is an artifact of authoring
order rather than a real dependency.

## The mechanical check

Compare the files the work touches against the base it ultimately targets:

```bash
git ls-tree <base-branch> -- <paths the work touches>
```

Empty output means those files do not exist on the base yet, so whichever
unlanded item introduces them is a candidate owner of this work.

`${CHECK_SCOPE_OVERLAP_SCRIPT}` runs this against every candidate branch at
once and reports which paths are missing from the base and which unlanded
branches already touch them. Prefer it over running the command by hand — it is
the step most often skipped, and skipping it is how the collisions above
happened.

## It is a trigger to look, not a verdict

Ask what the work would be if the overlapping edits were removed:

- **Something substantial remains** — the edits are ordinary work on top of an
  unlanded parent, and the item is real. Carry on, but say which files overlap
  so the two branches do not build the same thing twice.
- **Nothing remains** — it exists *only* to change what the parent introduces,
  so it is not separate work. Fold it.

Weighing the two parts usually settles it at a glance.

## Compare by purpose, not only by path

Two items that built the same thing will not show up as an overlapping path if
they named it differently. Read what the candidate branches actually changed —
`${CHECK_SCOPE_OVERLAP_SCRIPT}` reports each one's full changed-file list for
exactly this — and compare intent, not just filenames. When it happens, decide
which copy survives *before* either lands; afterwards it is a merge conflict
instead of a choice.

## What splitting anyway costs

The earlier item ships a state nobody should ever run; the later one spends its
review re-explaining the first; and if either is live infrastructure, landing
the earlier one alone regresses it until the later one follows.

Prefer one item with a coherent story over two that only make sense read in
order. Split on *what the work is*, not on when it was thought of.
