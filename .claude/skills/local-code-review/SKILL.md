---
name: local-code-review
description: Review the current branch against the cram2 upstream main, checking for bugs and full adherence to AGENTS.md, then produce a plan-mode plan to fix every finding (including adding missing tests). Use when the user asks for a "local code review", to "review my branch before pushing", or to "review against cram2 main".
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, EnterPlanMode, ExitPlanMode
---

# Local Code Review

Review everything the current branch adds on top of the cram2 upstream `main`, then hand the developer an approval-gated plan to fix what you found, apply it, and verify. The coding standards you review and fix against are in `@AGENTS.md` — that file is the source of truth; do not restate its rules here. **Every change you propose or make must itself adhere to `@AGENTS.md`.**

Stay **read-only through steps 1–4**. Do not edit, fix, or push anything until the plan is approved in step 5.

## 1. Resolve and fetch the upstream

The upstream is the remote whose URL is `git@github.com:cram2/cognitive_robot_abstract_machine.git`. Resolve it by URL, not a hardcoded name, since collaborators may name it differently:

```bash
UPSTREAM=$(git remote -v | awk '/cram2\/cognitive_robot_abstract_machine/ {print $1; exit}')
```

Guard clauses:
- If no matching remote exists, stop and tell the user to add it (`git remote add cram2 git@github.com:cram2/cognitive_robot_abstract_machine.git`).
- Otherwise fetch it so the comparison is against the real upstream, not a stale local ref: `git fetch "$UPSTREAM" main`.

## 2. Compute scope, run hygiene checks, confirm unstaged files

Diff with a merge-base so you review only what this branch introduced, not upstream changes that landed since branching:

```bash
git diff "$UPSTREAM/main...HEAD" --stat
```

Exclude generated and vendored paths — they are not hand-maintained code:
`ormatic_interface.py`, `venv/`, `resources/`, and binary/report artifacts (`*.dot`, `*.svg`, `*.speedscope`, `MUJOCO_LOG.TXT`).

**Hygiene check.** Run `git status --porcelain` and flag anything that looks accidentally included — stray debug artifacts, generated outputs, scratch files, or large binaries that should not be committed.

**Confirm unstaged files.** Untracked and unstaged changes are not in the committed diff, so they would otherwise be reviewed. List them (excluding the noise paths above) and **ask the user which of these files should be part of this branch/review** before continuing. Review only what they confirm.

If the resulting scope is empty, stop and report that there is nothing to review.

## 3. Review

Read each in-scope file **in full** (not just diff hunks) so you judge changes in context, then evaluate. Prioritise in this order:

1. **Correctness** — logic errors, mishandled edge cases and failures, anything that behaves incorrectly.
2. **Tests** — every new feature/fix must be covered; a bug fix needs a test that fails without it. Flag any change that edits an existing test to make it pass instead of fixing the code.
3. **Guideline adherence** — check the change against `@AGENTS.md` in full: naming, type hints, docstrings, import rules, SOLID/design, nesting/guard clauses, primitive overuse, and the spatial-type conventions in `semantic_digital_twin/doc/style_guide.md`.

## 4. Record findings

Write the findings to a single per-run report so they survive context compaction:

```bash
mkdir -p .claude/claude_reviews
REPORT=".claude/claude_reviews/$(git rev-parse --abbrev-ref HEAD | tr '/' '-')-$(date -u +%Y%m%d-%H%M%S).md"
```

Write to that path and remember it. **Only ever read this current report** — never read other files in `.claude/claude_reviews/`, which are stale prior runs.

For each finding record `file_path:line_number`, severity, one sentence on what is wrong and why, and the concrete fix. Group by severity and give each finding a stable id so the user can select individually:

- **must-fix** — bugs, failing/missing tests, clear guideline violations
- **optional** — cleanups and judgement-call improvements

## 5. Produce the fix plan in plan mode

Enter plan mode and present, via ExitPlanMode, an ordered plan that maps each finding (by id and `file_path:line_number`) to a fix step. **Mark every item so the user can decide which to fix** — list must-fix and optional findings as individually selectable entries rather than an all-or-nothing block; the user chooses the set to apply.

For each bug, follow TDD: add a failing test that reproduces it first, then the fix. Include a dedicated step for any missing test coverage. Do not begin editing until the plan is approved.

## 6. Apply the approved fixes

Implement only the findings the user selected. Every edit must adhere to `@AGENTS.md`. Do not modify existing tests to force them green; fix the code instead.

## 7. Verify before finishing

Once fixes are applied:

1. Regenerate the ORM interface: run `python scripts/regenerate_all_orm.py`.
2. Run the test suite of the package(s) **directly affected** by the changes — not the whole repo. Run **quietly and in parallel**: do not pull passing output or prints into context, use minimal reporting (for example `pytest -q`), and only read and surface the output of tests that **fail**.
3. Use the pytest-xdist parallel runner as in CI (`-n`), but **leave headroom** so the agent stays responsive while tests run: prefer capping workers to a couple fewer than the available cores (for example `-n 10` on a 12-core machine) rather than `-n auto`, which claims every core.
4. After the affected-package run finishes, **ask the user whether to run the full test suite** across all packages. Only run everything if they confirm, using the same quiet, parallel, failures-only approach.

If anything fails, report the failing tests and stop for the user; do not leave the branch in a broken state silently.
