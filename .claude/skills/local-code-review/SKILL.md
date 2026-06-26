---
name: local-code-review
description: Review the current branch against the cram2 upstream main, checking for bugs and full adherence to AGENTS.md, then produce a plan-mode plan to fix every finding (including adding missing tests). Use when the user asks for a "local code review", to "review my branch before pushing", or to "review against cram2 main".
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, Agent, EnterPlanMode, ExitPlanMode
---

# Local Code Review

Review everything the current branch adds on top of the cram2 upstream `main`, then hand the developer an approval-gated plan to fix what you found, apply it, and verify. The coding standards you review and fix against are in `@AGENTS.md` — that file is the source of truth; do not restate its rules here. **Every change you propose or make must itself adhere to `@AGENTS.md`.**
When presenting the developer with a plan, and the developer clarifies the plan or wants it changed, incorporate the feedback into the plan and present it to the developer again for approval.

Stay **read-only through steps 1–5**. Do not edit, fix, or push anything until the plan is approved in step 6.

**Presenting choices:** whenever you offer the user multiple compatible options — failing tests to fix, bugs to fix, optional cleanups — always present them as a Markdown checklist (`- [ ] <id>: <description>`) with stable ids, so the user can decide fine-grainedly which individual items to act on. Never collapse multiple independent choices into a single all-or-nothing question. (A single mutually-exclusive decision, like the fan-out choice in step 4, is a normal question, not a checklist.)

## Durable state: the review report is the source of truth

Do not track review progress in your head — it is lost when context is compacted. The on-disk report is the single source of truth for *what has been reviewed*, *what was found*, and *what the user selected*. Track everything there as you go.

There is exactly one **active** report per run. Its path is recorded in the pointer file `.claude/claude_reviews/.active` so the current run can always be identified, even after compaction:

- **At the start of a run** (step 4) you create the report and write its path into `.claude/claude_reviews/.active`, overwriting any previous pointer.
- **To find the current report at any later step, or after a context compaction**, read `.claude/claude_reviews/.active` to get the path, then open that report. Never guess, and never read the other (stale) reports in the directory.
- **Validate before resuming.** The report header records the branch and the `HEAD` commit it was created against. If the current `HEAD` or branch no longer matches, the prior report is stale — tell the user and start a fresh run rather than resuming against changed code.

The report begins with a metadata header — branch, base (`$UPSTREAM/main` sha), `HEAD` sha, UTC timestamp, and a `status` field (`reviewing` → `awaiting-approval` → `fixing` → `verifying` → `done`) that you update as the run progresses. This header tells you, on resume, exactly which phase to continue from.

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

**Confirm unstaged files.** Untracked and unstaged changes are not in the committed diff. List them (excluding the noise paths above) and **ask the user, as a checklist, which of these files should be part of this branch/review**. Review only what they confirm.

If the resulting scope is empty, stop and report that there is nothing to review.

## 3. Run the affected tests first (baseline)

Before reviewing, establish whether the affected code already passes its tests. Identify the package(s)/module(s) touched by the in-scope changes and run their tests **quietly and in parallel**, surfacing only failures into context: use minimal reporting (for example `pytest -q`) and the pytest-xdist runner (`-n`) capped a couple below the available cores to leave headroom for the agent (for example `-n 10` on a 12-core machine), never `-n auto`.

If any tests fail, present the failing tests as a **checklist** and ask the user which to fix. Fixing a failing test means fixing the code, never editing the test to make it pass. Track the selected failures as must-fix work carried into the plan. Then proceed to the review regardless of the test outcome.

## 4. Review the full diff

First create the report and pointer for this run (see *Durable state* above):

```bash
mkdir -p .claude/claude_reviews
REPORT=".claude/claude_reviews/$(git rev-parse --abbrev-ref HEAD | tr '/' '-')-$(date -u +%Y%m%d-%H%M%S).md"
printf '%s\n' "$REPORT" > .claude/claude_reviews/.active
```

Write the metadata header, then a **coverage checklist listing every in-scope file unchecked** (`- [ ] path/to/file — pending`). Set `status: reviewing`.

Review **every** in-scope file — completeness matters more than speed. Under-reporting is the main failure mode of this step, so do not stop at a handful of findings and do not cap how many you report.

What every review (inline or delegated) must do per file, in priority order: evaluate **correctness** (logic errors, mishandled edge cases/failures), **tests** (new features/fixes must be covered; flag tests edited just to pass), and **`@AGENTS.md` adherence** in full (naming, type hints, field/attribute docstrings, import rules, SOLID/design, nesting/guard clauses, primitive overuse, and the spatial-type conventions in `semantic_digital_twin/doc/style_guide.md`). Read each file **in full** for context; a file may only be marked done after it has actually been read in full.

**Track coverage on disk, incrementally.** Immediately after reviewing each file, update its line in the report's coverage checklist — tick it and mark it `clean` or list the finding ids raised against it. This is how progress is tracked: never rely on memory. After a context compaction, the unticked files in the active report are exactly what remains to review.

### Choosing inline vs fan-out

Measure the diff with `git diff "$UPSTREAM/main...HEAD" --shortstat`.

- If it is **small (≤ 800 changed lines and ≤ 15 files)**, review **inline** without asking: enumerate the in-scope files and review them one file (or module) at a time, in batches rather than a single sweep so you do not skim or run out of room mid-review.
- If it is **large (> 800 changed lines or > 15 files)**, do not decide for the user. **Show the user the `--shortstat` output and ask them to choose** between two modes (a single mutually-exclusive decision, so ask it as a normal question, not a checklist):
  - **Fan out** — one review subagent per affected package, run in parallel. Faster and more thorough on large diffs, but **uses more total tokens**: each subagent has its own context window and separately reads its files plus `AGENTS.md`, and the parent then consolidates the results.
  - **Single agent** — review everything inline yourself. **Uses fewer tokens** and keeps one shared context, but is slower on a large diff and more prone to skimming.
  State this token trade-off explicitly in the prompt so the user can weigh cost against coverage.

**If the user chooses fan out:** group the in-scope files by affected package and dispatch **one review subagent per package** (use the Agent tool; the `Explore` agent type is a good read-only fit). Give each subagent: the list of files in its package, the absolute paths to `AGENTS.md` and `semantic_digital_twin/doc/style_guide.md`, the per-file evaluation criteria above, and the exact finding format from step 5. Each subagent reviews its slice **read-only** (it must not fix anything) and returns its findings plus its own per-file coverage list. When consolidating, renumber finding ids to be unique across the whole report, update the coverage checklist from the subagents' results, and confirm **every in-scope file appears and is ticked** — re-dispatch a subagent for any package whose files are missing.

## 5. Record findings

Add the findings to the active report created in step 4 (the one whose path is in `.claude/claude_reviews/.active`). Alongside the coverage checklist it already holds, write the findings as a **checklist** (`- [ ] <id>: ...`), each with `file_path:line_number`, severity, one sentence on what is wrong and why, and the concrete fix. Group by severity:

- **must-fix** — bugs, failing/missing tests, clear guideline violations
- **optional** — cleanups and judgement-call improvements

The coverage checklist and the findings checklist together make completeness auditable: every in-scope file is accounted for, and every finding is individually selectable. Set `status: awaiting-approval` once findings are recorded.

## 6. Produce the fix plan in plan mode

Enter plan mode and present, via ExitPlanMode, the findings as a **checklist the user marks** to choose which to fix — must-fix and optional items each individually selectable, never all-or-nothing. Map every selectable item to a fix step by its id and `file_path:line_number`.

For each bug, follow TDD: add a failing test that reproduces it first, then the fix. Include a dedicated step for any missing test coverage. Do not begin editing until the plan is approved.

## 7. Apply the approved fixes

Implement only the items the user selected. Every edit must adhere to `@AGENTS.md`. Do not modify existing tests to force them green; fix the code instead.

## 8. Verify before finishing

Once fixes are applied:

1. Regenerate the ORM interface: run `python scripts/regenerate_all_orm.py`.
2. Re-run the affected package(s)' tests the same way as step 3 — quiet, parallel with headroom, failures-only into context.
3. **If any tests fail, never stop silently.** Present the failing tests as a **checklist** and ask the user which to fix — do this regardless of whether the failures look related to the changes or were already failing at the step 3 baseline. Note for each whether it is newly introduced (passed at baseline, fails now) or pre-existing, so the user can decide with context, but always offer to fix every failure. Fix the selected tests by fixing the code (never edit a test to make it pass), then re-run and repeat until the affected packages are green or the user declines further fixes.
4. Once the affected packages are green, **ask the user whether to run the full test suite** across all packages, and only run everything if they confirm. Apply the same failing-tests checklist handling to any failures there.

Do not leave the branch in a broken state silently — the run only finishes once the affected packages pass or the user has explicitly chosen to leave specific failures unfixed.
