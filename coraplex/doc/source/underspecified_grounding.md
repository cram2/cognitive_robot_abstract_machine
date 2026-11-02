(underspecified_grounding)=

# Grounding Underspecified Statements

```{contents}
:local:
:depth: 1
```

An `a(...)` statement in a plan describes an action without fixing all of its parameters. This page describes how
such a statement turns into the action it runs while the plan executes: on its own, together with the statements it
starts beside in a parallel, and again after its action failed.

The running example opens a drawer with the left arm while the right arm picks up the milk, with the left hand allowed
to touch the drawer's handle meanwhile.

## The pieces

| Piece | Package | What it does |
|---|---|---|
| `NodeGroundingStatements` | cramph | The base of the two nodes below. Grounds its statements into combinations of their candidates, one candidate per statement. |
| `StatementNode` | cramph | One per `a(...)` statement. Waits until a chooser hands it a child, and waits again when that child fails. Its combinations are its own candidates. |
| `ParallelStatementNode` | cramph | Stands in for the branches of a parallel that start with a statement. Chooses one child per run: those branches as one parallel. |
| {class}`~coraplex.plans.underspecified.GroundTheStatementsOfAParallelTogether` | coraplex, plan transformation | Part of {attr}`~coraplex.plans.executors.PlanExecutor.universal_adaptations`. Moves those branches into a `ParallelStatementNode` and leaves the others in the parallel. |
| {class}`~coraplex.plans.underspecified.ChooserGroundingStatements` | coraplex, chooser | The base of the two choosers below. Tries the combinations of a `NodeGroundingStatements` as the remainder of the plan around it, and runs the first that passes. |
| {class}`~coraplex.plans.underspecified.UnderspecifiedChildChooser` | coraplex, chooser | Grounds a `StatementNode`, which runs the candidate that passed. |
| {class}`~coraplex.plans.underspecified.ParallelStatementChildChooser` | coraplex, chooser | Grounds a `ParallelStatementNode`, which runs its branches with the candidates that passed handed over. |
| {class}`~coraplex.plans.underspecified.SubtreeRebuild` | coraplex | Rebuilds a subtree onto a world with candidates put in. `remainder_of` picks what is left to run; `build` is for the real run, `build_for_trial` for a trial. |
| {class}`~coraplex.plans.underspecified.ActionTrial` | coraplex | One kinematically simulated world copy, shared by both choosers. `succeeds(rebuild)` is its only trial. |

## Before the plan runs

When the plan is compiled, the transformation looks at every parallel before anything below it. A branch belongs to
the grounding if a statement starts when the branch starts: the branch is a statement, or its first step is one,
recursively, but never inside an action. Those branches move into one `ParallelStatementNode`, which keeps them as a
template. Every other branch stays in the parallel.

```{figure} _static/images/underspecified_grounding/statements_grouped_together.svg
:alt: A parallel with two branches starting with statements and a collision allowance keeps its place; the two statement branches move into a ParallelStatementNode among its children, kept as a template outside the statechart, while the allowance stays a child of the parallel.
:width: 100%

The two teal statements become `ParallelStatementNode.statements` and are grounded as one choice. The cup pick-up
and the place (dashed) stay ordinary `StatementNode`s and ground when their branch reaches them. The allowance (blue)
has no statement, so it stays a child of the parallel and starts with it. A parallel whose branches all start with a
statement, or one that needs a number of its children to succeed (`minimum_success`), goes into the
`ParallelStatementNode` as a whole.
```

## Why the other branches start with the parallel

Grounding takes seconds: in the GARMI hackathon demo, one combination trial takes about 5 s. If the whole parallel
waited for it, so would everything beside the statements, such as the node that lets the left hand stay on the
drawer's handle. Between the end of the previous parallel and the start of this one, the hand would then hold the
handle with no allowance in force, and collision avoidance would stop the plan.

```{figure} _static/images/underspecified_grounding/branches_starting_with_the_parallel.svg
:alt: Two timelines. Grouped as a whole, the left hand's allowance from the previous parallel ends when the new parallel starts, and the new allowance only starts after grounding, leaving a gap. With the other branches starting with the parallel, the new allowance starts at once and the gap is gone.
:width: 100%

On the robot, Giskard holds the robot still while the `ParallelStatementNode` waits, so the branches that already
started do not move meanwhile. What they keep in force, such as collision rules and monitors, is in force from the
start.
```

## Every statement goes through the same loop

A statement on its own, a statement in a parallel, and a statement that starts halfway through a branch all ground the
same way. There is one trial, and it always tries the remainder of the plan around the statement.

- Proposals come from the statement's query, ranked, and continue where the last grounding stopped, so a failed
  candidate is never proposed again. A statement that was grounded together with others continues the ranking its
  `ParallelStatementNode` used: first the other candidates it pulled, in rank order, then the rest of the query.
- A candidate that passes is adopted: the plan transformations rewrite it, and a `Stalled` monitor gives up on it once
  nothing below it approaches its goal.
- When the query runs out, the statement stops grounding and fails. The executor then raises
  {class}`~coraplex.plans.failures.EmptyUnderspecified` for it, or
  {class}`~coraplex.plans.failures.CandidateLimitReached` if it hit its candidate limit.

```{figure} _static/images/underspecified_grounding/statement_grounding_loop.svg
:alt: Life cycle of a statement - it waits, asks for the next candidate, tries it as the remainder in a world copy, adopts it if it passes, runs it; a failed trial asks for the next candidate, a failed or stalled child makes it wait again, and running out of candidates fails it with EmptyUnderspecified.
:width: 100%

The red paths are the retries. A failed trial never touched the real world, so the statement just takes the next
proposal. A failed or stalled child did run, so the statement waits again, and on the robot Giskard holds the robot
still while it grounds. The next trial therefore starts from rest, in the world the failure left behind.
```

## Statements that start together

When the parallel starts, its other branches start and the `ParallelStatementNode` is the only node that waits, so
the robot is held still once rather than once per statement. Its chooser walks through combinations of the
statements' candidates and tries each in what is left to run around it, the other branches included. The first
combination that passes is built onto the real world as one child. Every grouped statement in it has already adopted
its candidate, so nothing inside waits, the child travels to Giskard in one `ChildChoiceMessage`, and its branches
start on the same tick.

```{figure} _static/images/underspecified_grounding/combinations_of_candidates.svg
:alt: Combinations are tried in order of lowest rank sum; the third, open rank 1 with milk pick-up rank 0, passes its trial and is built as a Sequence holding the grouped branches, in which both grouped statements were handed their candidates and the later statements still wait to ground.
:width: 100%

In this example the first two combinations fail their trial and the third, open candidate 1 with pick-up candidate 0,
passes. The `ParallelStatementNode` chooses once per run and gets no stall monitor of its own; each candidate in the
parallel has its own. A branch that fails later does not send the parallel back here. Only the failed statement grounds
again, as the next section shows.
```

## A failed branch grounds again on its own

Suppose the open's child fails while the milk is already picked up and being placed. Only the open's statement grounds
again. Each proposal for it is tried with what is left of the plan around it, rebuilt in the world copy as it stands
now:

- Nodes that ended are left out, because what they did is already in the world. The milk pick-up does not run again.
- A statement whose candidate is still running keeps that candidate, so the place is tried beside the new open.
- A node without a statement that still runs, such as the allowance, is rebuilt from its parameters and runs again in
  the trial.
- The grounding statement gets the proposal.
- A branch is cut at its first statement that has no candidate, together with everything after it, because what
  follows depends on how that statement grounds.

```{figure} _static/images/underspecified_grounding/failed_branch_grounds_alone.svg
:alt: Left, the live plan when the open's child failed - a parallel holding the ParallelStatementNode's chosen branches and the running allowance; right, the trial built from it in the world copy - the new open candidate beside the running place and a rebuilt allowance, the succeeded milk pick-up left out and the later cup pick-up cut.
:width: 100%

The remainder root is the outer parallel, so the allowance beside the `ParallelStatementNode` is part of the trial.
If the trial passes, the open's statement adopts the proposal, and nothing else in the live plan changes. If every
proposal fails, the statement fails, which fails the parallel around it. The milk pick-up never runs twice.
```

## Trial build and real build

A {class}`~coraplex.plans.underspecified.SubtreeRebuild` builds the same subtree in two ways, and they differ only in
what a statement with a candidate becomes. In the trial, the statement becomes the candidate's steps directly, so it
cannot ground again and a failing candidate fails the trial. Statements inside the candidate, such as the
`a(PickUpAction)` a `TransportAction` builds, still ground normally in the trial.

```{figure} _static/images/underspecified_grounding/trial_build_and_real_build.svg
:alt: A statement with a candidate becomes the candidate's steps in a trial, and a StatementNode handed the candidate in the real run, which is rewritten, watched by a Stalled monitor, and can ground anew from the ranking it came from.
:width: 100%

The real build is what the `ParallelStatementNode` runs: every grouped statement is a `StatementNode` that already
adopted its candidate through {class}`~coraplex.plans.underspecified.CandidateHandOver`, so it can still ground again
on its own later. It takes over the ranking the combination came from (`StatementNode.ground_on_from`), so it never
proposes the candidate it was handed again.
```

## How far the remainder reaches

The remainder root is the outermost parallel above the node being grounded, a statement or an
`ParallelStatementNode`, with no action in between. If there is no such parallel, the root is the node itself. The
search stops at an action because a trial rebuilds an action from its parameters, which would ground everything inside
it again.

```{figure} _static/images/underspecified_grounding/remainder_root.svg
:alt: Three cases - for a statement in a parallel grouped beside an allowance, the outer parallel is the remainder root; for a statement inside an action the statement itself is; for a statement outside any parallel the statement itself is.
:width: 100%

Each column reads from the grounding statement (bottom) up through `node.path`. In the first column the
`ParallelStatementNode` itself is grounded against the same root, which is how its combinations are tried beside the
allowance. With the statement itself as root, the remainder is just the proposal, which is how a lone statement is
tried.
```

## Checked against the GARMI hackathon demo

The GARMI hackathon demo (`coraplex/demos/coraplex_garmi_hackerton_demo`) switches between hand-tuned grasps and
statements grounded by closeness. In the statement variant, the handle grasp and the milk pick-up start in one parallel
beside the left hand's allowance, and the place is a statement in the open ∥ place parallel. These runs are in
kinematic simulation.

| Scenario | Outcome | Wall time | What happened |
|---|---|---|---|
| Hand-tuned grasps | succeeded | 20.2 s | No statements, nothing grounded. |
| Grasps by closeness | succeeded | 31.5 s | Two choices, two trials (4.9 s and 3.8 s). The first combination passed: handle candidate 2 of 4, milk candidate 7 of 12. |
| Grasps by closeness, handle grasp made to fail once | `EmptyUnderspecified` | 32.8 s | Only the handle statement grounded again, beside the running pick-up. It skipped the grasp that failed and tried the other three, which all failed their trials. The pick-up did not run again. |

## Known gaps

```{warning}
**Actions that fail partway.** A failed attempt can leave the world half-changed, and the trial only sees what the
model shows. A pick-up that fails after its grasp leaves the box attached to the gripper. The next grasp passes its
kinematic trial and opens the gripper around the box, which on a real robot would drop it.
`test_a_pick_up_failing_after_its_grasp_grasps_the_held_box_again` pins this behaviour until actions have
preconditions.
```

```{warning}
**Running nodes run again in a trial.** A node without a statement that is still running when something grounds is
rebuilt from its parameters and runs again in the world copy. This happens for every grouped parallel beside such a
branch. A motion or a collision allowance is fine to run again; a node with an immediate effect, such as a function
call, can run twice in the copy.
```
