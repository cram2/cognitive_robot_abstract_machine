(underspecified_grounding)=

# Grounding Underspecified Statements

```{contents}
:local:
:depth: 1
```

An `a(...)` statement in a plan describes an action without fixing all of its parameters. This page describes how
such a statement turns into the action it runs while the plan executes: on its own, together with the statements it
starts beside in a parallel, and again after its action failed.

The running example is a simplified version of the GARMI clean-up demo (`coraplex/demos/coraplex_garmi_clean_up_demo`):
GARMI opens a drawer with the left arm while the right arm picks up the milk, and the left hand may touch the drawer's
handle meanwhile. With its grasps chosen by closeness, the demo grounds the grasp on the handle and the grasp on the
milk together, in one parallel.

## The pieces

| Piece | Package | What it does |
|---|---|---|
| `StatementNode` | cramph | One per `a(...)` statement. Waits until a chooser hands it a child, and waits again when that child fails. |
| `StatementsGroundedTogether` | cramph | The combinations of the candidates of several statements, one candidate per statement, lowest rank sum first. |
| {class}`~coraplex.plans.underspecified.UnderspecifiedChildChooser` | coraplex, chooser | Grounds a statement on its own, or together with the statements that start with the same parallel, and runs the first candidate, or combination, that passes its trial. |
| {class}`~coraplex.plans.underspecified.CandidateHandOver` | coraplex, chooser | Hands a statement the candidate it was grounded to together with another one. |
| {class}`~coraplex.plans.underspecified.SubtreeRebuild` | coraplex | Rebuilds what is left to run around a statement onto the world copy, with candidates put in for statements. `unfinished_nodes` picks what is left to run. |
| {class}`~coraplex.plans.underspecified.ActionTrial` | coraplex | One kinematically simulated world copy that every trial runs in. |

## Which statements are grounded together

A statement starts with a parallel if it is a branch of that parallel, or the first step of a branch, recursively, but
never inside an action. When the first of these statements asks for its child, it grounds all of them that have not
chosen a child since the parallel last started. Nothing in the plan is moved or copied for this: the parallel keeps its
place and its branches.

```{figure} _static/images/underspecified_grounding/statements_grouped_together.svg
:alt: A parallel of two sequences and a collision allowance. The open and the milk pick-up start with the parallel and are grounded together; the cup pick-up and the place come later in their sequences and ground when their sequence reaches them; the allowance has no statement.
:width: 60%

The open and the milk pick-up, marked *together*, start with the parallel and are grounded as one choice. The cup
pick-up and the place, marked *later*, ground on their own when their sequence reaches them. The allowance has no
statement and starts with the parallel.
```

## Why the other branches start with the parallel

Grounding takes seconds: in the GARMI clean-up demo, one trial of a combination takes about 5 s. If the whole parallel
waited for it, so would everything beside the statements, such as the node that lets the left hand stay on the
drawer's handle. Between the end of the previous parallel and the start of this one, the hand would then hold the
handle with no allowance in force, and collision avoidance would stop the plan.

```{figure} _static/images/underspecified_grounding/branches_starting_with_the_parallel.svg
:alt: Two timelines. With the whole parallel waiting, the left hand's allowance from the previous parallel ends when the new parallel starts, and the new allowance only starts after grounding, leaving a gap. With the other branches starting with the parallel, the new allowance starts at once and the gap is gone.
:width: 100%

While a statement waits for its child, the statechart holds the robot still, on the robot as in simulation, so the
branches that already started do not move meanwhile. What they keep in force, such as collision rules and monitors, is
in force from the start.
```

## Every statement goes through the same loop

A statement on its own, a statement in a parallel, and a statement that starts halfway through a branch all ground the
same way. There is one trial, and it always tries what is left to run around the statement.

- Proposals come from the statement's query, ranked, and continue where the last grounding stopped, so a failed
  candidate is never proposed again. A statement that was grounded together with others goes on with the candidates
  read for the combinations before the one it ran, in rank order, then with the rest of its query.
- A candidate that passes is adopted: the plan transformations rewrite it, and a `Stalled` monitor gives up on it once
  nothing below it approaches its goal.
- When the query runs out, the statement stops grounding and fails. The executor then raises
  {class}`~coraplex.plans.failures.EmptyUnderspecified` for it, or
  {class}`~coraplex.plans.failures.CandidateLimitReached` if it hit its candidate limit.

```{figure} _static/images/underspecified_grounding/statement_grounding_loop.svg
:alt: Life cycle of a statement - it waits, asks for the next candidate, tries it with what is left to run in a world copy, adopts it if it passes, runs it; a failed trial asks for the next candidate, a failed or stalled child makes it wait again, and running out of candidates fails it with EmptyUnderspecified.
:width: 100%

The paths back are the retries. A failed trial never touched the real world, so the statement just takes the next
proposal. A failed or stalled child did run, so the statement waits again, and the robot is held still while it
grounds. The next trial therefore starts from rest, in the world the failure left behind.
```

## Statements that start together

All statements that start with a parallel start on the same tick and wait together, so the robot is held still once
rather than once per statement. The first of them that is asked walks through the combinations of their candidates and
tries each in what is left to run around it, the other branches included. The combinations come lowest rank sum first
(krrood's `CartesianProductByRankSum`), and a statement's query is asked for a further candidate only once a
combination needs it. Once a combination passes, the statement that asked runs its own candidate and hands every other
statement its candidate through {class}`~coraplex.plans.underspecified.CandidateHandOver`, so none of them is asked
again and their children start on the same tick.

```{figure} _static/images/underspecified_grounding/combinations_of_candidates.svg
:alt: Combinations are tried in order of lowest rank sum; the third, open rank 1 with milk pick-up rank 0, passes its trial. The statement that asked first runs its candidate and hands the other statement of the parallel its own, in place; the later statements still wait to ground.
:width: 100%

In this example the first two combinations fail their trial and the third, open candidate 1 with pick-up candidate 0,
passes. On the robot, the client sends Giskard one message per statement, in the order their children joined the plan,
and Giskard takes them all on one tick. A branch that fails later does not ground the others again. Only the failed
statement grounds again, as the next section shows.
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
- A parallel that needs a number of successes (`minimum_success`) needs one fewer for every child left out or cut,
  except a child that ended without succeeding, since that child can never count again.

```{figure} _static/images/underspecified_grounding/failed_branch_grounds_alone.svg
:alt: Left, the live plan when the open's child failed - a parallel holding the two sequences and the running allowance; right, the trial built from it in the world copy - the new open candidate beside the running place and a rebuilt allowance, the succeeded milk pick-up left out and the later cup pick-up cut.
:width: 100%

The remainder root is the parallel, so the allowance is part of the trial. If the trial passes, the open's statement
adopts the proposal, and nothing else in the live plan changes. If every proposal fails, the statement fails, which
fails the parallel around it. The milk pick-up never runs twice.
```

## A trial keeps the conditions of the plan

A rebuilt node keeps the transition conditions of the node it was rebuilt from, so a condition the plan wired by hand,
such as a branch that starts only once its sibling failed, holds in the trial as well. A condition reads the rebuilt
counterpart of every node it reads, and the current state of a node the trial leaves out, so it reads a node that
already ended as that node ended. A statement becomes the candidate's steps in the trial, so it cannot ground again and
a failing candidate fails the trial. Statements inside the candidate, such as the `a(PickUpAction)` a `TransportAction`
builds, still ground normally in the trial.

## How far the remainder reaches

The remainder root is the outermost parallel above the statement being grounded, with no action in between. If there
is no such parallel, the root is the statement itself. The search stops at an action because a trial rebuilds an
action from its parameters, which would ground everything inside it again.

```{figure} _static/images/underspecified_grounding/remainder_root.svg
:alt: Three cases - for a statement in a parallel beside an allowance, the outer parallel is the remainder root; for a statement inside an action the statement itself is; for a statement outside any parallel the statement itself is.
:width: 100%

Each column reads from the grounding statement (bottom) up through `node.path`. In the first column the root also
holds the allowance, which is how a statement is tried beside it. With the statement itself as root, what is left to
run is just the proposal, which is how a lone statement is tried.
```

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
rebuilt from its parameters and runs again in the world copy. A motion or a collision allowance is fine to run again; a
node with an immediate effect, such as a function call, can run twice in the copy.
```
