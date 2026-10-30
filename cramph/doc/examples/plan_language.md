---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Plan Language

The plan language structures what a plan does. It is the set of cramph composites: statechart
nodes that run their children in a given order and decide, from how the children ended,
whether they succeeded themselves. A plan is a tree of these composites with the steps that
do the work at its leaves. [Statecharts](../statecharts.md#templates) describes how each of
them wires its children; this page runs them.

| Name                | Description                                                                                                |
|---------------------|------------------------------------------------------------------------------------------------------------|
| **Sequence**        | Runs its children one after another and fails as soon as one of them fails.                                |
| **TryInOrder**      | Runs its children one after another until one succeeds, and fails only if all of them failed.              |
| **Parallel**        | Holds all children at once and succeeds once enough of them, all by default, are at their goals together. |
| **TryAll**          | Runs all children at once and succeeds once one of them succeeded.                                         |
| **RepeatUntil**     | Attempts a child again whenever an attempt fails, until it succeeds or a monitor calls the repeating off.  |
| **Monitored nodes** | `CancelledWhenTrue`, `PausedWhileTrue` and `PausedUntilTrue` cancel or pause a child depending on a monitor. |

## Running a plan

The steps below are `CountTicks` nodes, which succeed once they ran for a number of ticks,
and `FunctionCall` nodes, which call a Python function. Packages built on cramph put their own
steps there, such as the actions of a robot. The `run` function gives every plan an executor of
its own, since an executor runs one statechart, and ends the statechart once the plan
succeeded or failed. A plan whose owner decides when it ends, such as a parallel at the top of
a statechart, is still running when it observes True and ends the statechart.

```{code-cell} ipython3
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import EndStatechart, StatechartNode
from cramph.statechart import Statechart
from cramph.threaded_nodes import FunctionCall
from semantic_digital_twin.world import World


def run(plan: StatechartNode) -> None:
    """
    Run `plan` until it succeeded or failed, and print its life cycle state and what it
    observes once the statechart ended.
    """
    executor = StatechartExecutor(context=StatechartContext(world=World()))
    statechart = Statechart(context=executor.context)
    statechart.add_node(plan)
    statechart.add_node(EndStatechart.when_true(plan))
    statechart.add_node(EndStatechart.when_failed(plan))
    executor.compile(statechart)
    executor.tick_until_end()
    print(
        f"{plan.name}: {plan.life_cycle_state.name}, observes {plan.observation_state.name}"
    )


def fail():
    raise RuntimeError("this step failed")


def failing_step(name: str) -> FunctionCall:
    """
    :return: A step that fails as soon as it runs.
    """
    return FunctionCall(name=name, function=fail, failure_types=(RuntimeError,))
```

## Sequence

A sequence runs its children one after another. If one of them fails, the sequence fails and
the children after it never start.

```{code-cell} ipython3
from cramph.composites import Sequence

drive = CountTicks(name="drive", ticks=3)
park = CountTicks(name="park", ticks=2)

run(Sequence(name="sequence", nodes=[drive, park]))
```

```{code-cell} ipython3
park = CountTicks(name="park", ticks=2)

run(Sequence(name="failing sequence", nodes=[failing_step("drive"), park]))
print(f"park: {park.life_cycle_state.name}")
```

## Try In Order

Try in order also runs its children one after another, but a failing child does not end it:
the next child is tried instead. It fails only if all of its children failed.

```{code-cell} ipython3
from cramph.composites import TryInOrder

run(
    TryInOrder(
        name="try in order",
        nodes=[failing_step("slow approach"), CountTicks(name="fast approach", ticks=3)],
    )
)
```

## Parallel

Parallel holds all of its children at once and succeeds once enough of them, all of them by
default, are at their goals on the same tick.

```{code-cell} ipython3
from cramph.composites import Parallel

run(
    Parallel(
        name="parallel",
        nodes=[CountTicks(name="drive", ticks=3), CountTicks(name="park", ticks=2)],
    )
)
```

## Try All

TryAll is to Parallel what TryInOrder is to Sequence: it runs all of its children at once and
succeeds once one of them succeeded.

```{code-cell} ipython3
from cramph.composites import TryAll

run(
    TryAll(
        name="try all",
        nodes=[failing_step("slow approach"), CountTicks(name="fast approach", ticks=3)],
    )
)
```

## Combining composites

Composites are statechart nodes themselves, so they nest. Here a sequence runs as one child of
a parallel.

```{code-cell} ipython3
drive = CountTicks(name="drive", ticks=3)
park = CountTicks(name="park", ticks=2)
raise_torso = CountTicks(name="raise torso", ticks=2)

run(Parallel(name="combined", nodes=[drive, Sequence(nodes=[park, raise_torso])]))
```

## Calling Python code

A `FunctionCall` calls its function in a thread of its own and succeeds once the function
returned. It fails when the function raises one of its `failure_types`; any other exception is
raised out of the execution.

```{code-cell} ipython3
def greet():
    print("Hello from a function call")


run(
    Parallel(
        name="with code",
        nodes=[
            CountTicks(name="park", ticks=2),
            FunctionCall(name="lambda", function=lambda: print("Hello from a lambda")),
            FunctionCall(name="function", function=greet),
        ],
    )
)
```

## Repeat

`RepeatUntil` attempts a task again whenever an attempt fails, until the task succeeds or its
stop monitor fires. Counting the attempts with `CountNodeResets` limits how often it tries, and
the exception it is given is raised once the attempts run out.

```{code-cell} ipython3
from cramph.composites import RepeatUntil
from cramph.exceptions import RepetitionsExhausted
from cramph.monitors import CountNodeResets

attempt = failing_step("attempt")
plan = RepeatUntil(
    name="repeat",
    task=attempt,
    stop_retry_monitor=CountNodeResets(node=attempt, target=3),
    exception=RepetitionsExhausted(repeated_node=attempt, maximum_repetitions=3),
)

try:
    run(plan)
except RepetitionsExhausted as exhausted:
    print(exhausted)
```

## Monitors

A monitor attaches a condition to a part of the plan that is evaluated alongside it, so it can
act on that part while it runs. The condition is a statechart node, here a `CountTicks` that
turns True after a number of ticks.

- `CancelledWhenTrue` stops its node once the monitor observes True and gives up on the plan
  with the exception it is given, such as `PlanCancelled`.
- `PausedWhileTrue` holds its node for as long as the monitor observes True.
- `PausedUntilTrue` holds its node until the monitor observes True, then lets it run.

```{code-cell} ipython3
from cramph.composites import CancelledWhenTrue
from cramph.exceptions import PlanCancelled

two_ticks = CountTicks(name="two ticks", ticks=2)
plan = CancelledWhenTrue(
    name="cancelled",
    monitor=two_ticks,
    monitored_node=CountTicks(name="long step", ticks=10),
    exception=PlanCancelled(monitor=two_ticks),
)

try:
    run(plan)
except PlanCancelled as cancelled:
    print(cancelled)
```

```{code-cell} ipython3
from cramph.composites import PausedUntilTrue

run(
    PausedUntilTrue(
        name="paused until true",
        monitor=CountTicks(name="two ticks", ticks=2),
        monitored_node=CountTicks(name="step", ticks=3),
    )
)
```
