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

# Running a Statechart

This example builds the fallback plan from [Statecharts](../statecharts.md#example), runs it
with a `StatechartExecutor` and plots what happened.

## Building the statechart

Every node receives a `StatechartContext`, which holds the world the statechart runs in. The
`StatechartExecutor` compiles the statechart and ticks it until an `EndStatechart` ends it, so
the statechart is built in the context of the executor that runs it.

The executor's `pacer` decides how the ticks are spread over time:

| Pacer                    | Ticks                                                                  |
|--------------------------|------------------------------------------------------------------------|
| `NoPacing` (the default) | as fast as possible                                                    |
| `RealTimePacer`          | once per `tick_duration` of the context, in wall clock time            |
| `SimulationPacer`        | `real_time_factor` times as fast as real time                          |
| `SteppedSimulationPacer` | in lockstep with a physics simulation, which it steps one tick forward between two ticks |

Every pacer but `NoPacing` needs a context that knows its `tick_duration`. `extensions` add
behaviour to the executor around compiling, every tick, every recompile and the end of a run;
an extension may also add the [context extensions](../statecharts.md#writing-a-node) its
nodes read.

```{code-cell} ipython3
from cramph.composites import Sequence, TryInOrder
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import EndStatechart
from cramph.statechart import Statechart
from cramph.threaded_nodes import FunctionCall
from semantic_digital_twin.world import World

executor = StatechartExecutor(context=StatechartContext(world=World()))
statechart = Statechart(context=executor.context)


def slip():
    raise RuntimeError("the gripper slipped")


slow_approach = FunctionCall(
    name="slow approach", function=slip, failure_types=(RuntimeError,)
)
fast_approach = CountTicks(name="fast approach", ticks=5)

plan = Sequence(
    nodes=[
        TryInOrder(nodes=[slow_approach, fast_approach]),
        CountTicks(name="retreat", ticks=5),
    ]
)
statechart.add_node(plan)
statechart.add_node(EndStatechart.when_true(plan))
```

## Ticking it

```{code-cell} ipython3
executor.compile(statechart)
executor.tick_until_end(timeout=100)

print(f"Ticks executed: {executor.tick_count}")
print(f"slow approach: {slow_approach.life_cycle_state.name}")
print(f"fast approach: {fast_approach.life_cycle_state.name}")
print(f"plan: {plan.life_cycle_state.name}")
```

## Plotting

`Statechart.draw` renders the structure of the statechart and its transition conditions with
graphviz. `Statechart.plot_gantt_chart` renders the recorded history with two bars per node,
coloured by its life cycle state and by its observation in every tick. Pass a context with a
`tick_duration` to label the x-axis in seconds instead of ticks. `Statechart.visualize` opens an
interactive drawing of the nodes that follows their life cycle states while the statechart
ticks.

```{code-cell} ipython3
from pathlib import Path
from tempfile import mkdtemp

output_directory = Path(mkdtemp())
statechart.draw(str(output_directory / "statechart.pdf"))
statechart.plot_gantt_chart(str(output_directory / "statechart_gantt_chart.pdf"))
```
