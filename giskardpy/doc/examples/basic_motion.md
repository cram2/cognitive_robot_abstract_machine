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

# Basic Motion Statechart Example

This example shows how to set up a basic `Statechart` that runs for a specified amount of time using a `CountSeconds` monitor.

```{code-cell} ipython3
from cramph.context import StatechartContext
from cramph.executor import SimulationPacer, StatechartExecutor
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.monitors import CountSeconds
from cramph.statechart import Statechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

# 1. Set up the executor with motion control and a simulation pacer
executor = StatechartExecutor(
    context=StatechartContext(world=World()),
    pacer=SimulationPacer(real_time_factor=2.0),
    extensions=[
        MotionControl(
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
        )
    ],
)

# 2. Create a Motion Statechart in the context of the executor
statechart = Statechart(context=executor.context)

# 3. Add a monitor that counts for 1 second
statechart.add_node(counter := CountSeconds(seconds=1.0))

# 4. Transition to EndMotion when the counter is finished
statechart.add_node(EndMotion.when_true(counter))

# 5. Compile and run the statechart
executor.compile(statechart)
executor.tick_until_end(timeout=1000)

print(f"Control cycles executed: {executor.tick_count}")
```
