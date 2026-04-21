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

This example shows how to set up a basic `MotionStatechart` that runs for a specified amount of time using a `CountSeconds` monitor.

```{code-cell} ipython3
from giskardpy.executor import SimulationPacer, Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

# 1. Create a Motion Statechart
msc = MotionStatechart()

# 2. Add a monitor that counts for 1 second
msc.add_node(counter := CountSeconds(seconds=1.0))

# 3. Transition to EndMotion when the counter is finished
msc.add_node(EndMotion.when_true(counter))

# 4. Set up the Executor with a Simulation Pacer
kin_sim = Executor(
    context=MotionStatechartContext(
        world=World(),
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    ),
    pacer=SimulationPacer(real_time_factor=2.0),
)

# 5. Compile and run the statechart
kin_sim.compile(msc)
kin_sim.tick_until_end(timeout=1000)

print(f"Control cycles executed: {kin_sim.control_cycles}")
```
