---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(multiverse-exercise)=
# Simulation with Multiverse

This exercise demonstrates how to use MultiSim to simulate a world using different backends 
and have access to external controllers.
You will:
- Load the table URDF from the loading worlds exercise
- Simulate it using MultiSim with the Mujoco backend in 5 seconds
- Add two robotic hands to the simulation and control them using a VR Headset with Hand Tracking

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]

from pkg_resources import resource_filename
from semantic_digital_twin.adapters.urdf import URDFParser

from multiverse_simulator import MultiverseViewer
from semantic_digital_twin.adapters.multi_sim import MujocoSim
import os
import time
import logging

logging.disable(logging.CRITICAL)
root_path = resource_filename("semantic_digital_twin", "../../")
table_urdf = os.path.join(root_path, "resources", "urdf", "table.urdf")
```

## 1. Load the table and simulate it for 5 seconds
Your goal:
- Parse `table.urdf` into a variable named `world`
- Simulate it using `MultiSim` for 5 seconds
- 
```{code-cell} ipython3
:tags: [exercise]
# TODO: Parse the URDF
# world = ...
# TODO: Setup MultiSim with Mujoco backend and simulate for 5 seconds
# viewer = MultiverseViewer()
# headless = True
# multi_sim = ...
# multi_sim.start_simulation()

# start_time = time.time()
# time.sleep(5.0)
# multi_sim.stop_simulation()
```

```{code-cell} ipython3
:tags: [example-solution]
world = URDFParser.from_file(file_path=table_urdf).parse()
viewer = MultiverseViewer()
headless = True
multi_sim = MujocoSim(world=world, viewer=viewer, headless=headless, step_size=1E-3)
multi_sim.start_simulation()

start_time = time.time()
time.sleep(5.0)
multi_sim.stop_simulation()
```
