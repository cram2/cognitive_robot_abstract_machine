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

(loading-worlds-exercise)=
# Loading Worlds

This exercise shows how to load a world description from a URDF file using the URDFParser.

You will:
- Compose a file path to a URDF file shipped with this repository
- Use URDFParser to create a World

## 0. Setup
Just execute this cell without changing anything.

```{code-cell} ipython3
:tags: [remove-input]
import logging
from semantic_digital_twin.adapters.urdf import URDFParser

from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.exceptions import ExerciseVerificationFailed

logging.disable(logging.CRITICAL)
```

## 1. Load the table world

Your goal:
- Load the URDF file into a World and store it in a variable named `world`

```{code-cell} ipython3
:tags: [exercise]

table_urdf = "package://iai_objects/urdf/table.urdf"

# TODO: parse the URDF into a World
world = ...

```

```{code-cell} ipython3
:tags: [example-solution]
table_urdf = "package://iai_objects/urdf/table.urdf"

world = URDFParser.from_file(file_path=table_urdf).create_world()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
from semantic_digital_twin.world import World
if world is ...: raise ExerciseVerificationFailed(requirement="Create a World by parsing the URDF file.")
if not isinstance(world, World): raise ExerciseVerificationFailed(requirement="`world` must be an instance of World.")
if len(world.bodies) != 6: raise ExerciseVerificationFailed(requirement="The loaded world must contain 6 bodies.")
if world.get_connection_by_name(name="left_front_leg_to_top") is None: raise ExerciseVerificationFailed(requirement="The world should contain a connection named 'left_front_leg_to_top'.")
rt = RayTracer(world=world); rt.update_scene(); rt.scene.show("jupyter")
```
