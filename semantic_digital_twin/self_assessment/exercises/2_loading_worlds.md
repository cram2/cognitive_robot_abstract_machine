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
import os
from pkg_resources import resource_filename
from semantic_digital_twin.adapters.urdf import URDFParser

from semantic_digital_twin.spatial_computations.raytracer import RayTracer

logging.disable(logging.CRITICAL)
```

## 1. Load the table world

Your goal:
- Load the URDF file into a World and store it in a variable named `world`

```{code-cell} ipython3
:tags: [exercise]

root = resource_filename("semantic_digital_twin", "../../")
table_urdf = os.path.join(root, "resources", "urdf", "table.urdf")

# TODO: parse the URDF into a World
world = ...

```

```{code-cell} ipython3
:tags: [example-solution]
root = resource_filename("semantic_digital_twin", "../../")
table_urdf = os.path.join(root, "resources", "urdf", "table.urdf")

world = URDFParser.from_file(table_urdf).parse()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
from semantic_digital_twin.world import World
assert world is not ..., "Create a World by parsing the URDF file."
assert isinstance(world, World), "`world` must be an instance of World."
assert len(world.bodies) == 6, "The loaded world must contain 6 bodies."
assert world.get_connection_by_name("left_front_leg_to_top") is not None, "The world should contain a connection named 'left_front_leg_to_top'."
rt = RayTracer(world); rt.update_scene(); rt.scene.show("jupyter")
```
