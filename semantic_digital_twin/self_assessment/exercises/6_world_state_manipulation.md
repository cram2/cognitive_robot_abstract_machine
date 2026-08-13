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

(world-state-manipulation-exercise)=
# World State Manipulation

This exercise shows how to manipulate the state of an active connection.

You will:
- Create a world with a prismatic connection
- Update the connection position via its convenience property

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
```

## 1. Create and move a prismatic connection
Your goal:
- Create a `World` with two bodies named `root` and `slider`
- Add a small red box shape to `root`, and a small green box shape to `slider` for visibility
- Add a `PrismaticConnection` along the X axis from `root` to `slider`
- Keep a reference to this connection in a variable named `root_C_slider`
- Set `root_C_slider.position = 0.2`

```{code-cell} ipython3
:tags: [exercise]
# TODO: build the world and set the prismatic connection position to 0.2
world = ...
red_box = ...
green_box = ...
root = ...
slider = ...
root_C_slider: PrismaticConnection = ...

```

```{code-cell} ipython3
:tags: [example-solution]
world = World()
red_box = ShapeCollection([Box(
    scale=Scale(0.1, 0.1, 0.1),
    color=Color(1.0, 0.0, 0.0),
)])
green_box = ShapeCollection([Box(
    scale=Scale(0.1, 0.1, 0.1),
    color=Color(0.0, 1.0, 0.0),
)])
root = Body(name=PrefixedName("root"), visual=red_box, collision=red_box)
slider = Body(name=PrefixedName("slider"), visual=green_box, collision=green_box)
with world.modify_world():
    root_C_slider: PrismaticConnection = PrismaticConnection.create_with_dofs(
        parent=root,
        child=slider,
        axis=Vector3.X(reference_frame=root),
        world=world,
    )
    world.add_connection(root_C_slider)
rt = RayTracer(world); rt.update_scene(); rt.scene.show("jupyter")
```
```{code-cell} ipython3
:tags: [example-solution]
root_C_slider.position = 0.2
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
assert world is not ..., "The world should be created."
assert root is not ..., "The root body should be created."
assert slider is not ..., "The slider body should be created."
assert root_C_slider is not ..., "The prismatic connection should be created."
assert isinstance(root, Body), "`root` must be a Body."
assert isinstance(slider, Body), "`slider` must be a Body."
assert isinstance(root_C_slider, PrismaticConnection), "`root_C_slider` must be a PrismaticConnection."
assert abs(root_C_slider.position - 0.2) < 1e-6, "The slider should be at position 0.1 along X."
rt = RayTracer(world); rt.update_scene(); rt.scene.show("jupyter")
```
