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

(creating-custom-bodies-exercise)=
# Creating Custom Bodies

This exercise guides you through creating a simple Body with visual and collision shapes and adding it to a World.

You will:
- Create a Body and assign simple shapes to its collision and visual collections
- Add the Body to a World and visualize it

## 0. Setup
Just execute this cell without changing anything. It imports the required classes and sets up the environment used in this exercise.

```{code-cell} ipython3
:tags: [remove-input]
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.geometry import Box, Sphere, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()
```

## 1. Create and attach geometry
Create two simple shapes relative to the body frame and assign them as collision and visual geometry.

Your goal:
- Create a red Box of size 0.2 x 0.2 x 0.2 centered at the Body origin
- Create a Sphere of radius 0.1 located at y = 0.3 in the Body frame
- Put the Box into the collision collection and the Sphere into the visual collection
- Create a Body with the name "exercise_body" and assign the collections to its collision and visual collections
- Add the Body to the World

```{code-cell} ipython3
:tags: [exercise]
# TODO: create and assign geometry to the body
collision_box: Box = ...
collision: ShapeCollection = ...
visual_sphere: Sphere = ...
visual: ShapeCollection = ...
body: Body = ...

```

```{code-cell} ipython3
:tags: [example-solution]
collision_box = Box(
    origin=HomogeneousTransformationMatrix(),
    scale=Scale(0.2, 0.2, 0.2),
    color=Color(1.0, 0.0, 0.0, 1.0),
)
visual_sphere = Sphere(
    origin=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.3),
    radius=0.1,
)
collision = ShapeCollection([collision_box])
visual = ShapeCollection([visual_sphere])

body = Body(name=PrefixedName("exercise_body"), collision=collision, visual=visual)
with world.modify_world():
    world.add_body(body)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
assert collision_box is not ... and isinstance(collision_box, Box), "Create a Box and assign it to the collision collection."
assert visual_sphere is not ... and isinstance(visual_sphere, Sphere), "Create a Sphere and assign it to the visual collection."
assert isinstance(body.collision, ShapeCollection), "Use a ShapeCollection for body.collision."
assert isinstance(body.visual, ShapeCollection), "Use a ShapeCollection for body.visual."
assert len(body.collision) == 1 and len(body.visual) == 1, "Each collection should contain exactly one shape."
assert abs(collision_box.scale.x - 0.2) < 1e-6, "Collision box should have size 0.2 in x."
rt = RayTracer(world); rt.update_scene(); rt.scene.show("jupyter")
```
