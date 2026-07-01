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

(semantic_annotation_factories)=
# Factories

Factories, used by calling the `create_with_new_body_in_world` method of a semantic annotation inheriting from 
`HasRootBody`, are convenience builders that usually a create the respective semantic annotation as well as a new body 
with geometry that was generated based on parameters in the world provided to the factory.
Factories always spawn bodies relative to the world root. If you for some reason only have a transform relative to some 
other body, `world.transform(parent_T_your_body, world.root)` to get the correct `root_T_your_body`.
They are ideal for quickly setting up generic geometries of environments without having to wire all bodies, connections, 
and semantic annotations manually.

Each factory has a specification counterpart: `create_with_new_body_in_world` builds the annotation
in a world right away, while `get_default_annotation_specification` (and `get_default_body_specification`)
return the same geometry as a reusable specification you can materialize later. See
[](building-worlds-with-specifications) for when to prefer one over the other.

Used Concepts:
- [](world-structure-manipulation)
- [Entity Query Language](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/eql/intro.html)
- [](semantic_annotations)
- [](building-worlds-with-specifications)

## Create a drawer with a handle

```{code-cell} ipython3
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world import World


# Build a simple drawer with a centered handle
world = World.create_with_root_body()

with world.modify_world():
    drawer = Drawer.create_with_new_body_in_world(
        name="drawer",
        scale=Scale(0.2, 0.4, 0.2),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix(),
    )
    handle = Handle.create_with_new_body_in_world(
        name="drawer_handle",
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.1),
        world=world,
        scale=Scale(0.05, 0.1, 0.02)
    )
    drawer.add(handle)

assert drawer.handle is handle
assert drawer in world.semantic_annotations
assert handle in world.semantic_annotations
print(*world.semantic_annotations, sep="\n")
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```
