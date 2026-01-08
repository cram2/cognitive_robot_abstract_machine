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

Factories are convenience builders that create consistent worlds and their semantic annotations for you.
They are ideal for quickly setting up structured environments such as drawers, containers, and handles without
having to wire all bodies, connections, and semantic annotations manually.

Used Concepts:
- [](world-structure-manipulation)
- [Entity Query Language](https://abdelrhmanbassiouny.github.io/entity_query_language/intro.html)
- [](semantic_annotations)

## Create a drawer with a handle

```{code-cell} ipython3
from krrood.entity_query_language.entity import entity, variable
from krrood.entity_query_language.entity_result_processors import an, the

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.semantic_annotations.factories import (
    DrawerFactory,
    ContainerFactory,
    HandleFactory,
    Direction,
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world_description.geometry import Scale


# Build a simple drawer with a centered handle
world = DrawerFactory(
    name=PrefixedName("drawer"),
    container_factory=ContainerFactory(name=PrefixedName("container"), direction=Direction.Z, scale=Scale(0.2, 0.4, 0.2),),
    handle_factory=HandleFactory(name=PrefixedName("handle"), scale=Scale(0.05, 0.1, 0.02)),
    semantic_position=SemanticPositionDescription(
        horizontal_direction_chain=[
            HorizontalSemanticDirection.FULLY_CENTER,
        ],
        vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
    ),
).create()

print(*world.semantic_annotations, sep="\n")
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

You can query for components of the created furniture using EQL. For example, get all handles:

```{code-cell} ipython3

handles = an(entity(let(Handle, world.semantic_annotations)))
print(*handles.evaluate(), sep="\n")
```

## Add another handle and filter by context

```{code-cell} ipython3
# Create an extra handle world and merge it into the existing world at a different pose
useless_handle_world = HandleFactory(name=PrefixedName("useless_handle")).create()
print(useless_handle_world.semantic_annotations)

with world.modify_world():
    world.merge_world_at_pose(
        useless_handle_world,
        TransformationMatrix.from_xyz_rpy(x=1.0, y=1.0),
    )

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

With two handles in the world, the simple handle query yields multiple results:

```{code-cell} ipython3
handles = an(entity(let(Handle, world.semantic_annotations)))
print(*handles.evaluate(), sep="\n")
```

We can refine the query to get only the handle that belongs to a drawer:

```{code-cell} ipython3
drawer = let(Drawer, world.semantic_annotations)
handle = let(Handle, world.semantic_annotations)
result = an(entity(handle, drawer.handle == handle))
print(*result.evaluate(), sep="\n")
```
