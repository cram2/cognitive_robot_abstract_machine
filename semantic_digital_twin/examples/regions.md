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

(regions)=
# Regions

Regions are entities in the world similar to bodies; they live in the same
kinematic tree but represent semantic areas rather than physical geometry.
For example, a region can represent the surface of a table that you can
place objects on, or the opening of a container you can insert items into.

This tutorial explores a region describing the supporting surface of a table-top.

Used Concepts:
- [](creating-custom-bodies)
- [](world-structure-manipulation)
- [](world-state-manipulation)

First, let's create a simple table with one leg.

```{code-cell} ipython3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.world_entity import Body, Region
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()

root = Body(name=PrefixedName("root"))

table_leg = Body(name=PrefixedName("leg"))
leg_shapes = [
    Box(
        origin=HomogeneousTransformationMatrix(reference_frame=table_leg),
        scale=Scale(0.1, 0.1, 0.6),
    )
]
table_leg.collision = leg_shapes
table_leg.visual = leg_shapes

table_top = Body(name=PrefixedName("top"))
table_top_shapes = [
    Box(
        origin=HomogeneousTransformationMatrix(reference_frame=table_top),
        scale=Scale(1, 1, 0.05),
    )
]
table_top.collision = table_top_shapes
table_top.visual = table_top_shapes

with world.modify_world():
    root_to_leg = Connection6DoF.create_with_dofs(parent=root, child=table_leg, world=world)
    world.add_connection(root_to_leg)

    leg_to_top = FixedConnection(
        parent=table_leg,
        child=table_top,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.3, reference_frame=table_leg
        ),
    )
    world.add_connection(leg_to_top)
```

Next, we create a region describing the top of the table. We declare that the region is a very thin box that sits on top of the table-top.

```{code-cell} ipython3
table_surface = Region(
    name=PrefixedName("supporting surface of table"),
)

surface = Box(
    origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.05 / 2, reference_frame=table_surface),
    scale=Scale(1, 1, 0.001),
)
table_surface.area = [surface]
```

Regions are connected the same way bodies are connected.
Hence, you can specify how the regions move w. r. t. to a body or even another region.
We will now say the the region moves exactly as the table top moves.

```{code-cell} ipython3
with world.modify_world():
    world.add_kinematic_structure_entity(table_surface)
    connection = FixedConnection(table_top, table_surface)
    world.add_connection(connection)
print(world.regions)
```

We can now see that if we move the table, we also move the region.

```{code-cell} ipython3
print(table_surface.global_pose.to_position().to_np()[:3])

with world.modify_world():
    root_to_leg.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, reference_frame=table_leg
    )

print(table_surface.global_pose.to_position().to_np()[:3])
```

Note that Regions are a relatively new concept that may change in the future.
