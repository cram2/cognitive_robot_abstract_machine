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

(regions-exercise)=
# Regions

This exercise demonstrates how to define a Region and connect it into the kinematic tree so it moves with a body.

You will:
- Load the table URDF from the loading worlds exercise
- Create a region representing the table surface and attach it to the table surface named "top"
- Move the table and observe that the region follows

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
import os
import logging

from pkg_resources import resource_filename

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.world_entity import Body, Region

logging.disable(logging.CRITICAL)
root_path = resource_filename("semantic_digital_twin", "../../")
table_urdf = os.path.join(root_path, "resources", "urdf", "table.urdf")
```

## 1. Load the table and attach a surface region
Your goal:
- Parse `table.urdf` into a variable named `world`
- Find the table body named `top` in `world.bodies`
- Create a thin Region named `table_surface` and attach it to `top` via a `FixedConnection`
- Store the region in a variable named `surface_region`
- Save its initial position in a variable named `before_pos`

```{code-cell} ipython3
:tags: [exercise]
# TODO: load the URDF, find the top body, and attach a Region to it
# world = ...
# top = ...  # find the body whose name.name == "top"
# surface_region = Region(name=PrefixedName("table_surface"))
# surface_shape = Box(origin=HomogeneousTransformationMatrix(reference_frame=surface_region), scale=Scale(1.0, 1.0, 0.001))
# surface_region.area = [surface_shape]
# with world.modify_world():
#     world.add_kinematic_structure_entity(surface_region)
#     world.add_connection(FixedConnection(parent=top, child=surface_region))
# before_pos = surface_region.global_pose.to_position().to_np()[:3]
```

```{code-cell} ipython3
:tags: [example-solution]
world = URDFParser.from_file(table_urdf).parse()
# Find the table surface body named "top"
top = [b for b in world.bodies if b.name.name == "top"][0]
# Create a thin region at the table surface and attach it to the body "top"
surface_region = Region(name=PrefixedName("table_surface"))
surface_shape = Box(origin=HomogeneousTransformationMatrix(reference_frame=surface_region), scale=Scale(1.0, 1.0, 0.001))
surface_region.area = [surface_shape]
with world.modify_world():
    world.add_kinematic_structure_entity(surface_region)
    world.add_connection(FixedConnection(parent=top, child=surface_region))
# Remember the initial pose
before_pos = surface_region.global_pose.to_position().to_np()[:3]
```

## 2. Move the table and check the region
Your goal:
- Add a `Connection6DoF` between a new body named `root` and the current root of the table world
- Move the table by setting the 6DoF connection's `origin` to `HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, reference_frame=world.root)` inside a modification block

```{code-cell} ipython3
:tags: [exercise]
# TODO: move the table by adding a 6DoF to its root and updating its origin
# new_root = Body(name=PrefixedName("root"))
# with world.modify_world():
#     root_to_table = Connection6DoF.create_with_dofs(parent=new_root, child=world.root, world=world)
#     world.add_connection(root_to_table)
# with world.modify_world():
#     root_to_table.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, reference_frame=world.root)
```

```{code-cell} ipython3
:tags: [example-solution]
new_root = Body(name=PrefixedName("root"))
with world.modify_world():
    root_to_table = Connection6DoF.create_with_dofs(parent=new_root, child=world.root, world=world)
    world.add_connection(root_to_table)
with world.modify_world():
    root_to_table.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, reference_frame=world.root)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
after_pos = surface_region.global_pose.to_position().to_np()[:3]
assert (before_pos != after_pos).any(), "The region pose should change when the table moves."
```