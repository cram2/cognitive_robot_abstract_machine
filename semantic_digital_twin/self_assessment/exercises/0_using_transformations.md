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

(using-transformations-exercise)=
# Using Transformations

This exercise gives you a hands-on introduction to rigid-body transformations in Semantic World:
- What a transformation is (position + orientation)
- How to create transformations with our TransformationMatrix and RotationMatrix
- How to place and move objects in a world by setting a connection pose

Provided to you is a world with a table and a 20 cm cube, as well as a way to visualize the world. You can click inside the visualized scene and drag to rotate the view, scroll to zoom, and right-click + drag to pan. Use it to have a look at the table and the cube before you start.

You will:
- Craft a transform to put the cube on top of the table
- Move the cube around using translations and rotations


## 0. Setup
Just execute this cell without changing anything.
Imports the required classes and sets up the environment used in this exercise. If import errors occur, ensure you run this notebook from the project repository environment.

```{code-cell} ipython3
:tags: [remove-input]

import logging
import math
import os


from pkg_resources import resource_filename
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import Connection6DoF, FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

root = resource_filename("semantic_digital_twin", "../../")
urdf_path = os.path.join(root, "resources", "urdf", "table.urdf")
table_world = URDFParser.from_file(urdf_path).parse()

root_C_left_leg = table_world.get_body_by_name("left_front_leg").parent_connection
moved_root_T_left_leg = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5, y=-0.5)
moved_root_C_left_leg = FixedConnection(name=root_C_left_leg.name,
                                        parent=root_C_left_leg.parent,
                                        child=root_C_left_leg.child,
                                        parent_T_connection_expression=moved_root_T_left_leg)

with table_world.modify_world():
    table_world.remove_connection(root_C_left_leg)
    table_world.add_connection(moved_root_C_left_leg)

cube = Box(scale=Scale(0.2, 0.2, 0.2), color=Color(R=1, G=0, B=0, A=1))
box_body = Body(
    name=PrefixedName(name="cube", prefix="transformation_exercise"),
    collision=ShapeCollection([cube]),
    visual=ShapeCollection([cube]),
)
with table_world.modify_world():
    table_world_C_box = Connection6DoF.create_with_dofs(parent=table_world.root, child=box_body, world=table_world)
    table_world.add_connection(table_world_C_box)

rt = RayTracer(table_world); rt.update_scene(); rt.scene.show("jupyter")
```

## 1. Craft a transform: Place the cube on top of the table
Now we will move the cube using a rigid transform. The pose of a 6DoF connection can be set via
the `origin`, which is a `HomogeneousTransformationMatrix` between the parent (in this case world root) and the child (in this case the cube).
This naming style, while not strictly pythonic, makes calculating with transformations a lot easier. To learn more about this naming convention, please refer to our [style guide](https://cram2.github.io/semantic_digital_twin/style_guide.html)!

Our goal is now to place the cube on top of the table (the cube should be lifted 72cm from the ground). For this you need to create a transform `new_table_world_T_box` below, and then comment in the rest of the code in the code in the cell, which will apply the transform to the Connection6DoF which connects the cube to the table.

Our `HomogeneousTransformationMatrix` class has multiple factory methods to create transforms, but for now you can focus on the `from_xyz_rpy` method, which creates a transform from a position (x, y, and z coordinates) and orientation. The orientation is represented by roll (rotation around the x-axis), pitch (rotation around the y-axis), and yaw (rotation around the z-axis)), all in radians.

Your goal:
- Create a translation-only transform using `HomogeneousTransformationMatrix.from_xyz_rpy(x=..., y=..., z=..., reference_frame=table_world.root)`
- Choose an assignment of x, y, and z coordinates that place the cube on top of the table, right in the middle.

Store your transform in a variable named `new_table_world_T_box`.

```{code-cell} ipython3
:tags: [exercise]

# TODO: set the cube on top of the table by crafting a transform
new_table_world_T_box: HomogeneousTransformationMatrix = ...

# Visualization
rt = RayTracer(table_world); rt.update_scene(); rt.scene.show("jupyter")
```

```{code-cell} ipython3
:tags: [example-solution]
new_table_world_T_box = HomogeneousTransformationMatrix.from_xyz_rpy(
    z=0.72,
    reference_frame=table_world.root,
)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]

assert new_table_world_T_box is not ..., "Create and assign a HomogeneousTransformationMatrix to place the cube on the table."
assert isinstance(new_table_world_T_box, HomogeneousTransformationMatrix), "Use a HomogeneousTransformationMatrix for `T_root_cube_on_table`."
with table_world.modify_world():
    table_world_C_box.origin = new_table_world_T_box
assert abs(new_table_world_T_box.x.to_np()) < 1e-5, "The cube should be at the middle of the table."
assert abs(new_table_world_T_box.y.to_np()) < 1e-5, "The cube should be at the middle of the table."
assert abs(new_table_world_T_box.z.to_np() - 0.72) < 1e-5, "The cube should be at z=0.72 on top of the table."
rt = RayTracer(table_world); rt.update_scene(); rt.scene.show("jupyter")

```

## 2. Move the cube around with rotations and translations
For the next part, we want to do our first transform multiplication. For this it is essential that you really understand what transforms are, what the different numerical values (x, y, z, roll, pitch, yaw) mean, and maybe most importantly, what the importance of the `reference_frame` is. You also need to know how to read our notation for transforms (eg. `table_world_T_box`). 

- Task A: Move the cube to `x = 0.3`, `y = -0.4` while keeping it on top of the table (keep the same `z`). Do this by taking the current transform `table_world_T_box` and multiplying it with `box_T_moved_box` which is a transformation matrix that moves the cube along the x and y axis by 0.3 and -0.4.

- Task B: Rotate the cube by 45 degrees around the Z axis while keeping its position. This works similarly to Task A. To get the correct radians, you can use `math.radians`.

You may accomplish both tasks at once by constructing a single transform and applying it to the connection. 

Store your updated transform in `table_world_T_moved_box` and apply it to `table_world_C_box.origin`.

If you don't know how to combine two transforms, you can check out [the appropriate section in our style guide](https://cram2.github.io/semantic_digital_twin/style_guide.html#combine-multiple-transformations)!.

```{code-cell} ipython3 
:tags: [exercise]
# TODO: translate and rotate the cube
table_world_C_box = box_body.parent_connection
table_world_T_box = table_world_C_box.origin
box_T_moved_box: HomogeneousTransformationMatrix = ...
table_world_T_moved_box: HomogeneousTransformationMatrix = ...

```

```{code-cell} ipython3
:tags: [example-solution]

table_world_C_box = box_body.parent_connection
table_world_T_box = table_world_C_box.origin
yaw = math.radians(45)
box_T_moved_box = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=-0.4, yaw=yaw, reference_frame=box_body)
table_world_T_moved_box = table_world_T_box @ box_T_moved_box

```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]

assert table_world_T_moved_box is not ..., "Craft a new transform to move and rotate the cube and assign it to `table_world_T_moved_box`."
assert isinstance(table_world_T_moved_box, HomogeneousTransformationMatrix), "`table_world_T_moved_box` must be a HomogeneousTransformationMatrix."

with table_world.modify_world():
    table_world_C_box.origin = table_world_T_moved_box
    
assert abs(table_world_T_moved_box.x.to_np() - 0.3) < 1e-5, "The cube should be at x=0.3 after the move."
assert abs(table_world_T_moved_box.y.to_np() + 0.4) < 1e-5, "The cube should be at y=-0.4 after the move."
assert abs(table_world_T_moved_box.z.to_np() - new_table_world_T_box.z.to_np()) < 1e-5, "The cube should stay on top of the table after the move."
rt = RayTracer(table_world); rt.update_scene(); rt.scene.show("jupyter")
```

## Final Notes

This is just a very basic introduction to transformations. To learn more about transformations, please refer to the [Wikipedia Article about transformations](https://en.wikipedia.org/wiki/Transformation_matrix), as well as our [TransformationMatrix API](https://cram2.github.io/semantic_digital_twin/autoapi/semantic_digital_twin/spatial_types/spatial_types/index.html#semantic_digital_twin.spatial_types.spatial_types.TransformationMatrix) for further details.

