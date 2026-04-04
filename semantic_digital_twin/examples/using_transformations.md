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

(using-transformations)=
# Using Transformations

This tutorial introduces the mathematical idea of rigid-body transformations and how to use our `HomogeneousTransformationMatrix` to place and move things in a world.

You will learn:
- What a rigid transform is (position + orientation)
- How to create transforms using `HomogeneousTransformationMatrix.from_xyz_rpy`
- How to compose transforms using the `@` operator (parent_T_child style)
- How the `reference_frame` parameter determines which axes a motion is expressed in

At the end, you should be confident to complete the exercise: [](using-transformations-exercise).

```{note}
Notation: We follow the style `A_T_B` for "pose of B in the frame of A". This is explained in our style guide and makes composition readable.
```

## 0. Setup a tiny world
We will create a simple world with a square base plate and a small camera box mounted on top of it. We will position the base in the world, then attach the camera relative to the base.

```{code-cell} ipython3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
import math

world = World()

# Create an explicit root body like in other examples, so we can attach our base
root = Body(name=PrefixedName(name="root", prefix="world"))

# Geometry for a square base plate and a "camera" box
base_plate_shape = Box(scale=Scale(0.6, 0.6, 0.05), color=Color(0.3, 0.3, 0.3, 1.0))
camera_shape = Box(scale=Scale(0.10, 0.06, 0.05), color=Color(0.1, 0.2, 0.8, 1.0))

base_body = Body(
    name=PrefixedName(name="base", prefix="transform_example"),
    visual=ShapeCollection([base_plate_shape]),
    collision=ShapeCollection([base_plate_shape]),
)

camera_body = Body(
    name=PrefixedName(name="camera", prefix="transform_example"),
    visual=ShapeCollection([camera_shape]),
    collision=ShapeCollection([camera_shape]),
)


# Place the base: put it slightly above the ground so the top sits at z≈0.05
world_T_base = HomogeneousTransformationMatrix.from_xyz_rpy(z=0.025, reference_frame=root)

# Place the camera relative to the base: forward along base-y and a bit upward, yawed by +30°
base_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
    y=0.25,
    z=0.10,
    yaw=math.radians(30),
    reference_frame=base_body,
)

with world.modify_world():
    # Connect base to the world root, and camera to the base.
    world_C_base = Connection6DoF.create_with_dofs(parent=root, child=base_body, world=world)
    base_C_camera = Connection6DoF.create_with_dofs(parent=base_body, child=camera_body, world=world)

    world.add_connection(world_C_base)
    world.add_connection(base_C_camera)

# Set origins in a separate modification block so FK is compiled first
with world.modify_world():
    world_C_base.origin = world_T_base
    base_C_camera.origin = base_T_camera

# Visualize
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

## 1. What is a transform?
A rigid transform is a mapping between two Cartesian coordinate frames. It has:
- Translation: x, y, z (meters)
- Rotation: roll, pitch, yaw (radians), where
  - roll = rotation about x
  - pitch = rotation about y
  - yaw = rotation about z

Mathematically it is a 4×4 homogeneous matrix. Our `HomogeneousTransformationMatrix` wraps these values and provides convenient factories like `from_xyz_rpy(...)`.

```{code-cell} ipython3
# Compose transforms: world_T_camera = world_T_base @ base_T_camera
world_T_camera = world_T_base @ base_T_camera
print(
    "Camera position in world frame:",
    "x=%.3f" % float(world_T_camera.x.to_np().item()),
    "y=%.3f" % float(world_T_camera.y.to_np().item()),
    "z=%.3f" % float(world_T_camera.z.to_np().item()),
)
```

```{note}
Composition order matters. `parent_T_child @ child_T_grandchild` yields `parent_T_grandchild`. The result is expressed in the left-most frame (here: world).
```

## 2. Move along local vs. world axes
Suppose we want to move the camera 0.1 m forward along its own x-axis and rotate it an additional 45° about its own z-axis. To express a motion in the camera’s local axes, create the offset with `reference_frame=camera_body`.

```{code-cell} ipython3
# Move and rotate in the camera's own frame
camera_T_camera_offset = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=0.10,  # forward along camera-x
    yaw=math.radians(45),  # about camera-z
    reference_frame=camera_body,
)

# Update the connection origin in the base frame by composing on the right
base_T_camera_moved = base_T_camera @ camera_T_camera_offset

with world.modify_world():
    base_C_camera.origin = base_T_camera_moved

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

## 3. Reposition the base and see the chain update
If we rotate the base in the world by 90° yaw, the camera comes along because its pose is defined relative to the base.

```{code-cell} ipython3
# Pure rotation in the base frame (no translation)
base_T_base_rot = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=math.radians(90), reference_frame=base_body)
new_world_T_base = world_T_base @ base_T_base_rot

with world.modify_world():
    world_C_base.origin = new_world_T_base

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

## Where to go next
- Style guide on transformations and naming: https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/style_guide.html
- API docs for `HomogeneousTransformationMatrix`: https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/autoapi/semantic_digital_twin/spatial_types/spatial_types/index.html#semantic_digital_twin.spatial_types.spatial_types.HomogeneousTransformationMatrix
- Now try the exercise: [](using-transformations-exercise)
