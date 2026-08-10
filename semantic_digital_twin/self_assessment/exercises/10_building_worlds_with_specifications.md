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

(building-worlds-with-specifications-exercise)=
# Building Worlds with Specifications

This exercise demonstrates specifications: reusable, world-independent recipes for bodies,
connections, annotations, and whole scenes.

You will:
- Spawn a table from one reusable leg specification
- Give a body a degree of freedom through a connection specification
- Build an annotation specification with a nested part
- Describe a whole scene with a `WorldSpecification` and materialize it twice
- Place a robot into a scene with a `RobotSpecification`

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
import logging
import os
from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.api import (
    BodySpecification,
    PrismaticConnectionSpecification,
    RobotSpecification,
    SemanticAnnotationWithRootSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle, Milk
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.exceptions import ExerciseVerificationFailed

logging.disable(logging.CRITICAL)
root_path = Path(files("semantic_digital_twin")).parent.parent
table_urdf = os.path.join(root_path, "resources", "urdf", "table.urdf")
```

## 1. Spawn a table from one reusable leg specification
Your goal:
- Create a world with a root body and store it in a variable named `world`
- Spawn a table top from `BodySpecification.box("table_top", Scale(1.2, 0.8, 0.05))` and store
  the body in a variable named `table_top`
- Build a single leg specification `BodySpecification.box("leg", Scale(0.05, 0.05, 0.7))` and
  spawn it four times under the names `leg_0` … `leg_3`, attached to `table_top`, at the corner
  offsets `(±0.55, ±0.35, -0.35)` (override `name`, `parent`, and `parent_T_self` per spawn)

```{code-cell} ipython3
:tags: [exercise]
# TODO: spawn the table top, then reuse one leg specification for all four legs
# world = World.create_with_root_body()
# table_top = BodySpecification.box("table_top", Scale(1.2, 0.8, 0.05)).spawn(world)
# leg_specification = BodySpecification.box("leg", Scale(0.05, 0.05, 0.7))
# for index, (x, y) in enumerate([...]):
#     leg_specification.spawn(world, name=f"leg_{index}", parent=..., parent_T_self=...)
```

```{code-cell} ipython3
:tags: [example-solution]
world = World.create_with_root_body()
table_top = BodySpecification.box("table_top", Scale(1.2, 0.8, 0.05)).spawn(world)
leg_specification = BodySpecification.box("leg", Scale(0.05, 0.05, 0.7))
for index, (x, y) in enumerate([(0.55, 0.35), (0.55, -0.35), (-0.55, 0.35), (-0.55, -0.35)]):
    leg_specification.spawn(
        world,
        name=f"leg_{index}",
        parent=table_top,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=x, y=y, z=-0.35),
    )
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if len(world.bodies) != 6: raise ExerciseVerificationFailed("Expected the root, the table top, and four legs.")
for index in range(4):
    leg = world.get_body_by_name(f"leg_{index}")
    if leg.parent_connection.parent is not table_top: raise ExerciseVerificationFailed("Each leg should attach to the table top.")
```

## 2. Give a body a degree of freedom
Your goal:
- Spawn a body named `sliding_drawer` with scale `Scale(0.4, 0.5, 0.2)` whose
  `connection_specification` is a `PrismaticConnectionSpecification` with axis `Vector3.X()`
- Store the body in a variable named `sliding_drawer`

```{code-cell} ipython3
:tags: [exercise]
# TODO: pair the body specification with a prismatic connection specification
# sliding_drawer = BodySpecification.box(
#     name="sliding_drawer",
#     scale=...,
#     connection_specification=...,
# ).spawn(world)
```

```{code-cell} ipython3
:tags: [example-solution]
sliding_drawer = BodySpecification.box(
    name="sliding_drawer",
    scale=Scale(0.4, 0.5, 0.2),
    connection_specification=PrismaticConnectionSpecification(axis=Vector3.X()),
).spawn(world)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if not isinstance(sliding_drawer.parent_connection, PrismaticConnection): raise ExerciseVerificationFailed("The drawer should slide.")
```

## 3. Build an annotation specification with a nested part
Your goal:
- Use `Drawer.get_specification` with `Drawer.get_default_root_specification` for scale
  `Scale(0.4, 0.5, 0.6)` to build a drawer specification named `spec_drawer`
- Mount a `Handle` specification named `spec_handle` (scale `Scale(0.1, 0.05, 0.05)`) onto the
  drawer's `handle` field through `part_specifications`
- Spawn it into `world` and store the annotation in a variable named `drawer_with_handle`

```{code-cell} ipython3
:tags: [exercise]
# TODO: compose the drawer specification with a nested handle part and spawn it
# drawer_with_handle = Drawer.get_specification(
#     "spec_drawer",
#     Drawer.get_default_root_specification(scale=...),
#     part_specifications={"handle": ...},
# ).spawn(world)
```

```{code-cell} ipython3
:tags: [example-solution]
drawer_with_handle = Drawer.get_specification(
    "spec_drawer",
    Drawer.get_default_root_specification(scale=Scale(0.4, 0.5, 0.6)),
    part_specifications={
        "handle": Handle.get_specification(
            "spec_handle",
            Handle.get_default_root_specification(scale=Scale(0.1, 0.05, 0.05)),
        ),
    },
).spawn(world)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if not isinstance(drawer_with_handle.handle, Handle): raise ExerciseVerificationFailed("The handle part should be mounted.")
if drawer_with_handle.handle.root.parent_connection.parent is not drawer_with_handle.root: raise ExerciseVerificationFailed("The handle should be a kinematic child of the drawer root.")
```

## 4. Describe a whole scene and materialize it twice
Your goal:
- Build a `WorldSpecification` from `table_urdf` whose `objects` contain a `Milk` annotation
  specification (root `BodySpecification.box("milk", Scale(0.1, 0.1, 0.2))`) and a plain
  `BodySpecification.box("cup", Scale(0.07, 0.07, 0.1))`
- Store it in a variable named `scene_specification`
- Materialize it twice with `to_domain_object` into variables named `first_world` and `second_world`

```{code-cell} ipython3
:tags: [exercise]
# TODO: describe the scene once, materialize it twice
# scene_specification = WorldSpecification.from_urdf(
#     table_urdf,
#     objects=[...],
# )
# first_world = scene_specification.to_domain_object()
# second_world = scene_specification.to_domain_object()
```

```{code-cell} ipython3
:tags: [example-solution]
scene_specification = WorldSpecification.from_urdf(
    table_urdf,
    objects=[
        SemanticAnnotationWithRootSpecification(
            name="milk",
            semantic_annotation_type=Milk,
            root_specification=BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
        ),
        BodySpecification.box("cup", Scale(0.07, 0.07, 0.1)),
    ],
)
first_world = scene_specification.to_domain_object()
second_world = scene_specification.to_domain_object()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if first_world is second_world: raise ExerciseVerificationFailed("Each materialization should yield an independent world.")
if len(first_world.get_semantic_annotations_by_type(Milk)) != 1: raise ExerciseVerificationFailed("The first world should contain exactly one Milk.")
if len(second_world.get_semantic_annotations_by_type(Milk)) != 1: raise ExerciseVerificationFailed("The second world should contain exactly one Milk.")
if first_world.get_body_by_name("cup") is None: raise ExerciseVerificationFailed("The first world should contain a body named 'cup'.")
```

## 5. Place a robot into the scene
A robot is described by a `RobotSpecification`, which bundles the robot's semantic annotation
class with the poses that place it. `WorldSpecification.robots` takes a list of them.

Your goal:
- Build a `WorldSpecification` from `table_urdf` whose `robots` hold a single `RobotSpecification`
  for `PR2`, localized at `world_T_odom = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)`
- Store it in a variable named `robot_specification` and materialize it into `robot_world`

```{code-cell} ipython3
:tags: [exercise]
# TODO: describe a scene that contains a robot
# robot_specification = WorldSpecification.from_urdf(
#     table_urdf,
#     robots=[RobotSpecification(semantic_annotation_type=..., world_T_odom=...)],
# )
# robot_world = robot_specification.to_domain_object()
```

```{code-cell} ipython3
:tags: [example-solution]
robot_specification = WorldSpecification.from_urdf(
    table_urdf,
    robots=[
        RobotSpecification(
            semantic_annotation_type=PR2,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
        )
    ],
)
robot_world = robot_specification.to_domain_object()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if len(robot_world.get_semantic_annotations_by_type(PR2)) != 1: raise ExerciseVerificationFailed("The robot should be annotated.")
odom_body = robot_world.get_body_by_name("odom")
if odom_body.parent_connection.parent is not robot_world.root: raise ExerciseVerificationFailed("odom hangs off the world root.")
root_T_odom = robot_world.compute_forward_kinematics(robot_world.root, odom_body)
if not abs(root_T_odom.to_position().to_np()[0] - 1.0) < 1e-6: raise ExerciseVerificationFailed("The localization pose should apply.")
```
