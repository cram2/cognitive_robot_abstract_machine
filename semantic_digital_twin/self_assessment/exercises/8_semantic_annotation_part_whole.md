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

(semantic-annotation-part-whole-exercise)=
# Part-Whole Relationships

This exercise demonstrates how `add` routes parts to the part-whole relationship fields of a
whole, and how to declare such a field on your own annotation class.

You will:
- Build a dresser whose drawer has a handle and a slider, wired together with `add`
- Declare a custom annotation that reuses the `HasHandle` mixin and adds its own part-whole
  relationship field for a part kind no mixin covers, then verify `add` routes into both

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
import logging
from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.semantic_annotations.mixins import HasHandle, HasRootBody
from semantic_digital_twin.semantic_annotations.part_whole import IsPartWholeRelationship
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Dresser, Handle, Slider
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.exceptions import ExerciseVerificationFailed

logging.disable(logging.CRITICAL)
```

## 1. Compose built-in annotations with `add`
Your goal:
- Create a world with a root body and store it in a variable named `world`
- Inside one `with world.modify_world():` block, use `create_with_new_body_in_world` to create
  a `Dresser` named `dresser` (scale `Scale(0.31, 0.31, 0.21)`), a `Drawer` named `drawer`
  (scale `Scale(0.3, 0.3, 0.2)`), a `Handle` named `drawer_handle`, and a `Slider` named
  `drawer_slider` with `Slider.parent_connection_specification(axis=Vector3.X())`
- Wire them together with `add`: the handle and the slider onto the drawer, the drawer onto the dresser
- Store the annotations in variables named `dresser`, `drawer`, `handle`, and `slider`

```{code-cell} ipython3
:tags: [exercise]
# TODO: create the four annotations and wire them together with add
# world = World.create_with_root_body()
# with world.modify_world():
#     dresser = Dresser.create_with_new_body_in_world(name="dresser", scale=Scale(0.31, 0.31, 0.21), world=world)
#     drawer = ...
#     handle = ...
#     slider = ...  # pass parent_connection_specification=Slider.parent_connection_specification(axis=Vector3.X())
#     drawer.add(...)
#     drawer.add(...)
#     dresser.add(...)
```

```{code-cell} ipython3
:tags: [example-solution]
world = World.create_with_root_body()
with world.modify_world():
    dresser = Dresser.create_with_new_body_in_world(
        name="dresser", scale=Scale(0.31, 0.31, 0.21), world=world
    )
    drawer = Drawer.create_with_new_body_in_world(
        name="drawer", scale=Scale(0.3, 0.3, 0.2), world=world
    )
    handle = Handle.create_with_new_body_in_world(
        name="drawer_handle",
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.15),
    )
    slider = Slider.create_with_new_body_in_world(
        name="drawer_slider",
        world=world,
        parent_connection_specification=Slider.parent_connection_specification(axis=Vector3.X()),
    )
    # One method, routed by type: handle -> drawer.handle, slider -> drawer.mechanical_joint
    drawer.add(handle)
    drawer.add(slider)
    # drawer -> dresser.drawers (a list field, so it is appended)
    dresser.add(drawer)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if drawer.handle is not handle: raise ExerciseVerificationFailed("The handle should be routed to drawer.handle.")
if drawer.mechanical_joint is not slider: raise ExerciseVerificationFailed("The slider should be routed to drawer.mechanical_joint.")
if drawer not in dresser.drawers: raise ExerciseVerificationFailed("The drawer should be appended to dresser.drawers.")
if handle.root.parent_connection.parent is not drawer.root: raise ExerciseVerificationFailed("The handle should be a kinematic child of the drawer.")
```

## 2. Reuse a mixin, declare a field only for a new part kind
Part kinds the library already models come with ready-made mixins (`HasHandle`, `HasDrawers`,
`HasDoors`, ...). Inheriting the mixin gives your annotation the field and its `add` routing for
free — do not re-declare such a field yourself. Only a part kind no mixin covers needs its own
field, marked with `IsPartWholeRelationship` metadata.

Your goal:
- Declare a dataclass `SafetyLatch(HasRootBody)` — a part kind no built-in mixin covers
- Declare a dataclass `ToolRack(HasHandle)` with two fields:
  - `latch: Optional[SafetyLatch]`, defaulting to `None`, marked with
    `metadata=IsPartWholeRelationship().as_dict()`
  - `label: Optional[str]`, defaulting to `None`, as a plain field
- Create a `ToolRack` named `rack` (scale `Scale(0.4, 0.1, 0.3)`), a `Handle` named
  `rack_handle`, and a `SafetyLatch` named `rack_latch` (scale `Scale(0.02, 0.02, 0.02)`) in
  `world`, then add the handle and the latch to the rack inside a modification block

```{code-cell} ipython3
:tags: [exercise]
# TODO: declare SafetyLatch and ToolRack, then route a handle and a latch into the rack with add
# @dataclass(eq=False)
# class SafetyLatch(HasRootBody):
#     ...
#
# @dataclass(eq=False)
# class ToolRack(HasHandle):  # the handle field comes from the mixin
#     latch: Optional[SafetyLatch] = field(default=None, metadata=...)
#     label: Optional[str] = field(default=None)
#
# with world.modify_world():
#     rack = ToolRack.create_with_new_body_in_world(...)
#     rack_handle = Handle.create_with_new_body_in_world(...)
#     rack_latch = SafetyLatch.create_with_new_body_in_world(...)
#     rack.add(rack_handle)
#     rack.add(rack_latch)
```

```{code-cell} ipython3
:tags: [example-solution]
@dataclass(eq=False)
class SafetyLatch(HasRootBody):
    """A part kind none of the built-in mixins cover."""


@dataclass(eq=False)
class ToolRack(HasHandle):
    """A rack with a handle and a safety latch as structural parts."""

    latch: Optional[SafetyLatch] = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """A part-whole relationship field: parts of type ``SafetyLatch`` are routed here by ``add``."""

    label: Optional[str] = field(default=None)
    """A plain field — *not* a part-whole relationship field, so ``add`` never touches it."""


with world.modify_world():
    rack = ToolRack.create_with_new_body_in_world(
        name="rack", scale=Scale(0.4, 0.1, 0.3), world=world
    )
    rack_handle = Handle.create_with_new_body_in_world(name="rack_handle", world=world)
    rack_latch = SafetyLatch.create_with_new_body_in_world(
        name="rack_latch", world=world, scale=Scale(0.02, 0.02, 0.02)
    )
    rack.add(rack_handle)
    rack.add(rack_latch)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if ToolRack.__dataclass_fields__["handle"] is not HasHandle.__dataclass_fields__["handle"]: raise ExerciseVerificationFailed("The handle field must come from the HasHandle mixin, not be re-declared.")
if rack.handle is not rack_handle: raise ExerciseVerificationFailed("The handle should be routed to the inherited rack.handle field.")
if rack.latch is not rack_latch: raise ExerciseVerificationFailed("The latch should be routed to rack.latch.")
if rack.label is not None: raise ExerciseVerificationFailed("add must not touch plain fields.")
if rack_handle.root.parent_connection.parent is not rack.root: raise ExerciseVerificationFailed("The handle should be a kinematic child of the rack.")
if rack_latch.root.parent_connection.parent is not rack.root: raise ExerciseVerificationFailed("The latch should be a kinematic child of the rack.")
```
