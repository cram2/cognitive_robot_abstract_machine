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

(semantic_annotation_part_whole)=
# Part-Whole Relationships

Many semantic annotations are made of *parts*: a drawer has a handle and a slider, a dresser has
drawers and doors. This structural *part-of* relation is the **part-whole relationship**, and the
semantic digital twin models it with the single, type-routed method
{py:meth}`semantic_digital_twin.semantic_annotations.mixins.PartWholeRelationship.add`.

Calling `whole.add(part)` does two things:

1. It finds the typed *part-whole relationship field* of `whole` whose element type matches
   `type(part)` and stores the part there (a single-valued field like `handle`, or appends to a
   list field like `drawers`).
2. It lets the part mount itself into the kinematic structure (by default it becomes a kinematic
   child of the whole), so moving the whole moves the part with it.

A part-whole relationship is *parthood*, not occupancy: a handle is *part of* a drawer. The cup
standing *on* a table is not part of the table — that "located-in/on" relation handled
separately by {py:class}`semantic_digital_twin.semantic_annotations.mixins.IsStorageSpace` and its
`add_object` method.

Used Concepts:
- [](world-structure-manipulation)
- [](semantic_annotations)
- [](semantic_annotation_factories)
- [Entity Query Language](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/eql/intro.html)

## Composing built-in annotations with `add`

Let's build a dresser whose drawer has a handle and a slider. We use the factories
(`create_with_new_body_in_world`) to quickly spawn each part with geometry, then wire them
together with `add`.

```{code-cell} ipython3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle, Slider, Dresser
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World

world = World()
root = Body(name=PrefixedName("root"))

with world.modify_world():
    world.add_body(root)

with world.modify_world():
    dresser = Dresser.create_with_new_body_in_world(
        name=PrefixedName("dresser"),
        scale=Scale(0.31, 0.31, 0.21),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix(),
    )
    drawer = Drawer.create_with_new_body_in_world(
        name=PrefixedName("drawer"),
        scale=Scale(0.3, 0.3, 0.2),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix(),
    )
    handle = Handle.create_with_new_body_in_world(
        name=PrefixedName("drawer_handle"),
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.15),
        world=world,
    )
    slider = Slider.create_with_new_body_in_world(
        name=PrefixedName("drawer_slider"),
        world_root_T_self=HomogeneousTransformationMatrix(),
        world=world,
        active_axis=Vector3.X(),
    )

    # One method, routed by type: handle -> drawer.handle, slider -> drawer.mechanical_joint
    drawer.add(handle)
    drawer.add(slider)
    # drawer -> dresser.drawers (a list field, so it is appended)
    dresser.add(drawer)

print("drawer.handle:", drawer.handle)
print("drawer.mechanical_joint:", drawer.mechanical_joint)
print("dresser.drawers:", dresser.drawers)

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

`add` raised nothing and put each part in the right place, because every part type matched
exactly one part-whole relationship field of its target.

## Declaring your own part-whole relationship field with `part_whole_relationship_field`

To give a custom annotation its own part-whole relationship field, declare it with
{py:func}`semantic_digital_twin.semantic_annotations.mixins.part_whole_relationship_field` and
inherit from `PartWholeRelationship`. The marker on the field — not where the field sits in the
class hierarchy — is what makes it a part-whole relationship field, so a plain `field(...)` on the
same class is simply *not* one and is ignored by `add`.

```{code-cell} ipython3
from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.semantic_annotations.mixins import (
    PartWholeRelationship,
    HasRootBody,
    part_whole_relationship_field,
)

@dataclass(eq=False)
class ControlPanel(HasRootBody, PartWholeRelationship):
    """A custom annotation that can hold a single handle as a structural part."""

    handle: Optional[Handle] = part_whole_relationship_field(default=None)
    """A part-whole relationship field: parts of type ``Handle`` are routed here by ``add``."""

    label: Optional[str] = field(default=None)
    """A plain field — *not* a part-whole relationship field, so ``add`` never touches it."""
```

Now we can build a `ControlPanel` and add a handle to it exactly like the built-in annotations:

```{code-cell} ipython3
with world.modify_world():
    panel = ControlPanel.create_with_new_body_in_world(
        name=PrefixedName("panel"),
        scale=Scale(0.3, 0.2, 0.02),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5),
    )
    panel_handle = Handle.create_with_new_body_in_world(
        name=PrefixedName("panel_handle"),
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, z=0.5),
        world=world,
        scale=Scale(0.05, 0.1, 0.02),
    )
    panel.add(panel_handle)

print("panel.handle is panel_handle:", panel.handle is panel_handle)
```

If you try to add a part whose type matches none of the part-whole relationship fields, `add`
refuses with
{py:class}`semantic_digital_twin.exceptions.CannotBeAPartOf` — a `ControlPanel` has nowhere to put
a `Drawer`:

```{code-cell} ipython3
from semantic_digital_twin.exceptions import CannotBeAPartOf

with world.modify_world():
    stray_drawer = Drawer.create_with_new_body_in_world(
        name=PrefixedName("stray_drawer"),
        scale=Scale(0.2, 0.2, 0.2),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.6),
    )
    try:
        panel.add(stray_drawer)
    except CannotBeAPartOf as error:
        print("Rejected as expected:", error)
```
