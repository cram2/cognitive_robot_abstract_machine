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

(semantic-annotation-factories-exercise)=
# Factories

This exercise demonstrates the annotation factories: `create_with_new_body_in_world` spawns an
annotation together with a generated body, and `get_default_root_specification` returns the same
geometry as a reusable, world-independent specification.

You will:
- Create a drawer with a handle using the factories
- Extract the drawer's default geometry as a specification and compare it to the factory result

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
import logging

import numpy as np

from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.exceptions import ExerciseVerificationFailed

logging.disable(logging.CRITICAL)
```

## 1. Create a drawer with a handle
Your goal:
- Create a world with a root body and store it in a variable named `world`
- Inside one `with world.modify_world():` block, create a `Drawer` named `drawer` with scale
  `Scale(0.2, 0.4, 0.2)` and a `Handle` named `drawer_handle` with scale `Scale(0.05, 0.02, 0.1)`
  placed at `HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.1)`, both via
  `create_with_new_body_in_world`
- Mount the handle onto the drawer with `add`

```{code-cell} ipython3
:tags: [exercise]
# TODO: create the drawer and the handle via the factories, then mount the handle
# world = World.create_with_root_body()
# with world.modify_world():
#     drawer = Drawer.create_with_new_body_in_world(...)
#     handle = Handle.create_with_new_body_in_world(...)
#     drawer.add(handle)
```

```{code-cell} ipython3
:tags: [example-solution]
world = World.create_with_root_body()
with world.modify_world():
    drawer = Drawer.create_with_new_body_in_world(
        name="drawer", scale=Scale(0.2, 0.4, 0.2), world=world
    )
    handle = Handle.create_with_new_body_in_world(
        name="drawer_handle",
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.1),
        scale=Scale(0.05, 0.02, 0.1),
    )
    drawer.add(handle)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if drawer not in world.semantic_annotations: raise ExerciseVerificationFailed("The drawer should be registered in the world.")
if handle not in world.semantic_annotations: raise ExerciseVerificationFailed("The handle should be registered in the world.")
if drawer.handle is not handle: raise ExerciseVerificationFailed("The handle should be mounted on the drawer.")
if not len(drawer.root.collision.shapes) > 0: raise ExerciseVerificationFailed("The factory should have generated geometry for the drawer.")
```

## 2. Extract the geometry as a specification
Your goal:
- Build the drawer's default root specification for the same scale `Scale(0.2, 0.4, 0.2)` with
  `Drawer.get_default_root_specification` and store it in a variable named `specification`
- Materialize a free-standing body from it with `to_domain_object`, named `free_drawer_body`,
  and store it in a variable named `free_standing`

```{code-cell} ipython3
:tags: [exercise]
# TODO: build the default root specification and materialize a free-standing body
# specification = Drawer.get_default_root_specification(...)
# free_standing = specification.to_domain_object("free_drawer_body")
```

```{code-cell} ipython3
:tags: [example-solution]
specification = Drawer.get_default_root_specification(scale=Scale(0.2, 0.4, 0.2))
free_standing = specification.to_domain_object("free_drawer_body")
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
# The specification measures the same extents as the drawer the factory spawned.
if not np.allclose(specification.scale.to_np(), drawer.scale.to_np(), atol=1e-6): raise ExerciseVerificationFailed("The generated body should have the scale requested by the specification.")
if free_standing in world.bodies: raise ExerciseVerificationFailed("to_domain_object must not add the body to any world.")
```
