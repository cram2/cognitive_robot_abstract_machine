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

(semantic-annotations-exercise)=
# Semantic Annotations

This exercise introduces creating custom SemanticAnnotations, connecting bodies with a free connection, and querying with the Entity Query Language (EQL).

Your goals:
- First: Create Cap and Bottle semantic annotations and three cylinder bodies with exact sizes. Annotate them as described.
- Second: Connect the cap and the large bottle under the world root using a Connection6DoF, positioning the cap perfectly on top of the bottle.
- Third: Use EQL to query for all Bottle semantic_annotations that have a Cap assigned to them.

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
from dataclasses import dataclass, field
from typing import Optional
from krrood.entity_query_language.entity import entity, variable
from krrood.entity_query_language.entity_result_processors import an, the

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()

with world.modify_world():
    virtual_root = Body(name=PrefixedName("root"))
    world.add_body(virtual_root)
```

## 1. Define semantic annotations and create/annotate bodies
Your goals:
- Define two custom dataclasses Bottle(SemanticAnnotation) and Cap(SemanticAnnotation).
- A Cap is the semantic_annotation of a single body.
- A Bottle is the semantic_annotation of a body and optionally a Cap.
- Create one body with a smaller cylinder shape (height 2cm, width 3cm) as visual and collision; use it to create a Cap semantic_annotation.
- Create a second body with a larger cylinder shape (height 30cm, width 8cm) as visual and collision; use it to create a Bottle semantic_annotation. Also assign the Cap semantic_annotation to the Bottle semantic_annotation.
- Create a third body with a medium cylinder shape (height 15cm, width 4cm) as visual and collision; use it to create a Bottle semantic_annotation. Do not assign a Cap semantic_annotation to the Bottle semantic_annotation.

```{code-cell} ipython3
:tags: [exercise]

cap_cylinder: Cylinder = ...
bottle_large_cylinder: Cylinder = ...
bottle_medium_cylinder: Cylinder = ...

# ShapeCollections
cap_shapes: ShapeCollection = ...
bottle_large_shapes: ShapeCollection = ...
bottle_medium_shapes: ShapeCollection = ...

# Bodies
cap_body: Body = ...
bottle_large_body: Body = ...
bottle_medium_body: Body = ...

cap = ...
bottle_large = ...
bottle_medium = ...


```

```{code-cell} ipython3
:tags: [example-solution]

@dataclass(eq=False)
class Cap(SemanticAnnotation):
    """Semantic annotation declaring that a Body is a bottle cap."""

    body: Body = field(kw_only=True)


@dataclass(eq=False)
class Bottle(SemanticAnnotation):
    """Semantic annotation declaring that a Body is a bottle; may reference a Cap."""

    body: Body = field(kw_only=True)
    cap: Optional[Cap] = field(default=None, kw_only=True)


# Geometries (sizes are in meters)
cap_cylinder = Cylinder(width=0.03, height=0.02, origin=HomogeneousTransformationMatrix())
bottle_large_cylinder = Cylinder(width=0.08, height=0.30, origin=HomogeneousTransformationMatrix())
bottle_medium_cylinder = Cylinder(width=0.04, height=0.15, origin=HomogeneousTransformationMatrix())

# ShapeCollections
cap_shapes = ShapeCollection([cap_cylinder])
bottle_large_shapes = ShapeCollection([bottle_large_cylinder])
bottle_medium_shapes = ShapeCollection([bottle_medium_cylinder])

# Bodies
cap_body = Body(name=PrefixedName("cap_body"), collision=cap_shapes, visual=cap_shapes)
bottle_large_body = Body(
    name=PrefixedName("bottle_large"), collision=bottle_large_shapes, visual=bottle_large_shapes
)
bottle_medium_body = Body(
    name=PrefixedName("bottle_medium"), collision=bottle_medium_shapes, visual=bottle_medium_shapes
)

# Semantic annotations (not added to the world yet)
cap = Cap(body=cap_body)
bottle_large = Bottle(body=bottle_large_body, cap=cap)
bottle_medium = Bottle(body=bottle_medium_body)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
# Verify local objects exist and have correct relationships
assert isinstance(cap, Cap)
assert isinstance(bottle_large, Bottle)
assert isinstance(bottle_medium, Bottle)
assert bottle_large.cap is cap
assert bottle_medium.cap is None

# Verify bodies are attached to annotations
assert cap.body is cap_body
assert bottle_large.body is bottle_large_body
assert bottle_medium.body is bottle_medium_body

# Verify dimensions were set as requested
assert cap_cylinder.width == 0.03 and cap_cylinder.height == 0.02
assert bottle_large_cylinder.width == 0.08 and bottle_large_cylinder.height == 0.30
assert bottle_medium_cylinder.width == 0.04 and bottle_medium_cylinder.height == 0.15
```

## 2. Connect cap and large bottle under the root and place the cap on top
Your goals:
- Connect the cap body and the large bottle body with Connection6DoF connections under the world root.
- Add the SemanticAnnotations and Connections to the world.
- Use the exact cylinder parameters to place the cap perfectly on top of the bottle.

```{code-cell} ipython3
:tags: [exercise]

```

```{code-cell} ipython3
:tags: [example-solution]
# Register bodies and annotations then create free connections under a dedicated root body
with world.modify_world():
    world.add_semantic_annotation(cap)
    world.add_semantic_annotation(bottle_large)
    world.add_semantic_annotation(bottle_medium)

    root_C_bottle_large = Connection6DoF.create_with_dofs(parent=virtual_root, child=bottle_large_body, world=world)
    bottle_large_C_cap = Connection6DoF.create_with_dofs(parent=bottle_large_body, child=cap_body, world=world)
    root_C_bottle_medium = Connection6DoF.create_with_dofs(parent=virtual_root, child=bottle_medium_body, world=world)

    world.add_connection(root_C_bottle_large)
    world.add_connection(bottle_large_C_cap)
    world.add_connection(root_C_bottle_medium)
    
z_offset = bottle_large_cylinder.height / 2.0 + cap_cylinder.height / 2.0
cap_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    z=z_offset
)
bottle_large_C_cap.origin = cap_pose
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
# There should be three free connections we just added
from semantic_digital_twin.world_description.connections import Connection6DoF

con_bottle = world.get_connection(world.root, bottle_large_body)
con_cap = world.get_connection(bottle_large_body, cap_body)
con_medium = world.get_connection(world.root, bottle_medium_body)

assert isinstance(con_bottle, Connection6DoF)
assert isinstance(con_cap, Connection6DoF)
assert isinstance(con_medium, Connection6DoF)

# Bottle at origin, cap at computed z
import numpy as np
bottle_T = world.compute_forward_kinematics_np(world.root, bottle_large_body)
cap_T = world.compute_forward_kinematics_np(world.root, cap_body)

assert np.isclose(bottle_T[2, 3], 0.0)
assert np.isclose(cap_T[2, 3], z_offset)

# Also verify semantic annotations are in the world now
bottles = world.get_semantic_annotations_by_type(Bottle)
caps = world.get_semantic_annotations_by_type(Cap)
assert len(bottles) == 2
assert len(caps) == 1

# Visualize
rt = RayTracer(world); rt.update_scene(); rt.scene.show("jupyter")
```

## 3. Query with EQL for Bottles that have a Cap
Your goals:
- Build an EQL query that returns all Bottle semantic_annotations in the world that have a Cap assigned.
- Store the query in a variable named `bottles_with_cap_query` and the evaluated list in `query_result`.

```{code-cell} ipython3
:tags: [exercise]
# TODO (Third):
#  - Use EQL to query for Bottle semantic annotations with a non-empty `cap` field
#  - Name the query `bottles_with_cap_query` and the list result `query_result`

...
```

```{code-cell} ipython3
:tags: [example-solution]
bottle = variable(Bottle, domain=world.semantic_annotations)
bottles_with_cap_query = an(
    entity(
        bottle).where(bottle.cap != None
    )
)
query_result = list(bottles_with_cap_query.evaluate())
print(query_result)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
assert query_result is not ..., "The query result should be stored in a variable."
assert len(query_result) == 1, "There should be exactly one Bottle with a Cap returned by the query."
# And it should be the large bottle we annotated with the cap
assert query_result[0] is bottle_large
```
