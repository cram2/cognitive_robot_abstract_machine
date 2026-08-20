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

(building-worlds-with-specifications)=
# Building Worlds with Specifications

In [](world-structure-manipulation) you built a world by hand: creating bodies, opening a
`world.modify_world()` block, wiring connections, and registering degrees of freedom one
statement at a time. That is precise, but the recipe for a single object ends up scattered
across many calls, and it is bound to one specific world.

A **specification** captures that recipe as a single, world-independent object. It describes
*what* an entity is — its geometry, its pose, the connection that attaches it, the semantic
meaning it carries — without touching any world. You materialize it later with one method call,
and the specification takes care of the bookkeeping: modification blocks, connections, degrees
of freedom, and annotation registration.

Two properties make specifications convenient:

- **Reusable.** A specification never binds to a world. Materializing it copies the prototype
  geometry and pose, so you can materialize the same specification into many worlds, or many
  times into one world under different names.
- **Composable.** Specifications nest. A body specification carries child specifications; an
  annotation specification carries nested part specifications; a world specification carries a
  whole list of objects to materialize.

Used Concepts:
- [](world-structure-manipulation)
- [](semantic_annotations)
- [](semantic_annotation_factories)

## The materialization verbs

Every specification carries a `name` and knows how to turn itself into a domain object. There
are three distinct contracts, and which one a specification offers tells you what kind of thing
it describes:

- **`spawn(world, ...)`** — used by specifications that describe an *entity together with the
  connection that attaches it to a parent*. Spawning materializes the entity, attaches it, and
  recursively materializes its children, all inside one modification block.
- **`connect(world, child, parent=)`** — used by [connection
  specifications](connecting-existing-entities), which join two *already existing* entities. A
  connection is not a kinematic structure entity — it *is* the attachment between two of them —
  so there is nothing to spawn and no parent to attach it to.
- **`to_domain_object(...)`** — materializes an entity in isolation, without attaching it to any
  world. Use it when you want a free-standing `Body` to pass somewhere else.

Throughout this guide we need a world with a root to materialize into.
`World.create_with_root_body()` gives us exactly that: a fresh world whose single root body is
named `map`.

```{code-cell} ipython3
import logging
logging.disable(logging.CRITICAL)

import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)

world = World.create_with_root_body()
assert world.root.name.name == "map"
print("Root body:", world.root.name)
```

## Body specifications

A `BodySpecification` describes a `Body`. The most direct way to build one is through the shape
constructors, mirroring the shapes you already know from [](creating-custom-bodies):

- `BodySpecification.box(name, scale, ...)`
- `BodySpecification.sphere(name, radius, ...)`
- `BodySpecification.cylinder(name, width, height, ...)`
- `BodySpecification.mesh(name, filename, ...)`

Calling `spawn` materializes the body and attaches it to the world root with a `FixedConnection`
by default.

```{code-cell} ipython3
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.world_description.geometry import Scale, Color

world = World.create_with_root_body()

table_top = BodySpecification.box(
    name="table_top", 
    scale=Scale(1.2, 0.8, 0.05), 
    color=Color(0.6, 0.4, 0.2, 1.0)
).spawn(world)

assert table_top.name.name == "table_top"
assert isinstance(table_top.parent_connection, FixedConnection)
assert table_top.parent_connection.parent is world.root
print("Spawned", table_top.name, "attached via", type(table_top.parent_connection).__name__)
```

### Placing and renaming

Each shape constructor accepts a `parent_T_self`: the default placement of the entity in its
parent frame. Because a specification is reusable, that placement and the name can also be
*overridden* when you spawn, together with the `parent` the entity attaches to.

```{code-cell} ipython3
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

# Bake a default pose straight into the specification.
leg = BodySpecification.box(
    name="leg_0",
    scale=Scale(0.05, 0.05, 0.7),
    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.55, y=0.35, z=-0.35),
).spawn(world, parent=table_top)

# The same specification is reusable; override the pose (and name) per spawn.
leg_spec = BodySpecification.box("leg", Scale(0.05, 0.05, 0.7))
for index, (x, y) in enumerate([(0.55, -0.35), (-0.55, 0.35), (-0.55, -0.35)], start=1):
    leg_spec.spawn(
        world,
        name=f"leg_{index}",
        parent=table_top,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=x, y=y, z=-0.35),
    )

# root + table_top + four legs
assert len(world.bodies) == 6
# The baked-in pose placed leg_0 at its parent-frame offset.
root_T_leg_0 = world.compute_forward_kinematics(world.root, leg)
np.testing.assert_allclose(root_T_leg_0.to_position().to_np()[:3], [0.55, 0.35, -0.35])
assert leg.parent_connection.parent is table_top
print("Bodies now in the world:", sorted(str(body.name) for body in world.bodies))
```

The legs attach to `table_top` rather than the world root, and the single `leg_spec` produced
independent bodies. The specification was neither consumed nor mutated.

### Nesting with child specifications

Re-attaching children by hand, as above, is fine for ad-hoc placement. When the parent/child
structure is fixed, encode it directly in the specification through `child_specifications`.
Spawning the parent then materializes the whole subtree.

```{code-cell} ipython3
world = World.create_with_root_body()

shelf = BodySpecification.box(
    name="shelf",
    scale=Scale(0.8, 0.3, 0.02),
    child_specifications=[
        BodySpecification.box("book", Scale(0.15, 0.2, 0.25)),
        BodySpecification.sphere("ball", 0.06),
    ],
).spawn(world)

book = world.get_body_by_name("book")
assert book.parent_connection.parent is shelf
assert world.get_body_by_name("ball").parent_connection.parent is shelf
print("book and ball are both children of the shelf")
```

### Other ways to describe geometry

Beyond primitive shapes, a body specification can describe composite geometry. These mirror the
corresponding `Body` constructors:

- `BodySpecification.mesh(name, filename, ...)` loads the geometry from a mesh file.
- `BodySpecification.from_event(name, event)` builds the geometry from the bounding boxes of a
  [random event](https://random-events.readthedocs.io/) — the construction used by semantic
  annotations with hollow or carved geometry.
- `BodySpecification.from_3d_points(name, points_3d)` builds the convex hull of a point cloud.

```{code-cell} ipython3
from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.spatial_types import Point3

resources = Path(files("semantic_digital_twin")).parent.parent / "resources"

world = World.create_with_root_body()

# A mesh body, loaded from file.
milk_mesh = BodySpecification.mesh(
    name="milk_mesh", filename=str(resources / "stl" / "milk.stl")
).spawn(world)

# A hollow crate: the outer box minus the carved-out interior, expressed as one event.
outer = Scale(0.3, 0.3, 0.3).to_simple_event().as_composite_set()
inner = Scale(0.26, 0.26, 0.3).to_simple_event().as_composite_set()
crate = BodySpecification.from_event("crate", outer - inner).spawn(world)

# The convex hull of a point cloud.
hull = BodySpecification.from_3d_points(
    "hull",
    [Point3(0, 0, 0), Point3(0.2, 0, 0), Point3(0, 0.2, 0), Point3(0, 0, 0.2)],
).spawn(world)

assert len(milk_mesh.collision.shapes) == 1
# Carving the interior out of the crate leaves a composite of boxes.
assert len(crate.collision.shapes) > 1
assert len(hull.collision.shapes) == 1
print("Crate is a composite of", len(crate.collision.shapes), "boxes")
```

A body specification also exposes body-only fields that the shape constructors leave at their
defaults: `inertial` for inertia properties and `visual_shapes` for a visual geometry that
differs from the collision geometry.

Because a specification carries its geometry, it can be measured before anything is spawned:
`scale` returns the extents of the combined collision geometry — the world-independent
counterpart of a spawned entity's `scale`.

```{code-cell} ipython3
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.inertial_properties import Inertial

world = World.create_with_root_body()

detailed = BodySpecification(
    name="detailed_box",
    shapes=Box(scale=Scale(1, 1, 1)).as_shape_collection(),
    visual_shapes=ShapeCollection([Box(scale=Scale(1.1, 1.1, 1.1))]),
    inertial=Inertial(mass=2.0),
).spawn(world)

assert detailed.visual is not detailed.collision
assert len(detailed.visual.shapes) == 1
assert detailed.inertial.mass == 2.0
assert BodySpecification.box("probe", Scale(1.2, 0.8, 0.05)).scale == Scale(1.2, 0.8, 0.05)
print("Distinct visual/collision geometry, mass =", detailed.inertial.mass)
```

## Region specifications

A `RegionSpecification` is the `Region` analogue of `BodySpecification`. Regions describe
abstract spatial areas rather than physical objects (see [](regions)), so a region specification
carries no inertia or visuals — only geometry, a pose, and children. It shares the same shape
constructors, including `parent_T_self`.

```{code-cell} ipython3
from semantic_digital_twin.api import RegionSpecification

world = World.create_with_root_body()

placement_area = RegionSpecification.box(
    name="placement_area", 
    scale=Scale(0.4, 0.4, 0.01)
).spawn(world)
assert len(placement_area.area.shapes) == 1
print("Spawned region:", placement_area.name)
```

## Connection specifications

By default a spawned body is rigidly fixed to its parent. To give it a degree of freedom — a
drawer that slides, a door that swings — pair the body specification with a **connection
specification**. Each connection family is its own specification type that carries exactly the
parameters that family needs:

| Specification | Connection | Use for |
|---|---|---|
| `FixedConnectionSpecification` | `FixedConnection` | A constant relative pose (the default) |
| `Connection6DoFSpecification` | `Connection6DoF` | A free-floating object that may move and rotate freely |
| `PrismaticConnectionSpecification` | `PrismaticConnection` | One translational DoF (a sliding drawer) |
| `RevoluteConnectionSpecification` | `RevoluteConnection` | One rotational DoF (a swinging door) |
| `ScrewConnectionSpecification` | `ScrewConnection` | One DoF coupling rotation and translation (a bottle cap on its thread) |

Every entity specification carries an optional `connection_specification`; when it is left unset,
`spawn` attaches the entity with a fixed connection. Set it to give the entity a degree of freedom.
The active families (`Prismatic`/`Revolute`/`Screw`) require a movement `axis`, and optionally accept a
`multiplier`, an `offset`, and `dof_limits`. A `ScrewConnectionSpecification` additionally requires a
`screw_pitch`: the distance between adjacent threads along the axis, which is what couples its
rotation to its translation.

```{code-cell} ipython3
from semantic_digital_twin.api import PrismaticConnectionSpecification
from semantic_digital_twin.spatial_types import Vector3

world = World.create_with_root_body()

drawer = BodySpecification.box(
    name="drawer",
    scale=Scale(0.4, 0.5, 0.2),
    connection_specification=PrismaticConnectionSpecification(axis=Vector3.Z()),
).spawn(world)

assert isinstance(drawer.parent_connection, PrismaticConnection)
print("Drawer is attached by a", type(drawer.parent_connection).__name__)
```

```{warning}
An active connection without an axis is rejected at spawn time. Spawning a
`PrismaticConnectionSpecification()` (no axis) raises `MissingConnectionAxisError`, because the
connection cannot generate its degree of freedom without one.
```

(connecting-existing-entities)=
## Connecting existing entities

The connection specifications above are also usable on their own, to join two entities that
already exist. Unlike a `spawn`, which materializes a new entity, `connect` requires you to
supply the `child` explicitly — a connection joins two pre-existing entities, it does not create
one. If you omit `parent`, the world root is used.

This pairs naturally with `to_domain_object`, which materializes a free-standing body without
attaching it anywhere.

```{code-cell} ipython3
from semantic_digital_twin.api import FixedConnectionSpecification

world = World.create_with_root_body()

# Materialize a body in isolation, then attach it with an explicit connection.
free_body = BodySpecification.box("crate", Scale(0.3, 0.3, 0.3)).to_domain_object()
connection = FixedConnectionSpecification().connect(
    world,
    parent=world.root,
    child=free_body,
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=2, z=3),
)

assert connection.child is free_body
root_T_crate = world.compute_forward_kinematics(world.root, free_body)
np.testing.assert_allclose(root_T_crate.to_position().to_np()[:3], [1, 2, 3])
print("Crate position:", root_T_crate.to_position().to_np()[:3].tolist())
```

## Semantic annotation specifications

Specifications also describe [semantic annotations](semantic_annotations). There are two ways to
build one, trading convenience for control.

### Building an annotation specification directly

`SemanticAnnotationWithRootSpecification` couples an annotation type with the specification of
the body or region it is rooted in. Use it when you want full control over the root geometry
(any `BodySpecification`/`RegionSpecification` you like), its pose, and its name. Spawning it
materializes the root entity, attaches it, registers the annotation, and materializes any
children.

```{code-cell} ipython3
from semantic_digital_twin.api import SemanticAnnotationWithRootSpecification
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

world = World.create_with_root_body()

milk = SemanticAnnotationWithRootSpecification(
    name="milk",
    semantic_annotation_type=Milk,
    root_specification=BodySpecification.box(
         name="milk", 
         scale=Scale(0.1, 0.1, 0.2),
         parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, z=0.8)
    ),
).spawn(world)

assert isinstance(milk, Milk)
assert milk in world.semantic_annotations
assert isinstance(milk.root.parent_connection, FixedConnection)
print("Spawned", type(milk).__name__, "rooted via", type(milk.root.parent_connection).__name__)
```

The root entity is what attaches to the parent, so the connection lives on the root
specification. You rarely have to state it: each annotation type builds its own through
`parent_connection_specification()` — a `FixedConnectionSpecification` for most annotations, a
`PrismaticConnectionSpecification` for a `Slider`, a `RevoluteConnectionSpecification` for a
`Hinge`. Because each connection family carries exactly the parameters it uses, that method takes
the parameters of *that* family and no others: `Slider.parent_connection_specification(axis=...)`
is valid, while offering an `axis` to a fixed annotation is an error rather than a silently
ignored argument.

What a type declares is only a default: setting `connection_specification` on the root
specification replaces it, so a `Milk` may rest rigidly or float freely as a `Connection6DoF`.

```{code-cell} ipython3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Slider

world = World.create_with_root_body()

slider = SemanticAnnotationWithRootSpecification(
    name="slider",
    semantic_annotation_type=Slider,
    root_specification=BodySpecification.box("slider", Scale(0.1, 0.1, 0.1)),
).spawn(world)  # Slider's default parent connection is prismatic about z

assert isinstance(slider.root.parent_connection, PrismaticConnection)
print("Slider root is attached by a", type(slider.root.parent_connection).__name__)
```

### Default root specifications and nested parts

Most annotation classes can build their own geometry from a `Scale`.
`get_default_root_kinematic_structure_entity_specification` returns a ready-made `BodySpecification` (or
`RegionSpecification`) with that geometry filled in. This is the easy path: it hides the
geometry construction — which for many annotations is a
composite shape built from [random events](https://random-events.readthedocs.io/) (a hollow
handle, a carved container case, a wall minus its apertures) — behind a single scale. Reach for
it when a default shape is good enough, and build the root specification directly (previous
section) when you need a custom root shape.

`get_annotation_specification` then wraps any root specification into the type's
`SemanticAnnotationWithRootSpecification`, so building an annotation specification is two
composed calls. Both halves are spawnable, so the wrapping step is easy to forget: spawning a
root specification on its own yields a bare body or region, with none of the annotation's
semantics attached. Geometry parameters beyond the scale — a handle's `thickness`, a container
case's `wall_thickness` — live on `get_default_root_kinematic_structure_entity_specification`, whose signature
names them, so geometry is described in exactly one place:

```python
Drawer.get_annotation_specification(
    "drawer",
    Drawer.get_default_root_kinematic_structure_entity_specification(
        scale=Scale(0.4, 0.5, 0.6), wall_thickness=0.05
    ),
)
```

That builder also takes a `connection_specification`, exactly as `BodySpecification.box(...)`
does, so custom geometry and a custom attachment stay a single expression. Neither call needs a
name for the root: a root specification built without one defers naming, and the annotation
stamps its own name onto the root entity at spawn time.

`get_annotation_specification`'s `part_specifications` argument mounts nested annotations onto part-whole
relationship fields, keyed by the field name — spelled out before anything touches a world.

```{code-cell} ipython3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle

world = World.create_with_root_body()

drawer = Drawer.get_annotation_specification(
    "drawer",
    Drawer.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.4, 0.5, 0.6)),
    part_specifications={
        "handle": Handle.get_annotation_specification(
            "handle", Handle.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 0.05, 0.05))
        ),
    },
).spawn(world)

assert isinstance(drawer.handle, Handle)
assert drawer.handle.root.parent_connection.parent is drawer.root
print("Drawer has a", type(drawer.handle).__name__, "mounted on its root")
```

Annotation specifications take the same spawn-time overrides as entity specifications: `name`,
`parent`, and `parent_T_self`. Because the annotation stamps its name onto its root, one
specification can produce many identically shaped annotations under different names and poses.

```{code-cell} ipython3
handle_specification = Handle.get_annotation_specification(
    "handle", Handle.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 0.05, 0.05))
)
left = handle_specification.spawn(
    world,
    name="left_handle",
    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.2),
)
right = handle_specification.spawn(
    world,
    name="right_handle",
    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.2),
)

assert left.name.name == "left_handle" and right.name.name == "right_handle"
assert left.root.name.name == "left_handle"
print("One specification, two handles:", left.name, "and", right.name)
```

### Inert constructor fields

`annotation_kwargs` carries the annotation's *inert* constructor fields — plain dataclass fields
that are neither geometry nor part-whole relationships. They are handed to the annotation
constructor unchanged at spawn time. A `Table`, for example, references the region it supports
objects on:

```{code-cell} ipython3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table

world = World.create_with_root_body()

surface = RegionSpecification.box("surface", Scale(1, 1, 0.01)).spawn(world)
table = Table.get_annotation_specification(
    "table",
    Table.get_default_root_kinematic_structure_entity_specification(scale=Scale(1, 1, 0.5)),
    annotation_kwargs={"supporting_surface": surface},
).spawn(world)

assert table.supporting_surface is surface
print("Table supports objects on", table.supporting_surface.name)
```

### To-many parts and geometry that reacts to them

A list value mounts several parts onto a to-many field, while a single value mounts onto a
singular field. The wall/aperture pair below shows two more behaviors at once: an `Aperture` is
*region-rooted* — its root is a `Region`, not a `Body`, so its default root specification is a
`RegionSpecification` — and a `Wall` cuts its collision geometry around the mounted apertures at
spawn time. Each part's placement in the whole is its root specification's `parent_T_self`,
which is a plain field you can set after building the specification.

```{code-cell} ipython3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture, Wall
from semantic_digital_twin.world_description.world_entity import Region

world = World.create_with_root_body()

plain_wall = Wall.get_annotation_specification(
    "plain_wall", Wall.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 3, 3))
).spawn(world)

window_left = Aperture.get_annotation_specification(
    "window_left", Aperture.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 0.5, 0.5))
)
window_left.root_specification.parent_T_self = (
    HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.8)
)
window_right = Aperture.get_annotation_specification(
    "window_right", Aperture.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 0.5, 0.5))
)
window_right.root_specification.parent_T_self = (
    HomogeneousTransformationMatrix.from_xyz_rpy(y=0.8)
)

wall = Wall.get_annotation_specification(
    "wall",
    Wall.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 3, 3)),
    part_specifications={"apertures": [window_left, window_right]},
).spawn(world)

assert len(wall.apertures) == 2
assert all(isinstance(aperture.root, Region) for aperture in wall.apertures)
# Cutting the windows out of the wall turns one box into a composite.
assert len(wall.root.collision.shapes) > len(plain_wall.root.collision.shapes)
print(
    "Wall went from", len(plain_wall.root.collision.shapes),
    "shape to", len(wall.root.collision.shapes), "shapes",
)
```

### Parts that carry the whole

Mounting a part onto a `mechanical_joint` field does more than attach a child: the joint is what
the *whole* hangs from. Spawning rewires the tree to `parent -> joint -> whole`, with the
joint's own parent connection (here a revolute connection about `z`) carrying the motion.

```{code-cell} ipython3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Hinge
from semantic_digital_twin.world_description.connections import RevoluteConnection

world = World.create_with_root_body()

hinge_part = Hinge.get_annotation_specification(
    "hinge",
    Hinge.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.05, 0.05, 0.05)),
    parent_connection_specification=Hinge.parent_connection_specification(
        axis=Vector3.Z()
    ),
)
flap = Drawer.get_annotation_specification(
    "flap",
    Drawer.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.4, 0.5, 0.2)),
    part_specifications={"mechanical_joint": hinge_part},
).spawn(world)

# world.root -(revolute)-> hinge -(fixed)-> flap
assert flap.root.parent_connection.parent is flap.mechanical_joint.root
assert flap.mechanical_joint.root.parent_connection.parent is world.root
assert isinstance(flap.mechanical_joint.root.parent_connection, RevoluteConnection)
print("The flap hangs from its", type(flap.mechanical_joint).__name__)
```

The specification validates part keys *at construction time*, so misuse — a list on a singular
field, a key that is not a part-whole field, or a part-whole field smuggled in through
`annotation_kwargs` — fails fast, before any world is mutated.

## World specifications

The largest building block is `WorldSpecification`, which describes an entire scene: an
environment, an optional robot, and the objects placed around them. Its environment is held as the
parser that builds it — usually obtained from a model file with the `from_urdf` or `from_mjcf`
classmethods. Calling `to_domain_object` re-parses the environment every time, so one specification
can produce many independent worlds and no two of them share an entity identifier.

```{code-cell} ipython3
import os
from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.api import WorldSpecification

table_urdf = os.path.join(
    Path(files("semantic_digital_twin")).parent.parent, "resources", "urdf", "table.urdf"
)

specification = WorldSpecification.from_urdf(
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

world = specification.to_domain_object()
assert len(world.get_semantic_annotations_by_type(Milk)) == 1
assert world.get_body_by_name("cup") is not None

# The specification is reusable: each call yields an independent world.
another_world = specification.to_domain_object()
assert world is not another_world
assert len(another_world.get_semantic_annotations_by_type(Milk)) == 1
print("Materialized two independent worlds, each with one milk and one cup")
```

`from_mjcf` is the MJCF twin of `from_urdf` and takes the same arguments:

```{code-cell} ipython3
mjcf_world = WorldSpecification.from_mjcf(
    str(resources / "mjcf" / "table.xml"),
    objects=[BodySpecification.box("cup", Scale(0.07, 0.07, 0.1))],
).to_domain_object()

assert mjcf_world.get_body_by_name("cup") is not None
print("MJCF table world has", len(mjcf_world.bodies), "bodies")
```

### Adding a robot

A world specification can also merge robots into the environment. Each one is described by a
`RobotSpecification`, which bundles a robot's [semantic annotation class](adding-robots) with
the poses that place it: `world_T_odom` sets the localization pose, and `odom_T_robot_start`
the robot's start pose. The robot is parsed from its own description and inserted as
`world.root -> odom -> connection -> robot`. The connection attaching the robot to its `odom`
is the drive determined by the robot's mobile base, or a fixed connection when the robot has
no mobile base.

```{code-cell} ipython3
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.robots.pr2 import PR2

world = WorldSpecification.from_urdf(
    file_path=table_urdf,
    robots=[
        RobotSpecification(
            semantic_annotation_type=PR2,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
            odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(y=2.0),
        )
    ],
).to_domain_object()
```

Because `robots` is a list, a world can hold several of them, each with its own localization —
including several instances of the same robot. Names are not unique in a world, so two PR2s
contribute two joints called `pr2/torso_lift_joint`. Each robot is therefore annotated while it
still owns the world it was parsed into, and its `odom` body carries its own identifier as a
name prefix, so the localization frames stay distinguishable.

```{code-cell} ipython3
two_robot_world = WorldSpecification.from_urdf(
    file_path=table_urdf,
    robots=[
        RobotSpecification(
            semantic_annotation_type=PR2,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
        ),
        RobotSpecification(
            semantic_annotation_type=PR2,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=-1.0),
        ),
    ],
).to_domain_object()

assert len(two_robot_world.get_semantic_annotations_by_type(PR2)) == 2
print("Two independently localized PR2s share one world")
```

## Putting it together

The example below assembles a small tabletop scene with a single specification — a table from
URDF, a drawer with a handle, and a milk carton placed above the table — and visualizes the
result. The milk's pose is baked into its root body specification through `parent_T_self`.

```{code-cell} ipython3
world = WorldSpecification.from_urdf(
    file_path=table_urdf,
    objects=[
        Drawer.get_annotation_specification(
            "drawer",
            Drawer.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.4, 0.5, 0.3)),
            part_specifications={
                "handle": Handle.get_annotation_specification(
                    "handle",
                    Handle.get_default_root_kinematic_structure_entity_specification(scale=Scale(0.1, 0.05, 0.05)),
                ),
            },
        ),
        SemanticAnnotationWithRootSpecification(
            name="milk",
            semantic_annotation_type=Milk,
            root_specification=BodySpecification.box(
                "milk",
                Scale(0.1, 0.1, 0.2),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, z=0.8),
            ),
        ),
    ],
).to_domain_object()

assert len(world.get_semantic_annotations_by_type(Drawer)) == 1
assert len(world.get_semantic_annotations_by_type(Milk)) == 1
print("Semantic annotations:")
print(*world.semantic_annotations, sep="\n")

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

```{warning}
As with the other tutorials, visualizing a world directly in a notebook with the `RayTracer` is
only meant for quick inspection. For proper visualization, see [](visualizing-worlds).
```

If you think you have understood everything in this tutorial, you may try out
[our self-assessment quiz for this user guide](building-worlds-with-specifications-quiz)
