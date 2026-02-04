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

(world-state-manipulation)=
# World State Manipulation

In this tutorial we will manipulate the state (free variables) of the world.

Concepts Used:
- [](visualizing-worlds)
- Factories (TODO)
- [Entity Query Language](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/eql/intro.html)
- [](world-structure-manipulation)

First, we create a dresser containing a single drawer using the respective factories.

```{code-cell} ipython3
import threading
import time

import numpy as np
from krrood.entity_query_language.entity import entity, variable, in_
from krrood.entity_query_language.entity_result_processors import the

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle, Slider, Dresser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale, Box, Color
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()
root = Body(name=PrefixedName("root"))

with world.modify_world():
    world.add_body(root)
with world.modify_world():
    drawer= Drawer.create_with_new_body_in_world(
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
    drawer.add_handle(handle)

    slider = Slider.create_with_new_body_in_world(
        name=PrefixedName("drawer_slider"),
        world_root_T_self=HomogeneousTransformationMatrix(),
        world=world,
        active_axis=Vector3.X()
    )
    drawer.add_slider(slider)

    dresser = Dresser.create_with_new_body_in_world(
        name=PrefixedName("dresser"),
        scale=Scale(0.31, 0.31, 0.21),
        world=world,
    )

    dresser.add_drawer(drawer)

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

Let's get a reference to the drawer we built above.

```{code-cell} ipython3
drawer = the(
    entity(
        variable(type_=Drawer, domain=world.semantic_annotations),
    )
).evaluate()
```

We can update the drawer's state by altering the free variables position of its prismatic connection to the dresser.

```{code-cell} ipython3
drawer.root.parent_connection.position = 0.1
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

Note that this only works in this simple way for connections that only have one degree of freedom. For multiple degrees of freedom you either have to set the entire transformation or use the world state directly.
To show this we first create a new root for the world and make a free connection from the new root to the dresser.

```{code-cell} ipython3
from semantic_digital_twin.world_description.connections import Connection6DoF, PrismaticConnection
from semantic_digital_twin.world_description.world_entity import Body

with world.modify_world():
    old_root = world.root
    new_root = Body(name=PrefixedName("virtual root"))
    
    # Add a visual for the new root so we can see the change of position in the visualization
    box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=new_root)
    box = Box(origin=box_origin, scale=Scale(0.1, 0.1, 0.1), color=Color(1., 0., 0., 1.))
    new_root.collision = [box]
    
    world.add_body(new_root)
    root_T_dresser = Connection6DoF.create_with_dofs(parent=new_root, child=old_root, world=world)
    world.add_connection(root_T_dresser)
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

Now we can start moving the dresser everywhere and even rotate it.

```{code-cell} ipython3
from semantic_digital_twin.world_description.world_entity import Connection

connection = variable(type_=Connection, domain=world.connections)
free_connection = the(entity(connection).where(connection.parent == world.root)).evaluate()
with world.modify_world():
    free_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1., 1., 0., 0., 0., 0.5 * np.pi)
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

The final way of manipulating the world state is the registry for all degrees of freedom, the {py:class}`semantic_digital_twin.world_description.world_state.WorldState`.
This class acts as a dict like structure that maps degree of freedoms to their state.
The state is an array of 4 values: the position, velocity, acceleration and jerk.
Since it is an aggregation of all degree of freedoms existing in the world, it can be messy to access.
We can close the drawer again as follows:

```{code-cell} ipython3
connection = variable(PrismaticConnection, domain=world.connections)
connection = the(entity(connection).where(in_("drawer", connection.child.name.name))).evaluate()
with world.modify_world():
    world.state[connection.dof.id] = [0., 0., 0., 0.]
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```
