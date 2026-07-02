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

(world-structure-manipulation)=
# World Structure Manipulation

This example demonstrates how to create and remove bodies, connections and degrees
of freedom (DoFs) in a World.

The structure of the world refers to the kinematic structure that describes relationships between bodies.
This kinematic structure has to be a tree.

The world structure is commonly manipulated in three ways:
- Adding/Removing a body
- Adding/Removing a connection
- Adding/Removing a region, which is discussed in [](regions).

Since the addition of a body to the world would, in most cases, violate the tree constraint, changes to the world structure have to be grouped in one
`with world.modify_world()` block. This context manager ensures that at the exit of such a block, the world is valid.

Let's create a simple world.

```{code-cell} ipython3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    RevoluteConnection,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()

# Create some bodies
root = Body(name=PrefixedName(name="root", prefix="world"))
base = Body(name=PrefixedName("base"))
link = Body(name=PrefixedName("link"))

# Group structural modifications so validation happens after the block
with world.modify_world():
    # 1) Add a passive 6DoF connection from root -> base.
    #    This will automatically create 7 passive DoFs (x, y, z, qx, qy, qz, qw)
    #    and register them in the world's state.
    c_root_base = Connection6DoF.create_with_dofs(parent=root, child=base, world=world)
    world.add_connection(c_root_base)

    # 2) Create a custom DoF and use it in an active RevoluteConnection
    #    from base -> link around the Z axis of the base frame.
    #    We add the DoF first, then reference it from the connection so the
    #    world's DoF set and the connection's DoFs match.
    dof = DegreeOfFreedom(name=PrefixedName("joint_z"))
    world.add_degree_of_freedom(dof)
    c_base_link = RevoluteConnection(
        parent=base,
        child=link,
        raw_dof=dof,
        axis=Vector3.Z(reference_frame=base),
    )
    world.add_connection(c_base_link)

print(f"Bodies after additions: {[str(b.name) for b in world.bodies]}")
print(f"Connections after additions: {[str(c.name) for c in world.connections]}")
print(f"Number of DoFs after additions: {len(world.degrees_of_freedom)}")
```

Now we want to remove the RevoluteConnection. This will also remove its DoF if no other connection uses it.
To keep the world a connected tree, we also have to remove the now-disconnected child body in the same block.

```{code-cell} ipython3
with world.modify_world():
    world.remove_connection(c_base_link)
    world.remove_kinematic_structure_entity(link)

# After the modification block, the world is validated automatically.
print(f"Final bodies: {[str(b.name) for b in world.bodies]}")
print(f"Final connections: {[str(c.name) for c in world.connections]}")
print(f"Final number of DoFs: {len(world.degrees_of_freedom)}")
```

Another world structure manipulation is the addition/removal of a DoF.
However, most DoFs are managed by connections and hence should not be mangled with directly.
If you ever feel the need to manage a degree of freedom manually you can do it like this:

```{code-cell} ipython3
with world.modify_world():
    dof = DegreeOfFreedom(name=PrefixedName("my_dof"))
    world.add_degree_of_freedom(dof)
    world.remove_degree_of_freedom(dof)
```
