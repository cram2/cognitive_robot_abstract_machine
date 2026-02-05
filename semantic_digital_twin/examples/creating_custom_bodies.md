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
(creating-custom-bodies)=
# Creating Custom Bodies

The tutorial demonstrates the creation of a body and its visual and collision information.
First, let's create a world.

```{code-cell} ipython3
from pkg_resources import resource_filename

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix, RotationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

world = World()
```

Next, let's create the visual and collision information for our body.

The collision describes the geometry to use when calculating collision relevant things, for instance if your robot is colliding with a table while moving.
The visual information is purely for esthetics.
Both of these are collections of shapes.

Supported Shapes are:
- Box
- Sphere
- Cylinder
- FileMesh/TriangleMesh

Finally, in our kinematic structure, each entity has a name. For this we can use a simple datastructure called `PrefixedName`. You always need to provide a name, but the prefix is optional. This is for human readability and allows for easy identification of entities. For uniqueness constraints, a UUID is used and stored in the `id` field.

```{code-cell} ipython3
import os
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere, Cylinder, FileMesh, Color

box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
box = Box(origin=box_origin, scale=Scale(1., 1., 0.5), color=Color(1., 0., 0., 1., ))

sphere_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(pos_x=0, pos_y=1., pos_z=1., quat_x=0., quat_y=0., quat_z=0.,
                                                   quat_w=1.)
sphere = Sphere(origin=sphere_origin, radius=0.4)

cylinder_origin = HomogeneousTransformationMatrix.from_point_rotation_matrix(point=Point3.from_iterable([1, -1, 2]),
                                                                  rotation_matrix=RotationMatrix.from_axis_angle(
                                                                      Vector3.from_iterable([1., 0., 0.]), 0.8, ),)
cylinder = Cylinder(origin=cylinder_origin, width=0.05, height=0.5)

mesh = FileMesh(origin=HomogeneousTransformationMatrix(),
            filename=os.path.join(resource_filename("semantic_digital_twin", "../../"), "resources", "stl", "milk.stl"))

collision = ShapeCollection([cylinder, sphere, box])
visual = ShapeCollection([mesh])
body = Body(name=PrefixedName("my first body", "my first prefix"), visual=visual, collision=collision)
```

When modifying your world, keep in mind that you need to open a `world.modify_world()` whenever you want to add or remove things to/from your world

```{code-cell} ipython3
with world.modify_world():
    world.add_body(body)

from semantic_digital_twin.spatial_computations.raytracer import RayTracer
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

If you think you have understood everything in this tutorial, you may try out 
[our self-assessment quiz for this user guide](creating-custom-bodies-quiz)

```{warning}
Using the above method to visualize your world only really makes sense in a notebook setting like this.
If you want learn how to properly visualize your worlds, check out the [](visualizing-worlds) tutorial.
```

```{warning}
If you are trying to create multiple bodies without connecting them,
you will run into trouble with the world validation.
If you want to see how to create multiple bodies, 
check out the [](world-structure-manipulation) tutorial.
```