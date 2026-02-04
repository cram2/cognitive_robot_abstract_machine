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

(semantic_annotations)=
# Semantic Annotations

Semantic annotations can be used to say that a certain body should be interpreted as a handle or that a combination of
bodies should be interpreted as a drawer.
Ontologies inspire semantic annotations. The semantic digital twin overcomes the technical limitations of ontologies by representing
semantic annotations as Python classes and by using Python's typing together with the Entity Query Language (EQL) for reasoning.
This tutorial shows you how to apply semantic annotations to a world and how to create your own semantic_annotations.

Used Concepts:
- [](creating-custom-bodies)
- [](world-structure-manipulation)
- [Entity Query Language](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/eql/intro.html)

First, let's create a simple world that contains a couple of apples.

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Sphere, Box, Scale
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.semantic_annotations.semantic_annotations import Produce

@dataclass(eq=False)
class Apple(Produce):
    """A simple custom semantic_annotation declaring that a Body is an Apple."""

```

Semantic annotations are world entities, so it needs to have a unique ´PrefixedName`. You can either provide it directly
when creating the semantic annotation or let semantic_digital_twin generate a unique name for you. 

```{code-cell} ipython3
world = World()
with world.modify_world():
    root = Body(name=PrefixedName("root"))

    # Our first apple
    apple_body = Body(name=PrefixedName("apple_body"))
    sphere = Sphere(radius=0.15, origin=HomogeneousTransformationMatrix(reference_frame=apple_body))
    apple_body.collision = [sphere]
    apple_body.visual = [sphere]

    world.add_connection(Connection6DoF.create_with_dofs(parent=root, child=apple_body, world=world))
    world.add_semantic_annotation(Apple(root=apple_body, name=PrefixedName("apple1")))

    # Our second apple
    apple_body_2 = Body(name=PrefixedName("apple_body_2"))
    sphere2 = Sphere(radius=0.15, origin=HomogeneousTransformationMatrix(reference_frame=apple_body_2))
    apple_body_2.collision = [sphere2]
    apple_body_2.visual = [sphere2]
    c2 = Connection6DoF.create_with_dofs(parent=root, child=apple_body_2, world=world)
    world.add_connection(c2)
    # Move it a bit so we can see both
    world.state[c2.x.id].position = 0.3
    world.state[c2.y.id].position = 0.2
    world.add_semantic_annotation(Apple(root=apple_body_2, name=PrefixedName("apple2")))

print(world.get_semantic_annotations_by_type(Apple))
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

Thanks to the semantic annotations, an agent can query for apples directly using EQL:

```{code-cell} ipython3
from krrood.entity_query_language.entity_result_processors import an
from krrood.entity_query_language.entity import variable, entity
apples = an(entity(variable(Apple, world.semantic_annotations)))
print(*apples.evaluate(), sep="\n")
```

SemanticAnnotations can become arbitrarily expressive. For instance, we can define a ProduceBox that inherits 
from HasCaseA   sRootBody, has a list of Produces, for example our Apples, and defines the `hole_direction` classproperty.

```{code-cell} ipython3
from semantic_digital_twin.semantic_annotations.mixins import HasCaseAsRootBody
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from krrood.ormatic.utils import classproperty
from dataclasses import dataclass, field

@dataclass(eq=False)
class ProduceBox(HasCaseAsRootBody):
    produces: List[Produce] = field(default_factory=list)
    
    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.Z()
```
 
This is our first semantic annotation! They need to be dataclasses, because it makes it trivial to create datastructures which can be used 
by [KRROOD's ORMatic](https://github.com/code-iai/krrood/tree/main/src/krrood/ormatic) to automatically create ORM tables from python classes. 
Furthermore they need to have the `eq=False` flag, because otherwise the hash function defined in the `SemanticAnnotation` base class would be overridden.

```{code-cell} ipython3
with world.modify_world():
    # To create a hollowed out box in this case we use the ProduceBox.create_with_new_body_in_world method. 
    # To learn more about how cool SemanticAnnotationFactories are, please visit the appropriate guide!
    produce_box_with_apples = ProduceBox.create_with_new_body_in_world(
        name=PrefixedName("produce_box_with_apples"),
        scale=Scale(1.0, 1.0, 0.3),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3),
    )
    produce_box_with_apples.produces = world.get_semantic_annotations_by_type(Apple)

print(f"produce box with {len(produce_box_with_apples.produces)} produces")
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

Because these are plain Python classes, any other agent that imports your semantic_annotation definitions will understand exactly what
you mean. Interoperability comes for free without hidden formats or conversion issues.

---

We can incorporate the attributes of our SemanticAnnotations into our reasoning.
To demonstrate this, let's first create another ProduceBox, but which is empty this time.

```{code-cell} ipython3
with world.modify_world():
    empty_produce_box = ProduceBox.create_with_new_body_in_world(
        name=PrefixedName("empty_produce_box"),
        scale=Scale(1.0, 1.0, 0.3),
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3),
    )

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

We can now use EQL to get us only the ProduceBoxes that actually contain apples!

```{code-cell} ipython3
from semantic_digital_twin.reasoning.predicates import ContainsType
from krrood.entity_query_language.entity_result_processors import an

fb = variable(ProduceBox, domain=world.semantic_annotations)
produce_box_query = an(entity(fb).where(ContainsType(fb.produces, Apple)))

query_result = produce_box_query.evaluate()
print(list(query_result)[0] == produce_box_with_apples)
```