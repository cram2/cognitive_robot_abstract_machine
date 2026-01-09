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
# Advanced Example: Rule Trees with Alternatives and Exceptions

This example demonstrates how to build a rule tree using refinement (specialization) and 
alternatives (mutually exclusive branches). It shows how to:
- Start from a base conclusion;
- Add a refined exception (more specific case) that overrides the base when a further condition is met;
- Add alternatives that apply under different conditions.

We will construct objects symbolically using symbolic_rule and Add, with variable placeholders to describe relationships.

Lets define our domain model and build a small world. We will then build a rule tree that adds Drawer, Door, and Wardrobe
instances to the world.

```{code-cell} ipython3
from krrood.entity_query_language.entity import entity, variable, and_, Symbol, inference
from krrood.entity_query_language.entity_result_processors import an

from krrood.entity_query_language.rule import refinement, alternative
from krrood.entity_query_language.conclusion import Add

from dataclasses import dataclass, field
from typing_extensions import List


# --- Domain model
@dataclass
class Body(Symbol):
    name: str
    size: int = 1


@dataclass
class Container(Body):
    ...


@dataclass
class Handle(Body):
    ...


@dataclass
class Connection(Symbol):
    parent: Body
    child: Body


@dataclass
class FixedConnection(Connection):
    ...


@dataclass
class RevoluteConnection(Connection):
    ...


@dataclass
class World(Symbol):
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


@dataclass
class View(Symbol):  # A common super-type for Drawer/Door/Wardrobe in this example
    ...


# Views we will construct symbolically
@dataclass
class Drawer(View):
    handle: Body
    container: Body


@dataclass
class Door(View):
    handle: Body
    body: Body


@dataclass
class Wardrobe(View):
    handle: Body
    body: Body
    container: Body


# --- Build a small "world"
container1, body2, body3, container2 = Container("Container1"), Body("Body2", size=2), Body("Body3"), Container(
    "Container2")
handle1, handle2, handle3 = Handle("Handle1"), Handle("Handle2"), Handle("Handle3")
world = World(1, [container1, container2, body2, body3, handle1, handle2, handle3])

# Connections between bodies/handles
fixed_1 = FixedConnection(container1, handle1)
fixed_2 = FixedConnection(body2, handle2)
fixed_3 = FixedConnection(body3, handle3)
revolute_1 = RevoluteConnection(container2, body3)
world.connections = [fixed_1, fixed_2, fixed_3, revolute_1]
```

First build the base query.

```{code-cell} ipython3
from krrood.entity_query_language.predicate import HasType

# --- Build the starting query
# Declare the variables
fixed_connection = variable(type_=FixedConnection, domain=world.connections)
revolute_connection = variable(type_=RevoluteConnection, domain=world.connections)
views = inference(View)()

# Define aliases for convenience
handle = fixed_connection.child
body = fixed_connection.parent
container = revolute_connection.parent

# Describe base query
# We use a single selected variable that we will Add to in the rule tree.
query = an(entity(views).where(HasType(fixed_connection.child, Handle)))
```

Then we build the rule tree.

```{code-cell} ipython3
# --- Build the rule tree
with query:
    # Base conclusion: if a fixed connection exists between body and handle,
    # we consider it a Drawer by default.
    Add(views, inference(Drawer)(handle=handle, container=body))

    # Exception (refinement): If the body is "bigger" (size > 1), instead add a Door.
    # This refinement branch is more specific and can be seen as a refinement to the base rule.
    with refinement(body.size > 1):
        Add(views, inference(Door)(handle=handle, body=body))

        # Alternative refinement when the first refinement didn't fire: if the body is also connected to a parent
        # container via a revolute connection (alternative pattern), add a Wardrobe instead.
        with alternative(body == revolute_connection.child, container == revolute_connection.parent):
            Add(views, inference(Wardrobe)(handle=handle, body=body, container=container))
```

Finally, we evaluate the rule tree.

```{code-cell} ipython3
# Evaluate the rule tree
results = list(query.evaluate())

# The results include objects built from different branches of the rule tree.
# Depending on the world, you should observe a mix of Drawer, Door, and Wardrobe instances.
assert len(results) == 3
assert any(isinstance(v, Drawer) and v.handle.name == "Handle1" for v in results)
assert any(isinstance(v, Door) and v.handle.name == "Handle2" for v in results)
assert any(isinstance(v, Wardrobe) and v.handle.name == "Handle3" for v in results)
print(*results, sep="\n")
```

### Notes
- refinement(*conditions): narrows the context with an additional condition (like an exception/specialization).
- alternative(*conditions): introduces a sibling branch with its own conditions; only if those are satisfied will that
branch contribute conclusions.
- Add(target, value): materializes a conclusion into the selected variable (here, a collection-like placeholder views).