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

# Pattern matching with `match` and `entity_matching`

EQL provides a concise pattern-matching API for building nested structural queries.
Use `match(type_)(...)` to describe a nested pattern on attributes, and wrap the outermost match
with `entity_matching(type_, domain)(...)` when you also need to bind a search domain.

The following example shows how nested patterns translate
into an equivalent manual query built with `entity(...)` and predicates.

```{code-cell} ipython3
from dataclasses import dataclass
from typing_extensions import List

from krrood.entity_query_language.entity import (
    let, entity, the,
    match, entity_matching, Symbol,
)
from krrood.entity_query_language.predicate import HasType


# --- Model -------------------------------------------------------------
@dataclass
class Body(Symbol):
    name: str


@dataclass
class Handle(Body):
    ...


@dataclass
class Container(Body):
    ...


@dataclass
class Connection(Symbol):
    parent: Body
    child: Body


@dataclass
class FixedConnection(Connection):
    ...


@dataclass
class World:
    connections: List[Connection]


# Build a small world with a few connections
c1 = Container("Container1")
h1 = Handle("Handle1")
other_c = Container("ContainerX")
other_h = Handle("HandleY")

world = World(
    connections=[
        FixedConnection(parent=c1, child=h1),
        FixedConnection(parent=other_c, child=h1),
    ]
)
```

## Matching a nested structure

`entity_matching(FixedConnection, world.connections)` selects from `world.connections` items of type
`FixedConnection`. Inner `match(...)` clauses describe constraints on attributes of that selected item.

```{code-cell} ipython3
fixed_connection_query = the(
    entity_matching(FixedConnection, world.connections)(
        parent=match(Container)(name="Container1"),
        child=match(Handle)(name="Handle1"),
    )
)
```

## The equivalent manual query

You can express the same query explicitly using `entity`, `let`, attribute comparisons, and `HasType` for
attribute type constraints:

```{code-cell} ipython3
fc = let(FixedConnection, domain=None)
fixed_connection_query_manual = the(
    entity(
        fc,
        HasType(fc.parent, Container),
        HasType(fc.child, Handle),
        fc.parent.name == "Container1",
        fc.child.name == "Handle1",
    )
)

# The two query objects are structurally equivalent
assert fixed_connection_query == fixed_connection_query_manual
```

## Evaluate the query

```{code-cell} ipython3
fixed_connection = fixed_connection_query.evaluate()
print(type(fixed_connection).__name__, fixed_connection.parent.name, fixed_connection.child.name)
```

Notes:
- Use `entity_matching` for the outer pattern when a domain is involved; inner attributes use `match`.
- Nested `match(...)` can be composed arbitrarily deep following your object graph.
- `entity_matching` is a syntactic sugar over the explicit `entity` + predicates form, so both are interchangeable.
