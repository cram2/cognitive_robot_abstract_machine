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

# Pattern matching with `match_variable` and `match`

EQL provides a concise pattern-matching API for building nested structural queries.
Use `match_variable(type_, domain=domain)(kwarg1=match(type_2)(...),...)` to describe a nested pattern on attributes.
This replaces `variable()` when you want to match a nested structure.

The following example shows how nested patterns translate
into an equivalent manual query built with `entity(...).where(...)`.

```{code-cell} ipython3
from krrood.entity_query_language.symbol_graph import SymbolGraph
from dataclasses import dataclass
from typing_extensions import List

from krrood.entity_query_language.entity import (
    variable, entity, Symbol,
)
from krrood.entity_query_language.entity_result_processors import the, a
from krrood.entity_query_language.match import (
    match_variable, match,
)
from krrood.entity_query_language.predicate import HasType


# --- Model -------------------------------------------------------------
@dataclass(unsafe_hash=True)
class Body(Symbol):
    name: str


@dataclass(unsafe_hash=True)
class Handle(Body):
    ...


@dataclass(unsafe_hash=True)
class Container(Body):
    size: int = 1


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
    bodies: List[Body]
    
@dataclass(unsafe_hash=True)
class Drawer(Symbol):
    handle: Handle
    container: Container


@dataclass
class Cabinet(Symbol):
    container: Container
    drawers: List[Drawer]

SymbolGraph()

# Build a small world with a few connections
c1 = Container("Container1")
h1 = Handle("Handle1")
other_c = Container("ContainerX", size=2)
other_h = Handle("HandleY")

world = World(
    connections=[
        FixedConnection(parent=c1, child=h1),
        FixedConnection(parent=other_c, child=h1),
    ],
    bodies = [c1, h1, other_c, other_h]
)
```

## Matching a nested structure

`match_variable(FixedConnection, domain=world.connections)` selects from `world.connections` items of type
`FixedConnection`. Inner `match(...)` clauses describe constraints on attributes of that selected item.

```{code-cell} ipython3
fixed_connection = match_variable(FixedConnection, domain=world.connections)(
        parent=match(Container)(name="Container1"),
        child=match(Handle)(name="Handle1")
    )
fixed_connection_query = the(entity(fixed_connection))
```

## The equivalent manual query

You can express the same query explicitly using `entity`, `variable`, attribute comparisons, and `HasType` for
attribute type constraints:

```{code-cell} ipython3
fc = variable(FixedConnection, domain=None)
fixed_connection_query_manual = the(
    entity(
        fc).where(
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
- Use `match_variable` as the outermost match because it allows binding domains and returns an expression.
- Nested `match(...)` can be composed arbitrarily deep following your object graph.
- `match_variable(...)` is syntactic sugar that allows creating a variable with a specific structure pre filtered.
- if you do not need a specific structure, just use `variable()` instead.