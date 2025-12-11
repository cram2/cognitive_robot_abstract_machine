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

# Pattern matching with `matching`

EQL provides a concise pattern-matching API for building nested structural queries.
Use `matching(type_)(...)` to describe a nested pattern on attributes. Bind a search domain with
`.from_(domain)` on the outermost match expression, and quantify it with `the(...)` or `a(...)`.

The following example shows how nested patterns translate
into an equivalent manual query built with `entity(...)` and predicates.

```{code-cell} ipython3
from krrood.entity_query_language.symbol_graph import SymbolGraph
from dataclasses import dataclass
from typing_extensions import List

from krrood.entity_query_language.entity import (
    let, entity, Symbol,
)
from krrood.entity_query_language.entity_result_processors import the, a
from krrood.entity_query_language.match import (
    matching,
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

`matching(FixedConnection).from_(world.connections)` selects from `world.connections` items of type
`FixedConnection`. Inner `matching(...)` clauses describe constraints on attributes of that selected item.

```{code-cell} ipython3
fixed_connection_query = the(
    matching(FixedConnection)(
        parent=matching(Container)(name="Container1"),
        child=matching(Handle)(name="Handle1"),
    ).from_(world.connections)
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
- Bind domains with `.from_(domain)` on the outermost `matching(...)`.
- Nested `matching(...)` can be composed arbitrarily deep following your object graph.
- `matching(...).from_(...)` is syntactic sugar over the explicit `entity` + predicates form shown above.

## Selecting inner objects with `select()`

Use `select(...)` on variables you want returned. The evaluation then returns a unification dictionary
mapping variables to concrete objects.

```{code-cell} ipython3
from krrood.entity_query_language.match import select

fixed_connection = the(
    matching(FixedConnection)(
        parent=matching(Container)(name="Container1"),
        child=matching(Handle)(name="Handle1"),
    ).from_(world.connections)
)

container_and_handle = select(container := fixed_connection.parent,
                              handle := fixed_connection.child)

answers = container_and_handle.evaluate()
print(answers[container].name, answers[handle].name)
```

You can additionally filter the selection with `where(...)`:

```{code-cell} ipython3
fixed_connection2 = the(
    matching(FixedConnection)(
        parent=matching(Container),
        child=matching(Handle),
    ).from_(world.connections)
)

selected = select(c := fixed_connection2.parent, h := fixed_connection2.child).where(c.size > 1)
ans = selected.evaluate()
print(ans[c].name, ans[h].name)
```

## Matching collections with `match_any()` and `match_all()`

When matching a container-like attribute (for example, a list), use `match_any(values)` to express that
at least one element equals one of the provided values, or `match_all(values)` to require that all given
values are present in the collection.

```{code-cell} ipython3
from krrood.entity_query_language.match import match_any, match_all
from krrood.entity_query_language.entity_result_processors import a, the

# Build a simple set of views
drawer1 = Drawer(handle=h1, container=c1)
drawer2 = Drawer(handle=Handle("OtherHandle"), container=other_c)
cabinet1 = Cabinet(container=c1, drawers=[drawer1, drawer2])
cabinet2 = Cabinet(container=other_c, drawers=[drawer2])
views = [drawer1, drawer2, cabinet1, cabinet2]

# Existential: any of the given drawers is present
cabinet_any = a(matching(Cabinet)(drawers=match_any([drawer1, drawer2])).from_(views))
print(len(list(cabinet_any.evaluate())))  # 2

# Universal: all given drawers are present
cabinet_all = the(matching(Cabinet)(drawers=match_all([drawer2])).from_(views))
print(cabinet_all.evaluate())
```

## Distinct results

Remove duplicates with `distinct()`. On multi-variable selections you can pass one or more variables
to deduplicate on a subset of keys.

```{code-cell} ipython3
from krrood.entity_query_language.match import select

# Distinct on a single variable
names = ["Handle1", "Handle1", "Handle2", "Container1", "Container1", "Container3"]
name_var = a(matching(str).from_(names))
q1 = select(name_var).where(name_var.startswith("Handle")).distinct()
print(list(q1.evaluate()))  # ["Handle1", "Handle2"]

# Distinct on a pair
handle_names = ["Handle1", "Handle1", "Handle2"]
container_names = ["Container1", "Container1", "Container3"]
h = a(matching(str).from_(handle_names))
c = a(matching(str).from_(container_names))
pairs = select(h, c).distinct()
print(list(pairs.evaluate()))

# Distinct on a subset (only by handle)
pairs_by_handle = select(h, c).distinct(h)
print(list(pairs_by_handle.evaluate()))
```

## Ordering results

Use `order_by(variable=..., key=..., descending=...)` on a selection to order results. It composes with `distinct()`.

```{code-cell} ipython3
values = [5, 1, 1, 2, 1, 4, 3, 3, 5]
v = a(matching(int).from_(values))
ordered = select(v).distinct().order_by(variable=v, descending=False)
print(list(ordered.evaluate()))  # [1, 2, 3, 4, 5]

names2 = ["Handle1", "handle2", "Handle3", "container1", "Container2", "container3"]
n = a(matching(str).from_(names2))
key = lambda x: int(x[-1])
ordered2 = select(n).order_by(variable=n, key=key, descending=True)
print(list(ordered2.evaluate()))
```

## Aggregations and counts

You can use result processors like `count(...)` and `max(...)` together with matches and selections.

```{code-cell} ipython3
import krrood.entity_query_language.entity_result_processors as eql

world  # from above

# Count matching entities
fixed_connections = a(matching(FixedConnection).from_(world.connections))
print(eql.count(fixed_connections).evaluate())

# Compute max over an attribute
container = a(matching(Container).from_(world.bodies))
max_size = select(eql.max(container.size)).evaluate()
print(max_size)
```

## Error on unquantified matches

Accessing attributes on an unquantified match (without `the(...)` or `a(...)`) raises `UnquantifiedMatchError`.

```{code-cell} ipython3
from krrood.entity_query_language.failures import UnquantifiedMatchError

container_m = matching(Container).from_(world.bodies)
try:
    _ = container_m.size
except UnquantifiedMatchError:
    print("Caught UnquantifiedMatchError as expected")
```
