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
# Domain Mapping

Domain mapping transforms iterable attributes or nested collections into element-wise bindings while preserving existing
variable bindings. This page covers two common patterns:

- Flattening an iterable attribute (`flatten`)
- Indexing into container attributes (capturing `__getitem__` symbolically)

## Setup

We first define a small world used by both examples below.

```{code-cell} ipython3
from dataclasses import dataclass, field
from typing_extensions import List, Dict

from krrood.entity_query_language.entity import (
    entity,
    set_of,
    variable,
    flatten,
    Symbol,
)
from krrood.entity_query_language.entity_result_processors import an

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
class View(Symbol):
    world: object = field(default=None, repr=False, kw_only=True)


@dataclass
class Drawer(View):
    handle: Handle
    container: Container


@dataclass
class World(Symbol):
    id_: int
    bodies: List[Body] = field(default_factory=list)
    views: List[View] = field(default_factory=list)


# Build a small world
world = World(1)
container1 = Container(name="Container1")
container3 = Container(name="Container3")
handle1 = Handle(name="Handle1")
handle3 = Handle(name="Handle3")
world.bodies.extend([container1, container3, handle1, handle3])

# Two drawers
drawer1 = Drawer(handle=handle1, container=container1)
drawer2 = Drawer(handle=handle3, container=container3)

# A simple view-like class with an iterable attribute `drawers`
class CabinetLike(View):
    def __init__(self, drawers):
        super().__init__()
        self.drawers = list(drawers)


cabinet = CabinetLike([drawer1, drawer2])
world.views = [cabinet]
```

## Flatten iterable attributes

Flatten turns an iterable-of-iterables into a flat sequence of items while keeping the original parent binding
(similar to [SQL UNNEST](https://www.postgresql.org/docs/current/functions-array.html#:~:text=unnest%20(%20anyarray%20)%20%E2%86%92%20setof%20anyelement)).
It is handy when a selected variable has an attribute that is a list and you want one row per element of that list.

```{code-cell} ipython3
views = variable(type_=View, domain=world.views)
drawers = flatten(views.drawers)  # UNNEST-like flatten of each view's drawers
query = an(set_of(views, drawers))

rows = list(query.evaluate())
# Each solution contains both the parent view and one flattened drawer
assert len(rows) == 2
assert {r[drawers].handle.name for r in rows} == {"Handle1", "Handle3"}
assert all(r[views] is cabinet for r in rows)
print(*map(lambda r: r[drawers], rows), sep="\n")
```

Notes:
- `flatten` works on any expression that yields an iterable (for example, an attribute like `views.drawers`).
- Each solution produced by `flatten` retains the original bindings (here, `views`), so they can be used in further constraints or selections.

## Indexing into container attributes

Indexing on symbolic variables is captured in the expression graph. You can index into containers (such as dictionaries or lists) held by your symbolic variable, and the operation is represented symbolically within the query.

```{code-cell} ipython3
@dataclass
class ScoredBody(Symbol):
    name: str
    props: Dict[str, int]


@dataclass
class ScoreWorld(Symbol):
    bodies: List[ScoredBody]


score_world = ScoreWorld([
    ScoredBody("Body1", {"score": 1}),
    ScoredBody("Body2", {"score": 2}),
])

b = variable(type_=ScoredBody, domain=score_world.bodies)
# Use indexing on a dict field; the indexing is preserved symbolically
query = an(entity(b).where(b.props["score"] == 2))

results = list(query.evaluate())
assert len(results) == 1 and results[0].name == "Body2"
print(*results, sep="\n")
```
