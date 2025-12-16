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
# Logical Operators

EQL supports a bunch of logical operators, namely `and_`, `or_`, `exists`, `for_all` and `not_`.
When you want to use these, you have to rely on the operators imported from EQL.entity, since the python operators
cannot be overloaded to the extent that EQL requires.

```{code-cell} ipython3
from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.entity import entity, or_, Symbol, variable, not_, and_
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
class World:
    id_: int
    bodies: List[Body]


# Build a small world
container1 = Container(name="Container1")
container2 = Container(name="Container2")
handle1 = Handle(name="Handle1")
handle2 = Handle(name="Handle2")
world = World(1, [container1, container2, handle1, handle2])
```

This way of writing `and`, `or` is exactly like constructing a tree which allows for the user to write in the same
structure as how the computation is done internally. Take note that whenever conditions are used in a query without an
explicit logical operator, `and` is assumed.

```{code-cell} ipython3
body = variable(type_=Body, domain=world.bodies)
query = an(entity(body).where(
                  or_(body.name.startswith("C"), body.name.endswith("1")),
                  or_(body.name.startswith("H"), body.name.endswith("1"))
                  )
           )
print(*query.evaluate(), sep="\n")
```

Universal, and existential conditionals are supported using `for_all` and `exists` respectively. These are mainly used
for dealing with collections and quantifying over them.

For example, lets add to our model two drawers and a cabinet like object.
```{code-cell} ipython3
from dataclasses import field

@dataclass
class View(Symbol):
    world: object = field(default=None, repr=False, kw_only=True)


@dataclass
class Drawer(View):
    handle: Handle
    container: Container


# A simple view-like class with an iterable attribute `drawers`
class CabinetLike(View):
    def __init__(self, drawers):
        super().__init__()
        self.drawers = list(drawers)


drawer1 = Drawer(handle=handle1, container=container1)
drawer2 = Drawer(handle=handle2, container=container2)
drawer3 = Drawer(handle=handle2, container=container1)
cabinet = CabinetLike([drawer1, drawer2])
world.views = [cabinet]
```

Now lets look for all drawers that are not part of any cabinet in the world.

```{code-cell} ipython3
from krrood.entity_query_language.entity import in_, for_all


# A variable ranging over drawers in the world
drawer = variable(Drawer, [drawer1, drawer2, drawer3])
views = variable(CabinetLike, world.views)
all_cabinets_drawers = views.drawers # A nested iterable where there is a list of views each with a list of drawers.
# Find drawers that are NOT in the list 
# (expected to find only the drawer3 since it is not part of any cabinet)
condition = for_all(all_cabinets_drawers, not_(in_(drawer, all_cabinets_drawers)))
non_cabinet_drawers_query = an(entity(drawer).where(condition))

found_non_cabinet_drawers = list(non_cabinet_drawers_query.evaluate())
assert len(found_non_cabinet_drawers) == 1
print(*found_non_cabinet_drawers, sep="\n")
```

Now if we look for drawers that are part of any cabinet using `exists` we should find the other two drawers.

```{code-cell} ipython3
from krrood.entity_query_language.entity import exists

# A variable ranging over drawers in the world
drawer = variable(Drawer, [drawer1, drawer2, drawer3])
cabinets = variable(CabinetLike, world.views)
all_cabinets_drawers = cabinets.drawers # A nested iterable where there is a list of views each with a list of drawers.
# Find drawers that are in the list
# (expected to find drawer1 and drawer2 since they are part of a cabinet)
condition = exists(drawer, in_(drawer, all_cabinets_drawers))
cabinet_drawers_query = an(entity(drawer).where(condition))
found_cabinet_drawers = list(cabinet_drawers_query.evaluate())
assert len(found_cabinet_drawers) == 2
print(*found_cabinet_drawers, sep="\n")
```

In EQL Negation is a filter that chooses only the False values of the expression that was negated.

```{code-cell} ipython3
body = variable(type_=Body, domain=world.bodies)
query = an(entity(body).where(
                  not_(or_(body.name.startswith("C"), body.name.endswith("1")),
                       )
                  )
           )
print(*query.evaluate(), sep="\n")
```

In some cases,
EQL tries to optimize the query when negation is used by replacing the original expression with an equivalent one that
is easier to compute, this happens for example when negating `exists(variable, expression)` it becomes `for_all(var, not_(expression))`.

```{code-cell} ipython3
body = variable(type_=Body, domain=world.bodies)
expression = not_(exists(body, body.name.startswith("A")))
print("exists(...) got translated to",type(expression))
```

EQL also optimizes what you mean by or_. Sometimes, it is more beneficial to treat the or statement as an `ElseIf` statement.
- `ElseIf` (else-if semantics) is used when both sides of `or_` reference the exact same set of non-literal symbolic variables; the right side is evaluated only if the left side is false for the current bindings.
- `Or` (union semantics) is used when the sides reference different variable sets (one introduces variables the other does not); both sides are evaluated and their solutions are unioned.

In other words: same variables → `ElseIf`; different variables → `Or` (Union).

An example for a query that gets optimized to `ElseIf` is

```{code-cell} ipython3
body = variable(type_=Body, domain=world.bodies)
or_expression = or_(
            body.name.startswith("C"),  # left uses {body}
            body.name.endswith("1"),  # right uses {body}
        )
print(type(or_expression))
```

And here is one where an actual union is performed.

```{code-cell} ipython3
body = variable(type_=Body, domain=world.bodies)
other = variable(type_=Body, domain=world.bodies)
or_expression = or_(
            body.name.startswith("C"),
            # Introduces `other`, so the variable sets differ → treated as Union
            and_(body.name == other.name, other.name.endswith("2")),
        )
print(type(or_expression))
```
