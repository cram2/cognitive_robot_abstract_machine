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

# Comparators

EQL features a bunch of comparators built into the language: `==`, `!=`, `<`, `<=`, `>`, `>=`, and membership via `in`/`contains`.
They are optimized and can be negated and composed with logical operators.

```{code-cell} ipython3
from dataclasses import dataclass
from typing_extensions import List

from krrood.entity_query_language.entity import (
    entity, variable, Symbol,
    in_, contains, not_, and_, or_,
)
from krrood.entity_query_language.entity_result_processors import an


@dataclass
class Body(Symbol):
    name: str
    weight: int
    tags: List[str]


@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(
    1,
    [
        Body("Container1", 10, ["metal", "blue"]),
        Body("Container2", 15, ["plastic", "red"]),
        Body("Handle1", 5, ["metal"]),
        Body("Handle2", 7, ["wood", "brown"]),
    ],
)
```


## Equality and inequality: `==` and `!=`

Use Python’s comparison operators. EQL overloads these on symbolic variables to produce comparator nodes.

```{code-cell} ipython3
b = variable(Body, domain=world.bodies)
query = an(entity(b).where(b.name == "Container1"))

print(*query.evaluate(), sep="\n")
```

Inequality `!=` works similarly:

```{code-cell} ipython3
b = variable(Body, domain=world.bodies)
query = an(entity(b).where(b.name != "Container1"))

print(*query.evaluate(), sep="\n")
# => all bodies except the one with name == 'Container1'
```

You can compare attributes between two variables as well:

```{code-cell} ipython3
left = variable(Body, domain=world.bodies)
right = variable(Body, domain=world.bodies)
# Same name, but different instances allowed by domain (not enforced here)
query = an(entity(left).where(left.name == right.name))

print(*query.evaluate(), sep="\n")
```


## Ordering: `<`, `<=`, `>`, `>=`

These work for numeric and comparable attributes.

```{code-cell} ipython3
b = variable(Body, domain=world.bodies)
heavy = an(entity(b).where(b.weight >= 10))

print(*heavy.evaluate(), sep="\n")
# => bodies with weight >= 10
```

Chaining with logical operators (implicit AND when multiple conditions are given):

```{code-cell} ipython3
b = variable(Body, domain=world.bodies)
query = an(
    entity(
        b).where(
        b.weight >= 10,
        b.name.startswith("C"),  # attribute/property comparisons can be mixed
    )
)

print(*query.evaluate(), sep="\n")
```


## Membership: `contains`, and `in_`

Membership has to be checked using EQL operators `in_(item, container)` or `contains(container, item)`.
Writing `item in literal_list` will be evaluated immediately by Python and not produce a symbolic comparator.


```{code-cell} ipython3
b = variable(Body, domain=world.bodies)
query = an(entity(b).where(in_(b.name, {"Container1", "Handle1"})))
print(*query.evaluate(), sep="\n")

b = variable(Body, domain=world.bodies)
query = an(entity(b).where(contains({"metal", "wood"}, b.tags[0])))
print(*query.evaluate(), sep="\n")
```

Tip: Membership works with any container type whose Python `operator.contains` is defined (lists, sets, strings, etc.). For strings, you can check substrings: `"Con" in b.name`.


## Evaluation order and performance notes

- Comparators are represented by the `Comparator` node in the expression tree. EQL automatically reorders which side gets evaluated first based on currently bound variables to reduce search space.
- When you negate a comparator (`not_`), EQL swaps the underlying Python operation (for example, `==` becomes `!=`) to keep the expression efficient instead of post-filtering results.
- Membership over large literal containers is efficient when you use `in_`/`contains` since it remains a symbolic node and can benefit from caching.
