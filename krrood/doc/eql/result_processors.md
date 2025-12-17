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

# Result Processors

Result processors in EQL are mappings that are applied to the results produced from a query/variable. Common processors are: `count`, `sum`, `average`, `max`, and `min`.

Result quantifiers like `the` and `an` are also a kind of result processor. See the dedicated page for details: {doc}`result_quantifiers`.

All result processors are evaluatable: they return a query object that exposes `.evaluate(...)`.

```{note}
You can pass either a variable created with `variable(...)` directly, or wrap it with `entity(...)`. Both forms are supported by the processors demonstrated below.
```

## Setup

```{code-cell} ipython3
from dataclasses import dataclass
from typing_extensions import List

import krrood.entity_query_language.entity_result_processors as eql
from krrood.entity_query_language.entity import entity, variable, contains


@dataclass
class Body:
    name: str
    height: int


@dataclass
class World:
    bodies: List[Body]


world = World([
    Body("Handle1", 1),
    Body("Handle2", 2),
    Body("Container1", 3),
    Body("Container2", 4),
    Body("Container3", 5),
])
```

## count

Count the number of results matching a predicate.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

query = eql.count(
    entity(
        body).where(
        contains(body.name, "Handle"),
    )
)

print(query.evaluate())  # -> 2
```

You can also count over a variable directly (without `entity(...)`).

```{code-cell} ipython3
query = eql.count(variable(Body, domain=world.bodies))
print(query.evaluate())  # -> 5
```

## sum

Sum numeric values from the results.

```{code-cell} ipython3
heights = [1, 2, 3, 4, 5]
value = variable(int, domain=heights)

query = eql.sum(entity(value))
print(query.evaluate())  # -> 15
```

If there are no results, `sum` returns `None`.

```{code-cell} ipython3
empty = variable(int, domain=[])
query = eql.sum(entity(empty))
print(query.evaluate())  # -> None
```

You can also sum without wrapping in `entity(...)`:

```{code-cell} ipython3
query = eql.sum(value)
print(query.evaluate())  # -> 15
```

## average

Compute the arithmetic mean of numeric values.

```{code-cell} ipython3
value = variable(int, domain=[1, 2, 3, 4, 5])
query = eql.average(entity(value))
print(query.evaluate())  # -> 3.0
```

## max and min

Find the maximum or minimum value.

```{code-cell} ipython3
values = [10, 7, 12, 3]
value = variable(int, domain=values)

max_query = eql.max(entity(value))
min_query = eql.min(entity(value))

print(max_query.evaluate())  # -> 12
print(min_query.evaluate())  # -> 3
```

Both also work without `entity(...)`:

```{code-cell} ipython3
print(eql.max(value).evaluate())  # -> 12
print(eql.min(value).evaluate())  # -> 3
```

## Result quantifiers are processors too

Result quantifiers such as `the` and `an` also process results. They quantify how many solutions exist and are therefore a kind of result processor. See {doc}`result_quantifiers` for usage and constraints on result counts.

## Evaluation

Every result processor is evaluatable. Call `.evaluate(...)` to obtain the final value.
