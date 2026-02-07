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

Result processors in EQL are mappings applied to results produced from a query or variable. They support enhanced grouping, ordering, and convenient result retrieval methods.

Currently, there are two kinds of result processors:

- Aggregators: `count`, `sum`, `average`, `max`, and `min`.
- Result Quantifiers: `the`, `a/an`, etc. See the dedicated page for details: {doc}`result_quantifiers`.

All result processors are evaluatable: they return a query object that exposes `.evaluate()`.

```{note}
You can pass either a variable created with `variable(...)` directly, or wrap it with `entity(...)`. Both forms are supported by the aggregators demonstrated below.
```

## Setup

```{code-cell} ipython3
from dataclasses import dataclass
from typing_extensions import List

import krrood.entity_query_language.entity_result_processors as eql
from krrood.entity_query_language.entity_result_processors import a, an
from krrood.entity_query_language.entity import entity, variable, contains, set_of


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

## Aggregators

The core aggregators are `count`, `sum`, `average`, `max`, and `min`. All aggregators support `key` (to extract numeric values) and `default` (returned if the result set is empty).

### count

Count the number of results matching a predicate.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

query = eql.count(
    entity(
        body).where(
        contains(body.name, "Handle"),
    )
)

print(query.tolist()[0])  # -> 2
```

You can also use `count()` without arguments to count the number of results in a group.
This is useful when you want to count all entities matching the group's conditions.

```{code-cell} ipython3
query = set_of(first_char := body.name[0], total := eql.count()).grouped_by(first_char)

for res in query.tolist():
    print(f"Group: {res[first_char]}, Count: {res[total]}")
```

### sum

Sum numeric values from the results. You can provide a `key` function to extract the numeric value from the results.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

query = eql.sum(body, key=lambda b: b.height)
print(query.tolist()[0])  # -> 15
```

If there are no results, `sum` returns `None` by default. You can specify a `default` value.

```{code-cell} ipython3
empty = variable(int, domain=[])
query = eql.sum(empty, default=0)
print(query.tolist()[0])  # -> 0
```

### average

Compute the arithmetic mean of numeric values. Like `sum`, it supports `key` and `default`.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)
query = eql.average(body, key=lambda b: b.height)
print(query.tolist()[0])  # -> 3.0
```

### max and min

Find the maximum or minimum value. These also support `key` and `default`.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

max_query = eql.max(body, key=lambda b: b.height)
min_query = eql.min(body, key=lambda b: b.height)

print(max_query.tolist()[0])  # -> Body(name='Container3', height=5)
print(min_query.tolist()[0])  # -> Body(name='Handle1', height=1)
```

## Grouping with `.grouped_by()`

Aggregators can now be grouped by one or more variables using the `.grouped_by()` method (which replaces the older `.per()` syntax).
When `.grouped_by()` is used with multiple selected variables in a `set_of`, each result is a dictionary mapping the variables to their values.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

# Grouping by the first character of the body name
# Use set_of to select both the group value and the aggregated result
query = set_of(first_char := body.name[0], count := eql.count(body)).grouped_by(first_char)
results = query.tolist() 

for res in results:
    # Results are returned as UnificationDicts
    group_value = res[first_char]
    count_value = res[count] 
    print(f"First Character: {group_value}, Count: {count_value}")
```

## The `having()` clause

You can filter aggregated results using `.having()`. Note that `having` must be used after `where` and can only contain conditions involving aggregators.

```{code-cell} ipython3
# Find groups with an average height greater than 3
query = set_of(first_char := body.name[0], avg_height := eql.average(body.height)) \
    .grouped_by(first_char) \
    .having(avg_height > 3)

for res in query.tolist():
    print(f"Group: {res[first_char]}, Average Height: {res[avg_height]}")
```

## Ordering with `.order_by()`

Query objects now support ordering of results.

```{code-cell} ipython3
# Order bodies by height in descending order
query = set_of(body).order_by(body.height, descending=True)
sorted_bodies = query.tolist()
for b in sorted_bodies:
    print(b)
```

## Multiple Aggregations

You can select multiple aggregations in a single query by using `set_of`. This is useful for computing several statistics at once for each group.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

query = set_of(
    first_char := body.name[0],
    avg_h := eql.average(body.height),
    max_h := eql.max(body.height),
    total := eql.count()
).grouped_by(first_char)

for res in query.tolist():
    print(f"Group {res[first_char]}: Avg={res[avg_h]}, Max={res[max_h]}, Count={res[total]}")
```

## Features and Syntax Constraints

- **Nested Aggregations**: Aggregators cannot be directly nested (e.g., `eql.max(eql.count(v))` is invalid). However, you can aggregate over a grouped query using `eql.max(eql.count(v).grouped_by(g))`.
- **Selection Consistency**: If any aggregator is selected in a `set_of`, all other selected variables must be included in the `grouped_by` clause.
- **Where vs. Having**: `where` filters individual rows before aggregation; `having` filters groups after aggregation. Aggregators are not allowed in `where` clauses.


