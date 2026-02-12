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
from __future__ import annotations
from dataclasses import dataclass
from typing_extensions import List, Optional

import krrood.entity_query_language.entity_result_processors as eql
from krrood.entity_query_language.entity_result_processors import a, an
from krrood.entity_query_language.entity import entity, variable, contains, set_of, distinct


@dataclass
class Body:
    name: str
    height: int
    world: Optional[World] = None


@dataclass
class World:
    bodies: List[Body]
    
    def __post_init__(self):
        for body in self.bodies:
            body.world = self


world = World([
    Body("Handle1", 1),
    Body("Handle2", 2),
    Body("Container1", 3),
    Body("Container2", 4),
    Body("Container3", 5),
])
```

## Evaluation

The result processors can be evaluated using `.evaluate()` which returns a lazy iterator over the results. `evaluate()` 
 of the result quantifier `an` takes an optional `limit` argument to limit the number of results returned.

```{code-cell} ipython3
from krrood.entity_query_language.failures import NonPositiveLimitValue

body = variable(type_=Body, domain=world.bodies)
query = an(entity(body).where(contains(body.name, "Handle")))
print(len(list(query.evaluate()))) # -> 2 results (all handle bodies)
print(len(list(query.evaluate(limit=2)))) # -> 2 results
print(len(list(query.evaluate(limit=1)))) # -> 1 result
print(len(list(query.evaluate(limit=3)))) # -> 2 results (only 2 total results available, limit is not a hard limit but a maximum)
try:
    list(query.evaluate(limit=0))
except NonPositiveLimitValue as e:
    print(e) # -> NonPositiveLimitValue: 0, limit must be a positive integer
```

## Aggregators

The core aggregators are `count`, `sum`, `average`, `max`, and `min`. All aggregators support:
- `key`: A function to extract or transform values before aggregation.
- `default`: The value returned if the result set is empty.
- `distinct`: A boolean (default `False`) that, if `True`, ensures only unique values are considered during aggregation.

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

Aggregators can now be grouped by one or more variables using the `.grouped_by()` method.
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

### Distinct Aggregation

Aggregators support a `distinct=True` keyword argument to perform the aggregation only over unique values within each group.

```{code-cell} ipython3
# Example world with duplicate height values
world = World([
    Body("Handle1", 10),
    Body("Handle2", 10),
    Body("Container1", 20),
])
body = variable(Body, domain=world.bodies)

# Regular sum: 10 + 10 + 20 = 40
total_height = eql.sum(body.height).tolist()[0]
print(total_height)

# Distinct sum: 10 + 20 = 30
distinct_total_height = eql.sum(body.height, distinct=True).tolist()[0]
print(distinct_total_height)
```

#### Difference between `distinct=True` and `distinct()` function

It is important to distinguish between the `distinct=True` keyword argument in an aggregator and the `distinct()` function applied to an aggregated query:

1.  **`eql.sum(..., distinct=True)`**: Applies distinctness to the **input values** of the aggregation (e.g., sum of unique heights).
2.  **`distinct(eql.sum(...).grouped_by(...))`**: Applies distinctness to the **results** of the aggregation (e.g., unique sum values across different groups).

```{code-cell} ipython3
from krrood.entity_query_language.entity import concatenate

# Finding unique total heights grouped by worlds
world2 = World([Body("Handle1", 10),
    Body("Handle2", 10),
    Body("Container1", 20)])
# concatenate bodies from both worlds
body2 = variable(Body, domain=world2.bodies)
body = concatenate(body, body2)

# without distinct()
total_sum_by_world = eql.sum(body.height).grouped_by(body.world).tolist() # -> [40, 40]
print(total_sum_by_world)

# with distinct()
unique_sums = distinct(eql.sum(body.height).grouped_by(body.world)).tolist() # -> [40]
print(unique_sums)
```

## Features and Constraints

- **Nested Aggregations**: Aggregators cannot be directly nested (e.g., `eql.max(eql.count(v))` is invalid). However, you can aggregate over a grouped query using `eql.max(eql.count(v).grouped_by(g))`.
- **Selection Consistency**: If any aggregator is selected in a `set_of`, all other selected variables must be included in the `grouped_by` clause.
- **Where vs. Having**: `where` filters individual rows before aggregation; `having` filters groups after aggregation. Aggregators are not allowed in `where` clauses.
- **Distinct**: Distinctness makes use of the hash and equality of the values. When using `distinct=True/distinct()`, ensure that
the values being distinct have appropriate `__hash__` and `__eq__` implementations for correct behavior.
- **grouped_by()**: Similar to Distinct, ensure that the grouping variables have appropriate `__hash__` and `__eq__`
implementations for correct behavior.

