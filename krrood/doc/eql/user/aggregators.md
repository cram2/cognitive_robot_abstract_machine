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

# Aggregation and Grouping

EQL supports powerful aggregation functions that allow you to summarize data across groups of entities. This includes common operations like counting, summing, and averaging.

## Grouping Results

The `.grouped_by()` method allows you to group results by one or more symbolic variables. When you use an aggregator in a query, EQL automatically calculates it for each group.

```python
from krrood.entity_query_language.factories import entity
import krrood.entity_query_language.factories as eql

# Group robots by their 'type' and count them by type, returning a count for the occurrences of each type.
query = eql.count(r.type).grouped_by(r.type)
```

## Using Aggregators

Aggregators are functions that take a symbolic variable and return a summary value.

Available aggregators include:
- `count()`: Counts the number of items in the group.
- `sum()`: Calculates the sum of a numeric attribute.
- `average()`: Calculates the average of a numeric attribute.
- `max()` and `min()`: Find the maximum or minimum value.

Most aggregators support these optional parameters:
- `key`: A function to extract the value for aggregation/comparison from the object.
- `default`: The value to return if the group is empty.
- `distinct`: Whether to consider only unique values.

```python
from krrood.entity_query_language.factories import sum, average

# Calculate the total and average battery level per robot type
query = set_of(r.type, sum(r.battery), average(r.battery)).grouped_by(r.type)
```

```{hint}
You can use `.distinct()` inside an aggregator to count only unique values: `count(r.name, distinct=True)`.
```

## Finding Extremes with `max()` and `min()`

The `max()` and `min()` aggregators find the extreme values in a group. While they often operate on numeric attributes, you can use the `key` argument to find the object that has an extreme property.

```python
import krrood.entity_query_language.factories as eql

# Find the maximum battery level
query = eql.max(r.battery)

# Find the robot object with the highest battery level
# This returns the ExampleRobot instance itself, not just the battery number
query = eql.max(r, key=lambda robot: robot.battery)
```

## Post-Aggregation Filtering with `.having()`

While `.where()` filters individual entities *before* they are grouped, `.having()` filters the results *after* they have been aggregated.

```python
# Only show types that have more than 5 robots
query = entity(r.type).grouped_by(r.type).having(count(r) > 5)
```

```{warning}
Always use `.where()` for conditions that can be evaluated on individual objects. Use `.having()` only
for conditions that depend on group-level aggregates.
```

## Full Example: ExampleWorld Statistics

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import variable, set_of, Symbol, count, sum, average

@dataclass
class ExampleRobot(Symbol):
    name: str
    type: str
    battery: int

robots = [
    ExampleRobot("R1", "Astromech", 100),
    ExampleRobot("R2", "Astromech", 80),
    ExampleRobot("C1", "Protocol", 20),
    ExampleRobot("C2", "Protocol", 40),
    ExampleRobot("K1", "Security", 90)
]

r = variable(ExampleRobot, domain=robots)

# We want to see:
# 1. The type of robot
# 2. How many of each type (count)
# 3. The average battery for that type
# BUT only for types where the total battery sum > 50
query = set_of(r.type, c:=count(r), avg_batt:=average(r.battery)) \
    .grouped_by(r.type) \
    .having(sum(r.battery) > 50)

for result in query.evaluate():
    print(f"Type: {result[r.type]} | Count: {result[c]} | Avg Battery: {result[avg_batt]}%")
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.count`
- {py:func}`~krrood.entity_query_language.factories.sum`
- {py:func}`~krrood.entity_query_language.factories.average`
- {py:func}`~krrood.entity_query_language.factories.max`
- {py:func}`~krrood.entity_query_language.factories.min`
- {py:meth}`~krrood.entity_query_language.query.query.Query.grouped_by`
- {py:meth}`~krrood.entity_query_language.query.query.Query.having`
