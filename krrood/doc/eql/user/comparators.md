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

# Comparisons and Membership

EQL supports all standard Python comparison operators and adds powerful membership checks for querying containers and lists.

## Standard Comparisons

You can use the following operators directly on symbolic attributes:
- **Equality**: `==` and `!=`
- **Ordering**: `<`, `<=`, `>`, `>=`

```python
# Select robots with high battery
query = entity(r).where(r.battery >= 80)
```

```{hint}
EQL automatically handles type coercion where appropriate, making your queries more resilient to minor data differences.
```

## Membership Checks

EQL provides two main ways to check for membership in collections: `in_()` and `contains()`.

### 1. The `in_()` Operator
Use `in_()` to check if a value exists within a specific collection.

```python
from krrood.entity_query_language.factories import in_

# Select robots whose names are in a specific list
allowed_names = ["R2D2", "BB8"]
query = entity(r).where(in_(r.name, allowed_names))
```

### 2. The `contains()` Operator
Use `contains()` to check if a symbolic attribute (which is a collection) contains a specific value.

```python
from krrood.entity_query_language.factories import contains

# Select robots that have 'laser' in their equipment list
query = entity(r).where(contains(r.equipment, "laser"))
```

```{warning}
Python's native `in` operator (e.g., `value in container`) cannot be overridden to return a symbolic expression. You **must** use EQL's `in_()` or `contains()` functions for these checks inside a query.
```

## Full Example: Filtering by Categories and Capabilities

Let's build a query that uses both standard comparisons and membership checks.

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List
from krrood.entity_query_language.factories import variable, entity, an, Symbol, in_, contains

@dataclass
class ExampleRobot(Symbol):
    name: str
    battery: int
    tags: List[str]

robots = [
    ExampleRobot("R2D2", 100, ["astromech", "repair"]),
    ExampleRobot("C3PO", 20, ["protocol", "translator"]),
    ExampleRobot("K2SO", 80, ["security", "combat"])
]

r = variable(ExampleRobot, domain=robots)

# We want robots that:
# 1. Have at least 50% battery
# 2. Are in either the "security" or "protocol" category (using in_)
# 3. Have the "combat" tag (using contains)
query = an(entity(r).where(
    r.battery >= 50,
    contains(r.tags, "combat")
))

for robot in query.evaluate():
    print(f"Match: {robot.name} (Tags: {robot.tags})")
```

## API Reference
- {py:class}`~krrood.entity_query_language.operators.comparator.Comparator`
- {py:func}`~krrood.entity_query_language.factories.in_`
- {py:func}`~krrood.entity_query_language.factories.contains`
