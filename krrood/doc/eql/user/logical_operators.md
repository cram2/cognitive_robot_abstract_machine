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

EQL provides intuitive ways to combine multiple constraints using logical operators. These allow you to build complex
filters beyond simple attribute checks.

## The Conjunction (AND)

You can combine conditions using the `and_()` operator or by passing multiple arguments to `.where()`.
Both methods are equivalent.

### 1. Multiple conditions in `.where()`
```python
# Select robots that are named 'R2D2' AND have battery > 50
query = entity(r).where(r.name == "R2D2", r.battery > 50)
```

### 2. Using the `and_()` operator
```python
# This produces the same result
query = entity(r).where(and_(r.name == "R2D2", r.battery > 50))
```

```{hint}
Using multiple arguments in `.where()` is generally cleaner for simple (unnested) conjunctions.
```

## The Disjunction (OR)

Use the `or_()` operator to specify that at least one of the conditions must be met.

```python
# Select robots that are either 'R2D2' OR have battery < 10
query = entity(r).where(or_(r.name == "R2D2", r.battery < 10))
```

## The Negation (NOT)

The `not_()` operator inverts a condition. It returns results that do **not** satisfy the specified constraint.

```python
# Select all robots EXCEPT those named 'R2D2'
query = entity(r).where(not_(r.name == "R2D2"))
```

```{note}
Negation can be particularly useful for "anti-joins" or excluding specific subsets from your results.
```

## Full Example: Complex Logic

Let's build a query that combines all these operators.

[//]: # (```{code-cell} ipython3)
```python3
from dataclasses import dataclass
from krrood.entity_query_language.factories import variable, entity, an, Symbol, not_

@dataclass
class ExampleRobot(Symbol):
    name: str
    battery: int
    online: bool

robots = [
    ExampleRobot("R2D2", 100, True),
    ExampleRobot("C3PO", 20, False),
    ExampleRobot("BB8", 80, True),
    ExampleRobot("Gonk", 5, True)
]

r = variable(ExampleRobot, domain=robots)

# We want robots that are (ONLINE and (battery > 50)) OR (NOT ONLINE and battery < 30)
query = an(entity(r).where(
    or_(and_(r.online , r.battery > 50), and_(not_(r.online) , r.battery < 30))
))

for robot in query.evaluate():
    print(f"Robot: {robot.name} (Online: {robot.online}, Battery: {robot.battery})")
```

## API Reference
- {py:class}`~krrood.entity_query_language.operators.core_logical_operators.AND`
- {py:class}`~krrood.entity_query_language.operators.core_logical_operators.OR`
- {py:class}`~krrood.entity_query_language.operators.core_logical_operators.Not`
