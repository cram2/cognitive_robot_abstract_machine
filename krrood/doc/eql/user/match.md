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

# Structural Pattern Matching

EQL provides a powerful and concise API for building nested structural queries using `match_variable` and `match`. This allows you to describe complex object relationships in a declarative way that mirrors the structure of your data.

## The `match()` Function

The `match()` function describes a pattern for an object's attributes. You can specify both the expected type and the values for its fields.

```python
from krrood.entity_query_language.factories import match

# Describe a robot named 'R2D2'
robot_pattern = match(ExampleRobot)(name="R2D2")
```

## The `match_variable()` Function

While `match()` describes a pattern, `match_variable()` in addition binds to a specific domain.

```python
from krrood.entity_query_language.factories import match_variable

# Create a variable for any 'Robot' in the 'world.robots' domain that matches the pattern
r = match_variable(ExampleRobot, domain=world.robots)(name="R2D2")
```

```{hint}
Use `match_variable` as the entry point for your structural queries when needing to specify a domain, and
use `match` for nested child attributes.
```

## Nested Matching

The real power of the match API comes from nesting. You can describe deeply nested object graphs in a single expression.

```python
# Match a connection whose parent is a ExampleContainer named 'C1' and child is a ExampleHandle named 'H1'
fixed_connection = match_variable(FixedConnection, domain=world.connections)(
    parent=match(ExampleContainer)(name="C1"),
    child=match(ExampleHandle)(name="H1")
)
```

```{note}
`match_variable` is syntactic sugar. Under the hood, it creates a {py:func}`~krrood.entity_query_language.factories.variable`
and automatically adds the corresponding {py:meth}`~krrood.entity_query_language.query.query.Query.where` clauses.
```

```{warning}
Use `match_variable` instead of `variable` only when you have multiple attribute checks. For simple queries, `variable().where(...)` is often more readable.
```

## Full Example: Finding Connected Parts

This example demonstrates how to find a complex structural relationship using nested matches.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import match_variable, match, entity, the, Symbol

@dataclass
class ExampleBody(Symbol):
    name: str

@dataclass
class ExampleContainer(ExampleBody):
    pass

@dataclass
class ExampleHandle(ExampleBody):
    pass

@dataclass
class ExampleConnection(Symbol):
    parent: ExampleBody
    child: ExampleBody

# Data
c1, h1 = ExampleContainer("Bin"), ExampleHandle("Grip")
world_connections = [ExampleConnection(c1, h1)]

# 1. Define the structural match
# We are looking for a connection between 'Bin' and 'Grip'
conn = match_variable(ExampleConnection, domain=world_connections)(
    parent=match(ExampleContainer)(name="Bin"),
    child=match(ExampleHandle)(name="Grip")
)

# 2. Build and execute the query
query = the(entity(conn))
result = query.first()

print(f"Found connection: {result.parent.name} <-> {result.child.name}")
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.match`
- {py:func}`~krrood.entity_query_language.factories.match_variable`
- {py:class}`~krrood.entity_query_language.query.match.Match`
