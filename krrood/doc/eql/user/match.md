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

EQL provides a powerful and concise API for building nested structural queries using `an()` (and
`the()`). When you pass a **type** to `an()`/`the()`, it builds a structural *match* — letting you
describe complex object relationships in a declarative way that mirrors the structure of your data.
(When you pass a *symbolic expression* instead, `an()`/`the()` act as result quantifiers — see
[Result Quantifiers](result_quantifiers.md).)

## Describing a pattern with `an(Type)`

Calling `an()` with a *type* describes a pattern for an object's attributes. You can specify both
the expected type and the values for its fields.

```python
from krrood.entity_query_language.factories import an

# Describe a robot named 'R2D2'
robot_pattern = an(ExampleRobot)(name="R2D2")
```

## Binding to a domain with `.from_()`

Chaining `.from_(domain)` after the call binds the match to a specific set of candidate objects.

```python
from krrood.entity_query_language.factories import an, a

# Create a variable for any 'Robot' in the 'world.robots' domain that matches the pattern
r = an(ExampleRobot)(name="R2D2").from_(world.robots)
```

```{hint}
Use `an(Type)(...).from_(domain)` as the entry point for your structural queries when you need to
specify a domain, and `an(Type)` for nested child attributes. Use `the(...)` instead of `an(...)`
when exactly one solution is expected.
```

## Nested Matching

The real power of the match API comes from nesting. You can describe deeply nested object graphs in a single expression.

```python
# Match a connection whose parent is a ExampleContainer named 'C1' and child is a ExampleHandle named 'H1'
fixed_connection = a(FixedConnection)(
    parent=an(ExampleContainer)(name="C1"),
    child=an(ExampleHandle)(name="H1")
).from_(world.connections)
```

```{note}
`.from_(domain)` is syntactic sugar. Under the hood, it sets the match's domain, which is later used
to create a {py:func}`~krrood.entity_query_language.factories.variable`,
 selects it using {py:func}`~krrood.entity_query_language.factories.entity` and automatically adds the corresponding {py:meth}`~krrood.entity_query_language.query.query.Query.where` clauses.
```

## Full Example: Finding Connected Parts

This example demonstrates how to find a complex structural relationship using nested matches.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import an, entity, the, Symbol

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

# 1. Define the structural match (exactly one expected, so build it with `the`)
# We are looking for a connection between 'Bin' and 'Grip'
conn = the(ExampleConnection)(
    parent=an(ExampleContainer)(name="Bin"),
    child=an(ExampleHandle)(name="Grip")
).from_(world_connections)

# 2. Execute the query
result = conn.first()

print(f"Found connection: {result.parent.name} <-> {result.child.name}")
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.an`
- {py:func}`~krrood.entity_query_language.factories.the`
- {py:class}`~krrood.entity_query_language.query.match.Match`
