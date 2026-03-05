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

# Mapped Variables and Attribute Access

Mapped variables allow you to derive new symbolic expressions from existing variables. Instead of defining a new variable for every piece of data, you can navigate your object graph using standard Python syntax like attribute access, indexing, and function calls.

## Attribute Access

The most common mapping is accessing an attribute of an object. EQL captures these accesses and evaluates them during query execution.

```python
from krrood.entity_query_language.factories import variable

robot = variable(ExampleRobot, domain=my_robots)
# Accessing the 'name' attribute symbolically
name_attr = robot.name # -> returns Attribute(robot, "name")
```

```{hint}
You can chain attribute accesses, e.g., `robot.parent.name`. EQL will handle the traversal automatically.
```

## Index and Key Access

If a variable represents a list or a dictionary, you can use standard indexing or key access.

```python
# Accessing the first element of a list
first_part = robot.parts[0] # -> returns IndexedVariable(robot.parts, 0)

# Accessing a value by key in a dictionary
config_val = robot.config["battery_limit"] # -> returns IndexedVariable(robot.config, "battery_limit")
```

## Flattening Collections

When an attribute returns a collection (like a list), you might want to treat each element as an individual result. The `flat_variable()` function "unpacks" these collections.

```python
from krrood.entity_query_language.factories import flat_variable

# robot.parts is a List[ExamplePart]. We want to iterate over each part. Since robot itself is a variable, if we just do
# robot.parts, we will get for each item of this iterable a list of parts, but we want instead to get for each item of
# the iterable a single part. And that's what flat_variable does.
part = flat_variable(robot.parts) # -> returns FlatVariable(robot.parts)
```

```{note}
`flat_variable` behaves similarly to a `JOIN` or `UNNEST` in SQL, creating a new solution for every element in the 
collection while keeping the original variable bindings.
```

## Symbolic Method Calls

You can also call methods on symbolic variables. These calls are deferred and executed when the query is evaluated.

```python
# Calling a method on the robot variable
status = robot.get_status(time.now()) # -> returns Call(robot, "get_status", [time.now()])
```

```{warning}
The method must exist on the underlying objects in the domain. If the method takes arguments, those arguments can also
 be symbolic variables!
```

## Full Example: Mapping and Flattening

This example shows how to navigate from a `ExampleWorld` to individual `ExamplePart` objects using attribute access
and flattening.

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List
from krrood.entity_query_language.factories import variable, entity, an, flat_variable, Symbol

@dataclass
class ExamplePart(Symbol):
    name: str

@dataclass
class ExampleRobot(Symbol):
    name: str
    parts: List[ExamplePart]

# Data setup
p1, p2 = ExamplePart("Arm"), ExamplePart("Leg")
robot = ExampleRobot("R2D2", [p1, p2])
my_robots = [robot]

# 1. Define the base variable
r = variable(ExampleRobot, domain=my_robots)

# 2. Use attribute access and flat_variable to reach the parts
p = flat_variable(r.parts)

# 3. Build a query to find parts belonging to 'R2D2'
query = an(entity(p).where(r.name == "R2D2"))

for part in query.evaluate():
    print(f"Found Part: {part.name} belonging to R2D2")
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.flat_variable`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Attribute`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Index`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Call`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.FlatVariable`
