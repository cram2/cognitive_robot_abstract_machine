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
config_val = robot.qp_controller_config["battery_limit"] # -> returns IndexedVariable(robot.config, "battery_limit")
```

## A Variable Over a Collection

`robot.parts` is the list of parts itself. To range over the parts one at a time, create a variable whose domain is
that expression, with `variable_from(...)`, `variable(Type, ...)` or `a(Type).from_(...)`:

```python
from krrood.entity_query_language.factories import a, variable_from

part = variable_from(robot.parts)          # each part of each robot
arm = a(ExamplePart).from_(robot.parts)    # each part of each robot that is an ExamplePart
```

The variable takes, for each value of `robot`, each element of that robot's parts, and keeps the binding of `robot`,
like a `JOIN` or `UNNEST` in SQL. Each variable created this way is a new one, so two of them range over the parts
independently. Whether a value is read as a collection follows its declared type: a field declared as a collection
gives its elements, `None` gives none, and a field declared as one object gives that object even when it can be
iterated. A value of undeclared type, such as a method's result, gives its elements when it is a collection.

```{note}
`flat_variable(...)` is deprecated; it is the same as `variable_from(...)`.
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
from krrood.entity_query_language.factories import variable, entity, an, variable_from, Symbol

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

# 2. Range a variable over the parts of each robot
p = variable_from(r.parts)

# 3. Build a query to find parts belonging to 'R2D2'
query = an(entity(p).where(r.name == "R2D2"))

for part in query.evaluate():
    print(f"Found Part: {part.name} belonging to R2D2")
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.variable_from`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Attribute`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Index`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Call`
- {py:class}`~krrood.entity_query_language.core.variable.ExpressionDomain`
