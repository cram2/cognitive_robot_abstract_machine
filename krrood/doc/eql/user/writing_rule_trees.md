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

# Advanced Inference: Rule Trees

Beyond simple queries, EQL supports an inference engine for building **Rule Trees**. This allows you to symbolically construct new objects or add information to existing variables based on complex, conditional logic.

## Core Concepts

A Rule Tree is built using three main components:
1.  **`deduced_variable(Type)`**: A special variable for objects that will be deduced by the rule.
2.  **`inference(Type)(**kwargs)`**: A special variable constructor for objects that will be "materialized" by the rule.
2.  **`add(target, value)`**: A conclusion clause that assigns a value to a symbolic variable.
3.  **ConclusionSelectors**: Logical branches that control rule evaluation flow and choose which conclusions are applied.
Examples: `refinement()`,`alternative()`, and `next_rule()`.


## The `with query:` Context

To build a rule tree, you use the query object as a context manager. Any `add`, `refinement`, or `alternative` inside
this block becomes part of that query's rule structure. We further discuss what each of these components do below.

```python
query = an(entity(views).where(...))

with query:
    add(views, default_conclusion)
    with refinement(extra_condition):
        add(views, specialized_conclusion)
```

```{warning}
Rule trees are for **inference** (adding data). For simple filtering, stick to `.where()` and standard queries.
```

## Conclusion Selectors

### 0. Query
One usually starts with a query that selects a `deduced_variable()`, and optionally add some conditions in the `where()`
clause. So in the example bellow we want to deduce views of type `View`, and we have a `FixedConnection` that connects
a `Body` to a `Handle` as the initial condition.
```python
views = deduced_variable(View)
handle = variable(ExampleHandle, domain=None)
body = variable(ExampleBody, domain=None)
fixed = variable(ExampleFixedConnection, domain=None)
query = an(entity(views).where(fixed.parent == body, fixed.child == handle))
````

### 1. Default Conclusion
The default conclusion is the one that is applied if none of the other branches match. So in the example bellow, we
 want to deduce a `Drawer` by default for every solution of the query.
```python
with query:
    add(views, inference(ExampleDrawer)(handle=handle, body=body))
```

### 2. Refining the condition using `refinement()`
Narrows the context with an additional condition. It behaves like a logical **AND** but is used specifically to 
specialize a rule. So in the example bellow, we want to deduce a `Door` instead of a `Drawer` if in addition to the
default condition the body size is greater than 1 meter.

```python
with query:
    add(views, inference(ExampleDrawer)(handle=handle, body=body))
    with refinement(body.size > 1):
        # This only happens if the body is big
        add(views, inference(ExampleDoor)(handle=handle, body=body))
```

### 3. Defining alternative conclusions using `alternative()`
Provides a sibling branch that concludes only where the previous branches didn't match. It behaves like an **Else-If**.
So in the example below, if the body and the handle are not connected through a fixed connection, but
through a revolute connection, we want to deduce a `DoorWithRevoluteHandle`.

The branches are alternatives *for the variables every one of their conclusions is about*, here the body and the
handle. The `alternative()` concludes for a body and a handle only if no earlier branch holds for them, however many
fixed connections the `where` tried and failed on: a variable that only the conditions use, like the fixed connection,
is tried the way an `exists` would be. A variable that the conclusions use but no condition binds is enumerated, so
every one of its values reaches a conclusion. When the conclusions of the branches share no variable at all, the
alternative is tried for every combination of values the previous branch failed for.

```python
with query:
    add(views, inference(ExampleDrawer)(handle=handle, body=body))
    revolute = variable(ExampleRevoluteConnection, domain=None)
    with alternative(revolute.parent == body, revolute.child == handle):
        add(views, inference(ExampleDoorWithRevoluteHandle)(handle=handle, body=body))
```

```{hint}
Use `refinement` for specialization (exceptions) and `alternative` for mutually exclusive cases.
```

## Full Example: Categorizing Connections

This example demonstrates how to build a rule tree that categorizes connections into either `Fixed` or `Revolute` views.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import (
    variable, entity, an, Symbol, deduced_variable, refinement, alternative, add, inference
)

@dataclass
class ExampleConnection:
    type_code: int
    name: str

@dataclass
class ExampleView:
    connection: ExampleConnection

@dataclass
class ExampleFixedView(ExampleView): pass

@dataclass
class ExampleRevoluteView(ExampleView): pass

# Data
connections = [ExampleConnection(1, 'c1'), ExampleConnection(2, 'c2'), ExampleConnection(3, 'c3'), ExampleConnection(4, 'm4')]
connection = variable(ExampleConnection, domain=connections)
view = deduced_variable(ExampleView)

# 1. Base query
query = entity(view).where(connection.name.startswith('c'))

# 2. Rule Tree definition
with query:
    # Default case (when connection name starts with 'c'):
    add(view, inference(ExampleView)(connection=connection))
    
    # If connection name starts with 'c' and type_code is 1, it's a ExampleFixedView
    with refinement(connection.type_code == 1):
        add(view, inference(ExampleFixedView)(connection=connection))
    
    # Otherwise, if connection name does not start with 'c' and the type_code is 4, it's a ExampleRevoluteView
    with alternative(connection.type_code == 4):
        add(view, inference(ExampleRevoluteView)(connection=connection))

# 3. Execution
results = query.tolist()
print(f"Inferred {len(results)} views from {len(connections)} connections.")
print("\n".join([str(v) for v in results]))
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.deduced_variable`
- {py:func}`~krrood.entity_query_language.factories.inference`
- {py:func}`~krrood.entity_query_language.factories.add`
- {py:func}`~krrood.entity_query_language.factories.refinement`
- {py:func}`~krrood.entity_query_language.factories.alternative`
