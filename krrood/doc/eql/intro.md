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

# The Philosophy of EQL
## EQL is Relational

Entity Query Language (EQL) embraces the relational model. It rethinks relational querying in native Python terms.

EQL does not attempt to replace Python with a domain-specific language. It makes Python itself the query language.

Where SQL requires explicit joins and table navigation, EQL treats relationships as natural object references.
Traversal replaces joins. Structure replaces boilerplate.

## EQL is Pythonic First

EQL is not a string-based query system. Queries are written directly in Python syntax, using real objects, real functions, and real control flow.

There is no artificial boundary between “query language” and “application language.”
If you can write a Python function, you can use it inside a query.

EQL takes the expressivity of Python and builds on that rather than hiding it.

## EQL Eliminates the Representation Layer

Traditional systems often introduce an additional abstraction layer (a DDL) and another intermediary that links between
these two representations (e.g., ORM). EQL does not.

EQL operates directly on user-defined objects.
There is no need to map entities into tables, nor to maintain a separate relational representation.

Objects remain objects. The data model you design is the data model you query.

## EQL is Declarative (What not How you want it)

The core principle of EQL is minimal extra detail.

If you want an object with certain properties, you describe those properties directly. You do not describe how to retrieve it, how to join it, or how to reconstruct it.

EQL focuses on **what you want** not **how to assemble it**, in other words, the query **expresses intent**,
not **execution strategy**.

## EQL Supports Structural Depth

Python’s built-in data structures — lists, dictionaries, nested objects — are first-class citizens in EQL.

There is no restriction to flat relational tables. Structures may be nested arbitrarily.

EQL treats relational querying as a structural operation over objects.

## EQL as a Description Language

EQL is not only a query language.
It is also a description language.

You can write first-order-logic statements directly in Python, allowing declarative constraints, structural matching, and semantic expression within the same framework.

This unifies:

- Querying

- Structural description

- Logical constraint

into a single, consistent language model.

## In Summary

EQL stands for:

- Direct object querying

- Native Python expressiveness

- No artificial ORM layer

- Structural traversal instead of joins

- Intent-driven declarative style

It does not attempt to make Python behave like SQL.

It assumes Python is already powerful enough — and builds a relational model that fits naturally inside it.

```{note}
Some parts of EQL interface were inspirations from [euROBIN](https://www.eurobin-project.eu/).
```

## The "Hello World" of EQL

Let's start with a simple example: finding a specific body in a "world".

### 1. Define your domain model
EQL works seamlessly with standard Python dataclasses.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ExamplePhysicalBody:
    name: str

@dataclass
class ExamplePhysicalWorld:
    bodies: List[ExamplePhysicalBody]
```

### 2. Prepare some data
```python
world = ExamplePhysicalWorld(bodies=[ExamplePhysicalBody("Robot"), ExamplePhysicalBody("Human")])
```

### 3. Build and run the query
We want to find a body named "Robot".

```python
from krrood.entity_query_language.factories import entity, variable, an

# 1. Define a symbolic variable representing any ExampleBody from world.bodies
body = variable(ExamplePhysicalBody, domain=world.bodies)

# 2. Create a query: we want "an" entity "body" WHERE "body.name" is "Robot"
query = an(entity(body).where(body.name == "Robot"))

# 3. Evaluate the query to get results
results = list(query.evaluate())
print(results)
```

## Bit-by-Bit Explanation

*   **`variable(ExampleBody, domain=world.bodies)`**: This creates a symbolic placeholder. It tells EQL that we are interested
in objects of type `ExampleBody` that are found in the `world.bodies` collection. See {py:func}`~krrood.entity_query_language.factories.variable`.
*   **`entity(body)`**: This starts the selection. We are saying "I want to select the objects represented by the `body`
variable". See {py:func}`~krrood.entity_query_language.factories.entity`.
*   **`.where(body.name == "Robot")`**: This adds a filter. Notice how we use standard Python comparison operators. EQL
captures these and translates them into symbolic constraints.
*   **`an(...)`**: This is an optional result quantifier. It tells EQL that we expect zero or more results. If no
quantifier is provided, `an()` is assumed. See {py:func}`~krrood.entity_query_language.factories.an`.
*   **`.evaluate()`**: This triggers the execution engine. It returns a generator of results that satisfy all conditions.

```{note}
All logic in EQL is deferred. The query is only executed when you call `.evaluate()` and iterate over it,
or when you call `.tolist()`, or `.first()`.
```

## Full Example

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List
from krrood.entity_query_language.factories import entity, variable, an

@dataclass
class ExampleBody:
    name: str

@dataclass
class ExampleWorld:
    bodies: List[ExampleBody]

world = ExampleWorld(bodies=[ExampleBody("Robot"), ExampleBody("Human")])

# Define the variable and build the query
body = variable(ExampleBody, domain=world.bodies)
query = an(entity(body).where(body.name == "Robot"))

# Execute and print results
for result in query.evaluate():
    print(f"Found: {result.name}")
```

## Automatic Domain Discovery with Symbol

EQL provides a mechanism to automatically discover the domain of a variable. This is especially useful when 
your objects are part of a global state.

### Caching with `Symbol`
By inheriting from `Symbol`, instances of your classes are automatically cached in a graph called the `SymbolGraph`. 
This allows EQL to automatically create the domain if no explicit domain is provided for your variables.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import entity, variable, Symbol, an

@dataclass
class ExampleBody(Symbol):
    name: str

# Instances are automatically cached in SymbolGraph upon creation
robot = ExampleBody("Robot")
human = ExampleBody("Human")

# No explicit domain provided to variable(); it's inferred from SymbolGraph
body = variable(ExampleBody, domain=None)
query = an(entity(body).where(body.name == "Robot"))

for result in query.evaluate():
    print(f"Found: {result.name}")
```
