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

# Writing Basic Queries

This guide covers the fundamental building blocks of EQL: defining variables with `variable()`, selecting them with
`entity()`, and filtering them with `.where()`.

## Defining Variables

All EQL queries start with symbolic variables. A variable represents a set of possible values (the domain) of a certain
type.

### 1. Variables with Explicit Domains
The most common way to define a variable is to provide a type and a collection of objects (the domain).

```python
from krrood.entity_query_language.factories import variable

# Define a variable 'r' of type 'Robot' from a list
robots_list = [ExampleRobot("R1"), ExampleRobot("R2")]
r = variable(ExampleRobot, domain=robots_list)
```

### 2. Variables with Implicit Domains
If no domain is provided, EQL will attempt to use a default domain. For types inheriting from `Symbol`,
it uses the global `SymbolGraph`.

```python
# 'r' will use all Robots in the SymbolGraph
r = variable(ExampleRobot, domain=None)
```

```{hint}
Always provide a domain if you know it. It significantly improves query performance by narrowing the search
space early.
```

## Selecting Entities

The `entity()` function specifies what you want to retrieve from the query. It returns a `Query` object that you can
further refine.

```python
from krrood.entity_query_language.factories import entity

# We want to select the objects represented by 'r'
query = entity(r)
```

```{note}
If you need to select multiple variables at once, use {py:func}`~krrood.entity_query_language.factories.set_of` instead.
```

## Aliasing and Named Results

When selecting multiple variables or complex expressions using `set_of()`, the query returns a result set where each item is a dictionary-like object (`UnificationDict`). To make it easier to refer to these results—both within the query and when iterating over results—you can use **aliasing**.

EQL supports both the walrus operator (`:=`) and standard Python assignment (`=`) for this purpose.

### 1. Inline Aliasing with the Walrus Operator (`:=`)
The walrus operator (available in Python 3.8+) is the most concise way to alias an expression directly within the selection function.

```python
from krrood.entity_query_language.factories import set_of, count

# Alias the count(r) expression to 'c' inline
query = set_of(r.type, c := count(r)).grouped_by(r.type)

for result in query.evaluate():
    # Use the alias 'c' to index the result dictionary
    print(f"Type: {result[r.type]} | Count: {result[c]}")
```

### 2. Aliasing with Normal Assignment (`=`)
Alternatively, you can assign a symbolic expression to a Python variable before passing it to the query:

```python
# Define the alias before the query
avg_batt = average(r.battery)
query = set_of(r.type, avg_batt).grouped_by(r.type)

for result in query.evaluate():
    # Use the 'avg_batt' variable to access the result
    print(f"Type: {result[r.type]} | Average Battery: {result[avg_batt]}%")
```

### Why use Aliasing?
1.  **Readability**: Shorter names make complex queries easier to follow.
2.  **Result Access**: It provides a clear handle for retrieving values from the result dictionary.
3.  **Code Reuse**: You can define an expression once and use it in multiple parts of the query, such as in `.where()`, `.having()`, or `.ordered_by()`.

```python
# Using an alias in multiple places
query = set_of(r.type, c := count(r)) \
    .grouped_by(r.type) \
    .having(c > 5) \
    .ordered_by(c, descending=True)
```

```{note}
Aliasing is especially powerful when combined with {doc}`aggregations <aggregators>`, as it allows you to name the summary statistics you calculate.
```

## Adding Filters with `.where()`

The `.where()` method is used to add constraints to your query. You can pass multiple conditions, which are treated as
a logical **AND**. Constraints can be Comparison Operators (`==`, `!=`, `<`, `<=`, `>`, `>=`), custom functions, 
predicates, or any boolean function/attribute that is accessed from a variable (e.g. `'abc'.startswith('a')`).

```python
# Select robots named 'R1'
query = entity(r).where(r.name == "R1")
```

## Full Example: Finding High-Battery Robots

Let's combine these concepts into a complete, runnable example.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import variable, entity, an, Symbol

# Define our model
@dataclass
class ExampleRobot:
    name: str
    battery: int

# Prepare data
robots = [ExampleRobot("R2D2", 100), ExampleRobot("C3PO", 20), ExampleRobot("BB8", 80)]

# 1. Define the variable
r = variable(ExampleRobot, domain=robots)

# 2. Build the query with filters
# We want robots with battery levels greater than 50
query = an(entity(r).where(r.battery > 50))

# 3. Execute and see the results
for robot in query.evaluate():
    print(f"Found {robot.name} with {robot.battery}% battery")
```

## API Reference
- {py:func}`~krrood.entity_query_language.factories.variable`
- {py:func}`~krrood.entity_query_language.factories.entity`
- {py:meth}`~krrood.entity_query_language.query.query.Query.where`
