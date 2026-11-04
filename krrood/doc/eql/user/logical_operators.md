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

## Quantifiers (FOR ALL and EXISTS)

`for_all(x, condition)` holds when the condition holds for **every** value of `x`, and `exists(x, condition)` holds when
it holds for **at least one** value of `x`. The values of `x` may depend on another variable, as when `x` ranges over a
collection attribute with `flat_variable`:

```python
student = variable(Student, domain=students)
course = flat_variable(student.takes_course)

# Students none of whose courses is "Databases"
query = an(entity(student).where(for_all(course, course.name != "Databases")))

# Students taking at least one "Logic" course
query = an(entity(student).where(exists(course, course.name == "Logic")))

# Students taking no "Logic" course
query = an(entity(student).where(not_(exists(course, course.name == "Logic"))))
```

A quantifier is evaluated once for each binding of its **outer-visible** variables: those that also occur somewhere
else in the same query, whether selected (like `student` above) or used by another condition. So the three queries above
range over each student's own courses. A variable that occurs only inside the quantifier, including inside the
quantified variable's own domain expression, is local to the quantifier and is quantified together with it: in
`for_all(cabinets.container, ...)`, where `cabinets` appears nowhere else, the condition must hold for the containers of
all cabinets. Over an empty collection, `for_all` is true and `exists` is false.

`exists` only filters the outer bindings. Each one appears in the results at most once, however many values of the
quantified variable satisfy the condition, and the local variables never appear in the results. If you need those
values, write the condition without `exists` (a join) instead. `not_(exists(...))` keeps the outer bindings for which no
value satisfies the condition, including those whose collection is empty.

```python
# One row per student taking a "Logic" course
query = an(entity(student).where(exists(course, course.name == "Logic")))

# One row per (student, course) pair whose course is "Logic": a join, so the course is in the results
query = set_of(student, course).where(course.name == "Logic")
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
- {py:class}`~krrood.entity_query_language.operators.logical_quantifiers.ForAll`
- {py:class}`~krrood.entity_query_language.operators.logical_quantifiers.Exists`
