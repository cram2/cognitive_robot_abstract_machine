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

# EQL for SQL Experts

If you are fluent in SQL, this guide will help you become productive with the Entity Query Language (EQL) quickly.
EQL provides a relational, pythonic way to express queries and rule-based reasoning directly in Python, while
minimizing boilerplate. A key difference from SQL is that joins are implicit: relationships are expressed as
conditions over Python objects, and EQL finds matching assignments for you.

This guide maps familiar SQL ideas (SELECT, WHERE, JOIN, EXISTS/IN, boolean logic, uniqueness) to EQL constructs.

<!-- #region -->

## TL;DR: SQL → EQL Mental Model

- **FROM aliasing**
  `variable(type, domain)` — Define placeholders (similar to table aliases) and their data source.

- **SELECT one column/entity**
  `entity(selected_variable).where(...conditions)` — Choose which variable to return.

- **SELECT multiple columns**
  `set_of(var1, var2, ...).where(...conditions)` — Return a tuple of variables.

- **WHERE predicates**
  Logical operators like `and_(...)`, `or_(...)`, `not_(...)`, comparators  `contains`/`in_` like and Python methods (
  e.g., `.startswith`).

- **JOIN (INNER)**
  Express relationships between placeholders/attributes (equality). EQL implies joins automatically.


- **Exact row enforcement**
  `the(...).evaluate()`—Raises an error for multiple matches. Use `an(...)` for many rows.

<!-- #endregion -->

## Minimal examples

Let's define a data model for some minimal examples.

```{code-cell} ipython3
from dataclasses import dataclass, field

from typing_extensions import List

from krrood.entity_query_language.entity import variable, Symbol, entity, and_, in_, contains, set_of
from krrood.entity_query_language.entity_result_processors import an


@dataclass
class Body(Symbol):
    name: str


@dataclass
class Connection(Symbol):
    parent: Body
    child: Body


@dataclass
class Prismatic(Connection):
    ...


@dataclass
class Fixed(Connection):
    ...


@dataclass
class World(Symbol):
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


# Construct a small world
world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
c1_c2 = Prismatic(world.bodies[0], world.bodies[1])
c2_h2 = Fixed(world.bodies[1], world.bodies[3])
world.connections = [c1_c2, c2_h2]

world = World(1, [Body("Body1"), Body("Body2")])
```

The SQL query

```sql
SELECT body.*
FROM bodies AS body
WHERE body.name = 'Body2';
```

can be translated to EQL as

```{code-cell} ipython3
b = variable(Body, domain=world.bodies)
query = an(entity(b).where(b.name == "Body2"))
print(*query.evaluate(), sep="\n")
```

## LIKE and string predicates

SQL

```sql
SELECT body.*
FROM bodies AS body
WHERE body.name LIKE 'Body%'
  AND body.name LIKE '%2';
```

EQL

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)
query = an(entity(body).where(and_(body.name.startswith("Body"), body.name.endswith("2"))
                                  ))
print(*query.evaluate(), sep="\n")
```

## IN and EXISTS

SQL

```sql
SELECT body.*
FROM bodies AS body
WHERE body.name IN ('Container1', 'Handle1');
```

EQL

```{code-cell} ipython3
names = ["Container1", "Handle1"]

body = variable(Body, domain=world.bodies)
in_results_generator = an(entity(body).where(in_(body.name, names)))
contains_results_generator = an(entity(body).where(contains(names, body.name)))
print(*in_results_generator.evaluate(), sep="\n")
print(*contains_results_generator.evaluate(), sep="\n")
```

EXISTS in SQL is naturally expressed by just introducing a placeholder and relating it in conditions. If the
relationships can be satisfied, it “exists.”

## Implicit JOINs via relationships

In SQL you join tables explicitly. In EQL you state the relationships between placeholders and attributes; the join is
inferred.

SQL (conceptually)

```sql
SELECT parent_container.*, prismatic_connection.*, drawer_body.*, fixed_connection.*, handle.*
FROM bodies AS parent_container
         JOIN prismatic AS prismatic_connection ON parent_container.id = prismatic_connection.parent_id
         JOIN bodies AS drawer_body ON drawer_body.id = prismatic_connection.child_id
         JOIN fixed AS fixed_connection ON drawer_body.id = fixed_connection.parent_id
         JOIN bodies AS handle ON handle.id = fixed_connection.child_id;
```

EQL

```{code-cell} ipython3
parent_container = variable(Body, domain=world.bodies)
prismatic_connection = variable(Prismatic, domain=world.connections)
drawer_body = variable(Body, domain=world.bodies)
fixed_connection = variable(Fixed, domain=world.connections)
handle = variable(Body, domain=world.bodies)

# SELECT (parent_container, prismatic_connection, drawer_body, fixed_connection, handle) WHERE relationships hold
query = an(set_of(parent_container, prismatic_connection, drawer_body, fixed_connection, handle).where(
                              and_(parent_container == prismatic_connection.parent,
                                   drawer_body == prismatic_connection.child,
                                   drawer_body == fixed_connection.parent,
                                   handle == fixed_connection.child)
                              ))
print(*query.evaluate(), sep="\n")                          
```

Notice how equality constraints play the role of JOIN conditions. The set_of(...) returns a tuple-like result where you
can access each variable by its identity if needed.
