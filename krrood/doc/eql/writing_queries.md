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

# Writing Queries

EQL can be well-used to answer symbolic questions through querying.

Whenever you write a query you have to wrap free variables in `variable` statements.
`variable` wraps your classes such that attribute access is intercepted and replaced by a symbolic expression.

This is different from plain python in the sense that it doesn't evaluate what you write directly but
treats your statements as something that will be evaluated later (lazily).
Queries typically compare attributes of variables where the assignments of the 
variables don't exist yet, hence an immediate evaluation would cause failures.

Frameworks like SQLAlchemy, as an Object-Relational Mapper (ORM), use metaprogramming techniques 
(specifically, class and attribute interception/rewriting) to manage database interactions and object state, 
which introduces performance overhead and requires developer awareness of the framework's internal mechanisms when 
designing application classes.

KRROOD explicitly avoids this overhead by using wrappings and hence also is less invasive.

This approach ensures that your class definitions remain pure and decoupled from the query mechanism 
outside the explicit symbolic context. Consequently, your classes can focus exclusively on their domain logic, 
leading to better adherence to the [Single Responsibility Principle](https://realpython.com/solid-principles-python/#single-responsibility-principle-srp).

Here is a query example that finds all bodies in a world whose name starts with "B":

```{code-cell} ipython3
from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.entity import entity, variable, Symbol
from krrood.entity_query_language.entity_result_processors import an


@dataclass
class Body(Symbol):
    name: str
    
    
@dataclass
class Handle(Body):
    pass


@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2"), Handle("Handle1"), Handle("Handle2"), Handle("Handle2")])

body = variable(Body, domain=world.bodies)
query = an(
    entity(
        body).where(body.name.startswith("B"),
    )
)
print(*query.evaluate(), sep="\n")
```

```{warning}
Conditions must be symbolic expressions not constants/literals because they are then always True or False. This
is not affected by any of the query variables, thus doesn't make sense to put as a condition. An error will be raised
if you try to do so.
```

## Retrieving Results

EQL provides convenient ways to retrieve query results.

### Using `.evaluate()`

The most general way to retrieve results is using `.evaluate()` (must use a result processor like an/the) to get an
iterator over the results. This is recommended when processing large result sets lazily.

```{code-cell} ipython3
query = an(entity(body).where(body.name.startswith("B")))
for res in query.evaluate():
    print(res)
```

### Using `.tolist()`

A common way to retrieve all results is using the `.tolist()` method, which is available on both `QueryObjectDescriptor`
(e.g., `set_of`, `entity`) and `ResultProcessors` (e.g., `count`, `sum`). It evaluates the query and returns the results
as a list.

```{code-cell} ipython3
import krrood.entity_query_language.entity_result_processors as eql
from krrood.entity_query_language.entity import set_of

# On a ResultProcessor
first_body = an(entity(body)).tolist()[0]
print(first_body)

# On a QueryObjectDescriptor
all_names = entity(body.name).tolist()
print(all_names)
```

### Ordering with `.order_by()`

Query objects now support ordering of results.

```{code-cell} ipython3
# Order bodies by height in descending order
query = entity(body).order_by(body.name[-1], descending=True)
sorted_bodies = query.tolist()
for b in sorted_bodies:
    print(b)
```

### Distinct with `distinct()`

The `distinct()` function (or the `.distinct()` method) is used to remove duplicate results from a query. 

```{code-cell} ipython3
from krrood.entity_query_language.entity import distinct

# Using the functional form
distinct_bodies = distinct(entity(body)).tolist()

# Using the method form
distinct_bodies = entity(body).distinct().tolist()
```

#### Distinct on specific attributes

You can specify specific attributes to determine distinctness by passing them as additional arguments to the `distinct()` function or method. This is useful when you want to retrieve unique entities based on a subset of their properties.

```{code-cell} ipython3
# Get unique bodies based on their name attribute
# If multiple bodies have the same name, only the first one encountered is returned
distinct_bodies_by_name = distinct(entity(body), body.name).tolist()
```




