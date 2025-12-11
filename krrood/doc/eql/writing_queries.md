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

Whenever you write a query you have to wrap free variables in `let` statements.
`let` wraps your classes such that attribute access is intercepted and replaced by a symbolic expression.

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

from krrood.entity_query_language.entity import entity, let, Symbol
from krrood.entity_query_language.quantify_entity import an


@dataclass
class Body(Symbol):
    name: str


@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

query = an(
    entity(
        body := let(Body, domain=world.bodies), body.name.startswith("B"),
    )
)
print(*query.evaluate(), sep="\n")
```

```{warning}
Conditions must be symbolic expressions not constants/literals because they are then always True or False. This
is not affected by any of the query variables, thus doesn't make sense to put as a condition. An error will be raised
if you try to do so.
```



