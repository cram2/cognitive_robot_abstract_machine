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

# Introduction

The Entity Query Language (EQL) is a relational query language that is pythonic, and intuitive.

The interface side of EQL is inspired by [euROBIN](https://www.eurobin-project.eu/) entity query language white paper.

## Basic Example

An important feature of EQL is that you do not need to do operations like JOIN in SQL, this is performed implicitly.
EQL tries to mirror your intent in a query statement with as little boilerplate code as possible.
For example, attribute access with and equal check to another value is just as you expect:

```{code-cell} ipython3
from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.entity import entity, variable, contains, Symbol
from krrood.entity_query_language.entity_result_processors import an


@dataclass
class Body(Symbol):
    name: str


@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

body = variable(Body, domain=world.bodies)
query = an(entity(body).where(contains(body.name, "2"),
                  body.name.startswith("Body"))
               )
print(*query.evaluate(), sep="\n")
```

where this creates a body variable that gets its values from world.bodies, and filters them to have their att "name"
equal to "Body1".

Note that it is possible to use both provided helper methods by EQL and other methods that are accessible through your
object instance and use them as part of the query conditions.

It's all your python definitions, and that is the beauty of it.
