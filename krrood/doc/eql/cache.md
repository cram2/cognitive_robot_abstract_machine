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


# Using Cached Symbols

EQL provides a cache for all symbol classes.
Whenever python creates an object of a class that inherits from `Symbol`, EQL will cache the instance in the symbol graph.
When you create a free variable via `variable` without a domain, EQL will use the `Symbol Graph` to extract the domain.

```{warning}
When you have a lot of instances of a type but your query requires only a few of them, EQL will check a lot of unnecessary instances.
Provide a domain in such cases.
```

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

body = variable(Body, domain=None)  # no domain here
query = an(
    entity(
        body).where(
        contains(body.name, "2"),
        body.name.startswith("Body"),
    )
)

print(*query.evaluate(), sep="\n")
```
