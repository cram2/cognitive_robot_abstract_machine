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

# Predicate and Symbolic Function

Custom predicates allow you to encapsulate reusable boolean logic that can be used inside queries. 
In addition, any Python function decorated with `@symbolic_function` can be used.
A Predicate class is a dataclass that inherits from EQL `Predicate` class.
These two become symbolic variables when any of the arguments are variables of a query.
If all the arguments are ordinary python objects, you can use them normally.

```{warning}
When using predicates and symbolic functions, at least one of the arguments must be a symbolic expression 
(like a variable constructed using `let` or another query). Otherwise, the predicate will be considered a 
constant/literal and cannot be used as a condition as it is either always True or always False.
```

Lets first define our model and some sample data.

```{code-cell} ipython3
from dataclasses import dataclass
from typing_extensions import List

from krrood.entity_query_language.entity import entity, let, Symbol
from krrood.entity_query_language.predicate import Predicate, symbolic_function
from krrood.entity_query_language.quantify_entity import an


@dataclass
class Body(Symbol):
    name: str


@dataclass
class Handle(Body):
    pass


@dataclass
class Container(Body):
    pass


@dataclass
class World(Symbol):
    id_: int
    bodies: List[Body]
    
# Sample world containing containers and handles
world = World(
    1,
    [
        Container("Container1"),
        Container("Container2"),
        Handle("Handle1"),
        Handle("Handle2"),
        Handle("Handle3"),
    ],
)
```

Now lets define a custom predicate and symbolic function and use them in a query.

```{code-cell} ipython3
# Define a reusable symbolic function: returns True if a body is a handle by name convention
@symbolic_function
def is_handle(body_: Body) -> bool:
    return body_.name.startswith("Handle")


@dataclass
class HasThreeInItsName(Predicate):
    body: Body

    def __call__(self):
        return '3' in self.body.name

# Build the query using the predicate inside symbolic mode
query = an(
    entity(
        body := let(type_=Body, domain=world.bodies),
        is_handle(body_=body),  # use the predicate just like any other condition
        HasThreeInItsName(body)
    )
)

# Evaluate and inspect the results
results = list(query.evaluate())
assert len(results) == 1
assert isinstance(results[0], Handle)
assert results[0].name == "Handle3"
print(*results, sep="\n")
```
