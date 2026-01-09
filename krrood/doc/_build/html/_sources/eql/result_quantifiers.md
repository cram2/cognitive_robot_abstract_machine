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

# Result Quantifiers

In EQL, there are two result quantifiers: `the` and `an`.

`the` is used to fetch a single solution and assert that there is exactly one solution. This behaves like [one](https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Result.one) in SQLAlchemy.

`an` is used to fetch all solutions (or any specified number of solutions). This creates an iterator which lazily
evaluates the query. This behaves like [all](https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Result.all) in SQLAlchemy if not constraining result count using `at_least`, `at_most`, or `exactly`.

Let's start with an example of a working query that requires exactly one result.

```{code-cell} ipython3
from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.entity import entity, variable, Symbol
from krrood.entity_query_language.entity_result_processors import an, the
from krrood.entity_query_language.result_quantification_constraint import AtLeast, AtMost, Exactly, Range
from krrood.entity_query_language.failures import MultipleSolutionFound, LessThanExpectedNumberOfSolutions, GreaterThanExpectedNumberOfSolutions


@dataclass
class Body(Symbol):
    name: str


@dataclass
class World(Symbol):
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

body = variable(Body, domain=world.bodies)
query = the(entity(body).where(
                   body.name.endswith("1")))
print(query.evaluate())
```

If there are multiple results, we get an informative exception.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)
query = the(entity(body))

try:
    query.evaluate()
except MultipleSolutionFound as e:
    print(e)
```

We can also get all results using `an`.

```{code-cell} ipython3
body = variable(Body, domain=None)
query = an(entity(body))

print(*query.evaluate(), sep="\n")
```


## Result Count Constraints

EQL allows constraining the number of results produced by `an(...)` using the `quantification` keyword argument.

Below we reuse the same `World` and `Body` setup from above. 
The world contains exactly two bodies, so all the following examples will evaluate successfully.

```{code-cell} ipython3
# Require at least one result
body = variable(Body, domain=world.bodies)
query = an(
    entity(body),
    quantification=AtLeast(1),
)

print(len(list(query.evaluate())))  # -> 2
```

You can also bound the number of results within a range using both `at_least` and `at_most`:

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)
query = an(
    entity(body),
    quantification=Range(AtLeast(1), AtMost(3))
)

print(len(list(query.evaluate())))  # -> 2
```

If you want an exact number of results, use `exactly`:

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)
query = an(
    entity(body),
    quantification=Exactly(2),
)

print(len(list(query.evaluate())))  # -> 2
```



## Handling Unmatched Result Counts

The result count constraints will raise informative exceptions when the number of results does not match the expectation. You can handle these with try/except and print the error message, for example:

```{code-cell} ipython3
# The world from above has exactly two bodies: Body1 and Body2

# at_least too high -> LessThanExpectedNumberOfSolutions
body = variable(Body, domain=world.bodies)
query = an(
    entity(body),
    quantification=AtLeast(3),
)
try:
    list(query.evaluate())
except LessThanExpectedNumberOfSolutions as e:
    print(e)

# at_most too low -> GreaterThanExpectedNumberOfSolutions
body = variable(Body, domain=world.bodies)
query = an(
    entity(body),
    quantification=AtMost(1),
)
try:
    list(query.evaluate())
except GreaterThanExpectedNumberOfSolutions as e:
    print(e)

# exactly mismatch -> can raise either LessThan... or GreaterThan...
body = variable(Body, domain=world.bodies)
query = an(
    entity(body),
    quantification=Exactly(1),
)
try:
    list(query.evaluate())
except (LessThanExpectedNumberOfSolutions, GreaterThanExpectedNumberOfSolutions) as e:
    print(e)
```