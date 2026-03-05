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

# Result Processors

Once you have built a query, you need to execute it to retrieve the results.
EQL provides three main methods for this: `.evaluate()`, `.tolist()`, and `.first()`.

## 1. The `.evaluate()` Method

The `.evaluate()` method is the standard way to execute a query. It returns a **generator** that yields results one by one.

```python
# Execute the query lazily
results_generator = query.evaluate()

for result in results_generator:
    print(result)
```

```{hint}
Use `.evaluate()` when you expect a large number of results and want to process them efficiently without
loading everything into memory.
```

## 2. The `.tolist()` Method

The `.tolist()` method consumes the entire generator and returns the results as a standard Python **list**.

```python
# Get all results at once
results_list = query.tolist()
print(f"Total results: {len(results_list)}")
```

```{note}
`.tolist()` is a convenience method equivalent to `list(query.evaluate())`.
```

## 3. The `.first()` Method

The `.first()` method is a convenience for retrieving only the **first** result from the query.

```python
# Get just one result
top_result = query.first()
```

```{warning}
If the query returns no results, `.first()` will raise a `StopIteration` exception.
```

## Full Example: Different Ways to Consume Results

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import variable, entity, an, Symbol

@dataclass
class ExampleRobot(Symbol):
    name: str

robots = [ExampleRobot("R1"), ExampleRobot("R2"), ExampleRobot("R3")]
r = variable(ExampleRobot, domain=robots)
query = an(entity(r))

# 1. Using evaluate()
print("Using evaluate():")
for res in query.evaluate():
    print(f" - {res.name}")

# 2. Using tolist()
all_robots = query.tolist()
print(f"Using tolist(): Found {len(all_robots)} robots")

# 3. Using first()
first_one = query.first()
print(f"Using first(): The first one is {first_one.name}")
```

## API Reference
- {py:meth}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression.evaluate`
- {py:meth}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression.tolist`
- {py:meth}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression.first`
