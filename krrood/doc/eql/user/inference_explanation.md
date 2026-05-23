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

# Inference Explanation

When EQL infers a new object via an `inference(...)` rule, it automatically attaches an
**`InferenceExplanation`** to that object. The explanation records:

- **Which query node** produced the instance.
- **Which conditions** in the query were satisfied (and their truth-value bindings).
- **The full call stack** at the point where the query was written.
- The **`OperationResult`** from the evaluation, carrying the complete variable bindings.

`explain_inference` returns `None` for instances that were not produced by an inference variable
(e.g. plain instances constructed directly).

---

## Data Model

All examples on this page share the same domain. We model a simple world of bodies, handles, and
the fixed connections between them. An `ExampleDrawer` is inferred whenever a fixed connection
links a body whose name starts with `"big"` to a handle.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import (
    variable, entity, Symbol, inference, or_,
)
from krrood.entity_query_language.explanation.explanation import explain_inference


@dataclass
class ExampleHandle(Symbol):
    name: str


@dataclass
class ExampleBody(Symbol):
    name: str


@dataclass
class ExampleFixedConnection(Symbol):
    parent: ExampleBody
    child: ExampleHandle


@dataclass
class ExampleDrawer(Symbol):
    body: ExampleBody
    handle: ExampleHandle


bodies = [ExampleBody("big_body"), ExampleBody("small_body")]
handles = [ExampleHandle("H1"), ExampleHandle("H2")]
connections = [
    ExampleFixedConnection(parent=bodies[0], child=handles[0]),
    ExampleFixedConnection(parent=bodies[1], child=handles[1]),
]
```

We use two variables — `fixed` ranges over fixed connections, `handle_var` ranges over handles —
and an `or_()` condition whose second arm (`startswith("huge")`) can never match our data.
Because EQL short-circuits `OR` as soon as the left operand succeeds, the right arm is never
entered. This makes the condition graph show two distinct transparency levels: the entered and
satisfied nodes at full opacity, and the unvisited `"huge"` branch faded out.

```{code-cell} ipython3
handle_var = variable(ExampleHandle, domain=handles)
fixed = variable(ExampleFixedConnection, domain=connections)

results = (
    entity(inference(ExampleDrawer)(body=fixed.parent, handle=fixed.child))
    .where(
        or_(
            fixed.parent.name.startswith("big"),   # True → OR short-circuits here
            fixed.parent.name.startswith("huge"),  # never entered
        ),
        fixed.child == handle_var,
    )
    .tolist()
)

print(f"Inferred {len(results)} drawer(s): {results}")
```

---

## Retrieving an Explanation

Pass any inferred instance to `explain_inference` to get its `InferenceExplanation`.

```{code-cell} ipython3
expl = explain_inference(results[0])
print(type(expl))
print(expl is None)  # False — the drawer was produced by inference
```

Calling it on a directly constructed object returns `None`:

```{code-cell} ipython3
direct = ExampleDrawer(body=bodies[0], handle=handles[0])
print(explain_inference(direct))  # None
```

---

## Simple Usage

### Human-readable summary

`as_string()` prints which inference variable created the instance, which query it belonged to,
and the call stack recorded at query-definition time.

```{code-cell} ipython3
print(expl.as_string())
```

Pass `focus_package` to restrict the stack to frames from a specific package:

```{code-cell} ipython3
import krrood
print(expl.as_string(focus_package=krrood))
```

### Satisfied conditions

`get_satisfied_conditions_as_string()` lists every condition that evaluated to `True` for this
inference, joined by `AND`.

```{code-cell} ipython3
print(expl.get_satisfied_conditions_as_string())
```

### Condition graph

`condition_graph()` returns a `QueryGraph` with `is_satisfied` flags on every node. Satisfied
condition nodes keep their type-based colour; unsatisfied ones are faded. Call `.visualize()`
to render it and `IPython.display.display` to show the figure inline.

```{code-cell} ipython3
import io
import matplotlib.pyplot as plt
from IPython.display import Image

graph = expl.condition_graph()
fig, ax = graph.visualize(filename="drawer_graph.pdf")

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
buf.seek(0)
plt.close(fig)
Image(buf.read())
```

---

## Meta-queries

`InferenceExplanation` inherits from `Symbol`, making it a first-class entity in the
`SymbolGraph`. Its methods return EQL **Entity** descriptors that can be chained, filtered, and
composed just like ordinary queries.

### Which conditions were satisfied?

Returns the raw condition expressions (comparators, predicates) that were `True` for this
instance.

```{code-cell} ipython3
conditions = expl.get_satisfied_condition_expressions_for_the_instance().tolist()
for c in conditions:
    print(c)
```

### Which variable nodes participated (by type)?

Returns the symbolic variable nodes whose type is a subclass of the given type.

```{code-cell} ipython3
fixed_nodes = expl.get_variable_nodes_of_given_type(ExampleFixedConnection).tolist()
print("FixedConnection variable nodes:", fixed_nodes)

handle_nodes = expl.get_variable_nodes_of_given_type(ExampleHandle).tolist()
print("Handle variable nodes:", handle_nodes)
```

### What were the actual bound values?

Returns the concrete domain objects that were bound to variable nodes of the given type.

```{code-cell} ipython3
bound_handles = expl.get_values_of_variable_nodes_of_given_type(ExampleHandle).tolist()
print("Bound handle values:", bound_handles)
```

### Which conditions relate variables of two different types?

Returns satisfied conditions that have at least one descendant variable node of each given type.
Here the condition `fixed.child == handle_var` joins an `ExampleFixedConnection` variable to an
`ExampleHandle` variable, so it should appear in the result.

```{code-cell} ipython3
cross = expl.get_conditions_that_relate_variables_of_types(
    ExampleFixedConnection, ExampleHandle
).tolist()
for c in cross:
    print(c)
```

The method is symmetric — swapping the two types returns the same set:

```{code-cell} ipython3
cross_reversed = expl.get_conditions_that_relate_variables_of_types(
    ExampleHandle, ExampleFixedConnection
).tolist()
print("Same result reversed:", cross_reversed == cross)
```

### Which conditions relate two variables of the same type?

Returns conditions that contain at least two **distinct** variable nodes of the given type in
their descendant tree. In this query there is only one `ExampleHandle` variable, so no condition
can relate two of them — the result is empty:

```{code-cell} ipython3
same_type = expl.get_conditions_that_relate_the_variables_of_type(ExampleHandle).tolist()
print("Same-type handle conditions:", same_type)  # []
```

To get non-empty results here, write a query that contains two distinct variables of the same
type and a condition that references both (e.g. `handle_var1.name != handle_var2.name`).

---

## API Reference
- {py:func}`~krrood.entity_query_language.explanation.explanation.explain_inference`
- {py:class}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.as_string`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_satisfied_conditions_as_string`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.condition_graph`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_satisfied_condition_expressions_for_the_instance`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_variable_nodes_of_given_type`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_values_of_variable_nodes_of_given_type`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_conditions_that_relate_the_variables_of_type`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_conditions_that_relate_variables_of_types`
