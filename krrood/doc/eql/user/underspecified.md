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

# Underspecified Queries and Generative Backends

The {py:func}`~krrood.entity_query_language.factories.underspecified` function is the entry point
for *generative* queries.  Where {py:func}`~krrood.entity_query_language.factories.match` **finds**
objects that already exist in a domain, `underspecified` **creates** new objects whose field values
are inferred by a generative backend.

This page explains:

1. How to build an underspecified query.
2. What each type of field specification means.
3. The crucial difference between specifying something in the **factory kwargs** versus in a
   **`.where()` condition**.
4. How to run the query with the {py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`.

---

## Building an Underspecified Query

Call `underspecified(MyClass)` exactly like you would call `match(MyClass)` — then immediately
invoke the result with keyword arguments that constrain each field:

```python
from krrood.entity_query_language.factories import underspecified

query = underspecified(KRROODPosition)(x=..., y=..., z=...)
```

The returned object is a {py:class}`~krrood.entity_query_language.query.match.Match` and supports
the same `.where()`, `.resolve()`, and nesting patterns as `match`.

---

## Field Specification: Three Kinds of Values

Every keyword argument passed to the factory call carries a distinct meaning for the generative
backend.

### Ellipsis (`...`) — free variable

```python
query = underspecified(KRROODPosition)(x=..., y=..., z=...)
```

An `Ellipsis` marks the field as *unconstrained*.  The backend samples a value for that dimension
from the **marginal** of the probabilistic model — no restriction is applied.  Use `...` whenever
you genuinely have no prior knowledge about a field.

```{important}
`...` can only be used on fields whose Python type appears in
{py:data}`random_events.variable.compatible_types` — currently `int`, `float`, `bool`, and
`enum.Enum` subclasses.  For structured/nested fields, nest another `underspecified(...)` call
instead of using `...`.
```

### Concrete literal — conditioning assignment

```python
query = underspecified(KRROODPosition)(x=0.5, y=..., z=...)
```

A concrete value becomes a **conditioning assignment**: the model is updated to
`P(y, z | x = 0.5)` before any samples are drawn.  This is the Bayesian "evidence"
operation — you know the exact value of the field and you want the rest of the sample
to be consistent with it.

```{note}
This is fundamentally different from a `.where(query.variable.x == 0.5)` filter (see below).
A kwargs literal conditions the *model* first; a `.where` condition filters the *samples*
after the model has already been conditioned.
```

### `variable(...)` with a domain — truncation assignment

```python
from krrood.entity_query_language.factories import variable

query = underspecified(KRROODPosition)(x=..., y=..., z=variable(int, domain=[1, 2, 3]))
```

A KRROOD {py:func}`~krrood.entity_query_language.factories.variable` passes the field's domain
as a **truncation constraint** to the model.  All other dimensions remain unaffected.

---

## Factory kwargs vs. `.where()` — the key difference

Both kwargs and `.where()` constrain what the backend produces, but they operate at
**different stages** of the probabilistic pipeline:

| | **Factory kwargs** | **`.where()` condition** |
|---|---|---|
| **When applied** | Before sampling — modifies the model | After conditioning — truncates the (already-conditioned) distribution |
| **What it expresses** | A constraint on a *single field* | A constraint that may relate *multiple fields* |
| **Probabilistic effect** | Conditioning (`P(rest \| field = value)`) or truncation of one marginal | Truncation of the joint distribution to a region |
| **Typical use** | "Fix `x` to 0.5", "restrict `z` to {1,2,3}" | "`x > 0.5`", "`type > charge`" |

### When to use kwargs

Use factory kwargs when you know the value of a field (or its allowed set) **independently** of
the other fields.  The generative model gets updated with this information first, so the remaining
free fields are sampled in a way that is statistically consistent with your constraint.

### When to use `.where()`

Use `.where()` when your constraint involves a **relationship between two or more fields**, or an
**inequality range** on a single field that cannot be expressed as a finite set.  The `.where()`
condition is translated into a truncation event that carves a region out of the already-conditioned
joint distribution.

```python
query = underspecified(KRROODPose)(
    position=underspecified(KRROODPosition)(x=..., y=..., z=...),
    orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),  # ← kwargs: condition the model
)
query.resolve()
query.where(query.variable.position.x > 0.5)   # ← .where(): truncate the conditioned distribution
```

Here `orientation` is fixed via kwargs (Bayesian conditioning), while `x > 0.5` is a
post-conditioning truncation because it describes a half-open interval rather than a single value.

```{note}
**Performance of `.where()` conditions depends on the algebra of the constraint.**
If your condition belongs to the *product algebra* (axis-aligned box constraints such as
`x > 0.5` or `z in {1, 2, 3}`), the truncation is computed exactly and sampling remains fast.
Conditions that fall *outside* the product algebra (e.g. `field_a > field_b`) cannot be
expressed as a closed-form event and require **rejection sampling**, which is slower and may
need many more samples to produce the requested number of results.
```

---

## Running the Query with ProbabilisticBackend

Once the query is built, pass it to a backend to obtain instances:

```{code-cell} ipython3
:tags: [raises-exception]

from dataclasses import dataclass
from krrood.entity_query_language.factories import underspecified, variable
from krrood.entity_query_language.backends import ProbabilisticBackend

@dataclass
class Position:
    x: float
    y: float
    z: float

# All fields free — samples from the prior
query = underspecified(Position)(x=..., y=..., z=...)

backend = ProbabilisticBackend(number_of_samples=5)
positions = list(backend.evaluate(query))
for p in positions:
    print(p)
```

### Example: conditioning on a literal

```python
# Fix x=0.5; sample y and z from P(y, z | x=0.5)
query = underspecified(Position)(x=0.5, y=..., z=...)
positions = list(ProbabilisticBackend(number_of_samples=5).evaluate(query))
assert all(p.x == 0.5 for p in positions)
```

### Example: restricting a field to a discrete set

```python
# z is sampled only from {1, 2, 3}
query = underspecified(Position)(x=..., y=..., z=variable(int, domain=[1, 2, 3]))
positions = list(ProbabilisticBackend(number_of_samples=5).evaluate(query))
assert all(p.z in (1, 2, 3) for p in positions)
```

### Example: cross-field constraint via `.where()`

```python
# Sample only poses where position.x > 0.5.
# The orientation is fixed by conditioning (kwargs); the x constraint is a range filter (.where).
query = underspecified(Pose)(
    position=underspecified(Position)(x=..., y=..., z=...),
    orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
)
query.resolve()
query.where(query.variable.position.x > 0.5)

poses = list(ProbabilisticBackend(number_of_samples=10).evaluate(query))
assert all(p.position.x > 0.5 for p in poses)
```

### Example: nested underspecified objects

Complex action types can be described by nesting multiple `underspecified` calls.
Each level of nesting introduces its own field constraints independently:

```python
query = underspecified(NestedAction)(
    obj=apple,                                         # condition: exactly this apple instance
    pose=underspecified(Pose)(
        position=underspecified(Position)(
            x=0.02,                                    # condition: x is exactly 0.02
            y=...,                                     # free
            z=...,                                     # free
        ),
        orientation=underspecified(Orientation)(
            x=..., y=..., z=...,
            w=variable(float, domain=[0.0, 1.0]),      # truncate: w in {0.0, 1.0}
        ),
    ),
)
```

- `obj=apple` is a **literal** — it conditions the model on that exact `Apple` instance.
  Using `variable(Apple, domain=[apple])` would instead create a truncation constraint, which
  has different probabilistic semantics.
- `position.x = 0.02` conditions the model: `P(y, z, orientation | x=0.02)`.
- `orientation.w` is truncated to `{0.0, 1.0}`.
- All remaining fields are sampled freely.

---

## Using a Custom Model Registry (e.g. a Learned JPT)

By default, `ProbabilisticBackend` uses a
{py:class}`~krrood.parametrization.model_registries.FullyFactorizedRegistry` that returns an
independent (fully-factorized) model over the free variables.  For realistic robot tasks you
usually want a **learned** model that captures correlations between fields — for example a
*Joint Probability Tree* (JPT) trained on recorded execution data.

You can inject a pre-trained
{py:class}`~probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit.ProbabilisticCircuit`
via a {py:class}`~krrood.parametrization.model_registries.DictRegistry`, which maps each target
class to its own model:

```python
import json
from krrood.adapters.json_serializer import from_json
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.parametrization.model_registries import DictRegistry
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit

# Load a pre-trained circuit (e.g. a JPT saved to disk)
with open("nested_action_model.json", "r") as f:
    learned_model: ProbabilisticCircuit = from_json(json.load(f))

# Register the model for the target class
registry = DictRegistry({NestedAction: learned_model})

# Build the underspecified query as usual
query = underspecified(NestedAction)(
    obj=Body(name="mug"),
    pose=underspecified(KRROODPose)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
    ),
)

# Pass the registry to the backend
backend = ProbabilisticBackend(registry, number_of_samples=10)
samples = list(backend.evaluate(query))
```

The backend calls `registry.get_model(parameters)` to retrieve the model before conditioning and
sampling, so swapping in a JPT or any other
{py:class}`~probabilistic_model.probabilistic_model.ProbabilisticModel` requires no changes to the
query itself.

---

## API Reference

- {py:func}`~krrood.entity_query_language.factories.underspecified`
- {py:func}`~krrood.entity_query_language.factories.match`
- {py:class}`~krrood.entity_query_language.query.match.Match`
- {py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`
- {py:class}`~krrood.parametrization.parameterizer.UnderspecifiedParameters`
- {py:class}`~krrood.parametrization.model_registries.ModelRegistry`
- {py:class}`~krrood.parametrization.model_registries.DictRegistry`
- {py:data}`~random_events.variable.compatible_types`
