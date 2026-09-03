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

# Probabilistic Queries

Consider a warehouse robot fleet. Every robot streams back telemetry -- speed, motor
temperature, battery charge -- and a probabilistic model has already been fitted to a
long history of that telemetry: given how a shift starts, what do the readings tend to
look like over the course of it? Nothing here builds that model; the questions below
are answered from it.

```python
from dataclasses import dataclass

@dataclass
class Robot:
    temperature: float
    speed: float
    charge: float
```

`set_of`/`entity` answer questions about *actual robots* -- "find me the robots
running hot right now". The questions below are different: not about which robots
exist, but about the *model itself*. Three constructs answer that kind of question:

| Question | Construct | Answer |
|---|---|---|
| "On average, how fast do robots move?" | {py:func}`~krrood.entity_query_language.factories.average` | a number |
| "What fraction of robots run dangerously hot?" | {py:func}`~krrood.entity_query_language.factories.probability_of` | a number |
| "Given today's charge level, what does the rest look like?" | {py:func}`~krrood.entity_query_language.factories.distribution_of` | a distribution |

Without live telemetry on hand -- only the fitted model -- all three questions below
get answered from it, via
{py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`, the backend
that knows how to reach it:

```python
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit, ProductUnit, leaf,
)
from random_events.interval import closed
from random_events.variable import Continuous

# Stand-in for the fleet's already-fitted model: over a shift, in isolation, each
# reading varies uniformly across its own range. (Named "Robot.<field>" so
# DictRegistry -- see below -- can look each field up by name.)
var_temperature = Continuous("Robot.temperature")
var_speed = Continuous("Robot.speed")
var_charge = Continuous("Robot.charge")

circuit = ProbabilisticCircuit()
root = ProductUnit(probabilistic_circuit=circuit)
root.add_subcircuit(leaf(UniformDistribution(variable=var_temperature, interval=closed(0, 1).simple_sets[0]), circuit))
root.add_subcircuit(leaf(UniformDistribution(variable=var_speed, interval=closed(0, 2).simple_sets[0]), circuit))
root.add_subcircuit(leaf(UniformDistribution(variable=var_charge, interval=closed(0, 3).simple_sets[0]), circuit))

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, average, distribution_of, probability_of, variable
from krrood.parametrization.model_registries import DictRegistry

backend = ProbabilisticBackend(model_registry=DictRegistry({Robot: circuit}))
```

## "On average, how fast do robots move?"

The simplest question, and it uses a construct already familiar from ordinary
querying -- `average`, the same one that averages a column of matched rows. Called
*bare* (no `set_of`, no `.grouped_by()`) against a model-backed backend, it answers
straight from the model instead of averaging a sample of rows:

```python
x = variable(Robot)
print(average(x.speed).first(backend=backend))
# 1.0
```

That's it -- same function used in an ordinary query, just pointed at a model instead
of a table of robots.

## "What fraction of robots run dangerously hot?"

Suppose anything above 0.5 counts as running hot.

```python
x = variable(Robot)
print(probability_of(x.temperature > 0.5).first(backend=backend))
# 0.5
```

There's a simpler way to think about this, already familiar from ordinary querying:
take a batch of robots, count how many are running hot, divide by how many robots
there are. That *is* what a probability means, so `probability_of` can answer it
exactly that way too -- when live telemetry is on hand instead of the fitted model:

```python
robots_online_now = [
    Robot(temperature=0.2, speed=1.0, charge=2.0),
    Robot(temperature=0.4, speed=1.0, charge=2.0),
    Robot(temperature=0.6, speed=1.0, charge=2.0),
    Robot(temperature=0.9, speed=1.0, charge=2.0),
]

y = variable(Robot, domain=robots_online_now)
print(probability_of(y.temperature > 0.5).first())
# 0.5 -- 2 of the 4 robots above are running hot
```

Same construct, same syntax -- `probability_of` picks whichever strategy fits the
backend it's given: count the online fleet when it's on hand (no fitted model needed
at all), or read the exact answer straight off the model when there's no live fleet to
count, only the model (as with `backend` above). The counting answer is only ever as
good as the fleet it's counting; the model-based one is exact regardless of fleet
size, because there never was a fleet to begin with.

## "Given today's charge level, what does the rest look like?"

Every robot left the dock at a 1.5 charge level today, and anything moving slower than
0.2 has already been screened out. The question isn't for a number now -- it's for the
*shape* of what's left: how temperature and speed vary once charge is pinned down and
the too-slow robots are gone.

This is exactly the kind of thing `a(...)`/`an(...)`/`the(...)` already describe -- the
same shape of match would normally *generate* example robots that fit these
conditions. `distribution_of` asks the identical question, just answered differently:
instead of generating robots, it hands back the distribution itself.

```python
match = a(Robot)(temperature=..., speed=..., charge=1.5)
match.where(match.variable.speed > 0.2)

result = distribution_of(match).first(backend=backend)
print(sorted(v.name for v in result.variables))
# ['Robot.charge', 'Robot.speed', 'Robot.temperature']
```

`charge=1.5` is today's dock charge level; `.where(speed > 0.2)` is the screening
step; `temperature`/`speed` are left open (`...`) -- what the shape should be seen of.
The result is an ordinary distribution object, queryable like the fleet's original
model:

```python
from random_events.product_algebra import SimpleEvent

speed_range = SimpleEvent.from_data({var_speed: closed(0.2, 2)}).as_composite_set()
print(result.probability(speed_range))
# 1.0 -- everything left satisfies speed > 0.2, by construction

temperature_range = SimpleEvent.from_data({var_temperature: closed(0, 0.5)}).as_composite_set()
print(result.probability(temperature_range))
# 0.5 -- unchanged: temperature never depended on speed or charge to begin with
```

To narrow down to temperature specifically:

```python
narrowed = distribution_of(match, marginalize_for=(match.variable.temperature,)).first(backend=backend)
print({v.name for v in narrowed.variables})
# {'Robot.temperature'}
```

## Recap

- **`average(x.speed)`** -- the same aggregator used for row-averaging; averages a
  real fleet when one is on hand, or reads the model's exact expectation when there's
  only a fitted model.
- **`probability_of(condition)`** -- counts a real fleet's matching robots when one is
  on hand, or reads the exact answer straight off the model when there's only a fitted
  model. Accepts any condition a `.where(...)` clause does.
- **`distribution_of(match, marginalize_for=...)`** -- the same match that would
  normally *generate* robots, answered with the shape of the distribution instead of
  samples from it. Optional `marginalize_for` narrows which readings the answer
  covers. Unlike the other two, there's no batch-counting equivalent for a whole
  distribution, so this one only ever works with a fitted model behind it.

Two of the names should look familiar already: `distribution_of`/`average` are how
`a(...)`/`an(...)`/`the(...)` and `average(...)` already read, just capable of being
answered from a model instead of only ever a table.

---

## API Reference

- {py:func}`~krrood.entity_query_language.factories.distribution_of`
- {py:func}`~krrood.entity_query_language.factories.probability_of`
- {py:func}`~krrood.entity_query_language.factories.average`
- {py:class}`~krrood.entity_query_language.operators.probabilistic_queries.ProbabilisticQuery`
- {py:class}`~krrood.entity_query_language.operators.probabilistic_queries.Distribution`
- {py:class}`~krrood.entity_query_language.operators.probabilistic_queries.Probability`
- {py:class}`~krrood.entity_query_language.operators.aggregators.Average`
- {py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend._resolve_average`
- {py:meth}`~krrood.parametrization.parameterizer.UnderspecifiedParameters.resolve_conditioned_and_truncated_model`
- {py:class}`~krrood.parametrization.parameterizer.ModelQueryParameters`
- {py:class}`~krrood.entity_query_language.exceptions.BackendCannotEvaluateProbabilisticQuery`
- {py:class}`~krrood.parametrization.exceptions.JointQueryAcrossClassesNotSupported`
