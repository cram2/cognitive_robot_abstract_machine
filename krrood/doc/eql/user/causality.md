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

# Causal (`do()`) Queries

Ice cream sales and drowning incidents both rise in summer. Looking at the data alone,
more ice cream sold comes with more drownings, but banning ice cream would not save
anyone; warm weather drives both. *Observing* a value tells you what tends to come with
it; *setting* a value tells you what actually follows from it. The first is ordinary
probabilistic conditioning (`X=`); Pearl calls the second "intervention" (`do(X)`) and
argued the two are different questions whenever two fields share a hidden common cause
-- conditioning picks up the confounding correlation, intervention cuts it
{cite:t}`pearl2009causality`.

{py:data}`~krrood.entity_query_language.factories.cause` and
{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` route a query
through
{py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`'s
backdoor-adjustment machinery instead of plain conditioning, when evaluated with
{py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`.

### Seeing the difference

A minimal model makes the distinction concrete. `season` drives both `ice_cream_sales`
and `drowning_incidents` -- warm days push both higher, cold days push both lower -- but
`ice_cream_sales` also varies somewhat independently of season, so it is not fully
determined by it:

```python
import math
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.utils import MissingDict
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit, ProductUnit, SumUnit, leaf,
)
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit, MarginalDeterminismTreeNode,
)
from enum import Enum, auto

class Season(Enum):
    WARM = auto()
    COLD = auto()

class DrowningLevel(Enum):
    HIGH = auto()
    LOW = auto()

# Named "SummerStatistics.<field>" up front so this same circuit can later be handed
# to a DictRegistry keyed on the SummerStatistics class defined below -- DictRegistry
# looks a query's fields up by this qualified name, unlike CausalCircuitRegistry.
season = Symbolic("SummerStatistics.season", domain=Set.from_iterable(Season))
ice_cream_sales = Continuous("SummerStatistics.ice_cream_sales")
drowning_incidents = Symbolic(
    "SummerStatistics.drowning_incidents", domain=Set.from_iterable(DrowningLevel)
)

circuit = ProbabilisticCircuit()
root = SumUnit(probabilistic_circuit=circuit)
# (season, ice_cream_sales range, drowning_incidents, branch weight): ice_cream_sales
# is HIGH 70% of the time in WARM, only 30% in COLD -- real overlap, not a deterministic
# tie -- while drowning_incidents follows season alone, regardless of ice_cream_sales.
branches = [
    (Season.WARM, (9, 10), DrowningLevel.HIGH, 0.35),
    (Season.WARM, (0, 1), DrowningLevel.HIGH, 0.15),
    (Season.COLD, (0, 1), DrowningLevel.LOW, 0.35),
    (Season.COLD, (9, 10), DrowningLevel.LOW, 0.15),
]
for season_value, ice_cream_range, drowning_value, weight in branches:
    branch = ProductUnit(probabilistic_circuit=circuit)
    branch.add_subcircuit(leaf(
        SymbolicDistribution(variable=season, probabilities=MissingDict(float, {hash(season_value): 1.0})),
        circuit,
    ))
    branch.add_subcircuit(leaf(
        UniformDistribution(variable=ice_cream_sales, interval=closed(*ice_cream_range).simple_sets[0]),
        circuit,
    ))
    branch.add_subcircuit(leaf(
        SymbolicDistribution(variable=drowning_incidents, probabilities=MissingDict(float, {hash(drowning_value): 1.0})),
        circuit,
    ))
    root.add_subcircuit(branch, math.log(weight))

causal_circuit = CausalCircuit.from_probabilistic_circuit(
    circuit,
    MarginalDeterminismTreeNode.from_causal_graph([ice_cream_sales], [drowning_incidents]),
    [ice_cream_sales],
    [drowning_incidents],
)

high_ice_cream = SimpleEvent.from_data({ice_cream_sales: closed(9, 10).simple_sets[0]}).as_composite_set()
high_drowning = SimpleEvent.from_data({drowning_incidents: Set.from_iterable([DrowningLevel.HIGH])}).as_composite_set()

# Conditioning: observe high ice cream sales, ask about drowning incidents.
conditioned, _ = circuit.truncated(high_ice_cream.fill_missing_variables_pure(circuit.variables))
conditioned.probability(high_drowning.fill_missing_variables_pure(conditioned.variables))
# 0.7 -- conditioning picks up the confound: high sales lean warm, which leans toward
# high drowning incidents.

# do(): backdoor_adjustment reconstructs P(drowning | do(ice cream sales = v)) for
# every v at once, as one joint circuit -- probability() over the whole thing (not
# narrowed to any one v) reads off its average over every possible intervention.
interventional = causal_circuit.backdoor_adjustment(ice_cream_sales, drowning_incidents)
interventional.probability(high_drowning.fill_missing_variables_pure(interventional.variables))
# 0.5 -- back to the unconfounded baseline: unlike conditioning, this number does not
# shift toward "high" just because ice cream sales does.
```

Conditioning and intervention give different answers on the exact same model -- that is
the distinction `cause` and `causes_effect()` exist to make queryable.

```{note}
Neither number here is *"if I specifically forced ice cream sales to be high, what would
happen to drowning"* -- narrowing `interventional` to a specific value of
`ice_cream_sales` needs the confounding `season` variable passed as an
`adjustment_variables=[season]` argument to `backdoor_adjustment`, so it can be summed
back out rather than left baked into the correlation. Without an explicit adjustment
variable, a *narrowed* intervention query cannot be told apart from conditioning; only
the *unnarrowed* one (shown above) is meaningful. See
[Adjusting for a confounder](#adjusting-for-a-confounder) below for exactly this, done
through EQL's `cause`/`causes_effect()`/`confounder` instead of calling
`backdoor_adjustment` directly.
```

### Doing this with EQL queries

The conditioning half runs through EQL exactly as any other query -- register the plain
`ProbabilisticCircuit` and use an ordinary `.where(...)`:

```python
from dataclasses import dataclass

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.model_registries import DictRegistry


@dataclass
class SummerStatistics:
    season: Season
    ice_cream_sales: float
    drowning_incidents: DrowningLevel


backend = ProbabilisticBackend(
    model_registry=DictRegistry({SummerStatistics: circuit}), number_of_samples=1000
)
high_ice_cream = a(SummerStatistics)(season=..., ice_cream_sales=..., drowning_incidents=...)
high_ice_cream.where(high_ice_cream.variable.ice_cream_sales >= 9.0)
results = high_ice_cream.tolist(backend=backend)
sum(r.drowning_incidents == DrowningLevel.HIGH for r in results) / len(results)
# approximately 0.7 -- matches the direct computation above, up to sampling noise: this
# estimates the fraction from drawn samples rather than computing it exactly, since
# average()/count() aggregation is not supported against ProbabilisticBackend.
```

The intervention half is where the note above matters: `cause`/`causes_effect()`
would search for the `ice_cream_sales` region that best explains high drowning on this
same, unadjusted model -- and, for exactly the reason given above, would find the same
confounded answer conditioning already gives, unless `season` is also marked
`confounder`. See the next section for exactly that.

(adjusting-for-a-confounder)=

### Adjusting for a confounder

`cause` fields are searched with an empty adjustment set by default -- fine when
nothing confounds the candidate and the effect, wrong otherwise. Mark a known
confounder with `confounder` and it is passed to `backdoor_adjustment` as its
adjustment set, the same `Z` from the formula above --
on the very same `SummerStatistics` model used above, this time through
{py:class}`~krrood.parametrization.model_registries.CausalCircuitRegistry`, which hands
the query the already-built `causal_circuit` directly instead of a plain
`ProbabilisticCircuit`:

```python
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.parametrization.model_registries import CausalCircuitRegistry

backend = ProbabilisticBackend(
    model_registry=CausalCircuitRegistry({SummerStatistics: causal_circuit})
)

# Without confounder: the search can't separate the confound from the effect.
naive = a(SummerStatistics)(season=..., ice_cream_sales=cause, drowning_incidents=...)
naive.causes_effect(naive.variable.drowning_incidents == DrowningLevel.HIGH)
backend.rank_causes(naive)[0].effect_probability_given_region
# 0.7 -- spurious, the same number plain conditioning on ice_cream_sales gave above.

# With confounder: season is summed back out, recovering the causal truth.
adjusted = a(SummerStatistics)(season=confounder, ice_cream_sales=cause, drowning_incidents=...)
adjusted.causes_effect(adjusted.variable.drowning_incidents == DrowningLevel.HIGH)
backend.rank_causes(adjusted)[0].effect_probability_given_region
# 0.5 -- back to the unconfounded baseline, correctly showing ice_cream_sales has no
# real effect on drowning_incidents once season is accounted for.
```

```{important}
`confounder` only reliably deconfounds when the marked variable's own distribution
*overlaps* across the confounder's states -- as `ice_cream_sales` does here (both
`HIGH` and `LOW` ranges occur, with different probabilities, in both seasons). A cause
variable whose value is deterministically tied to which branch it came from (no
overlap) has nothing for an adjustment set to correct, the same as an empty one.
```

---

## Declaring a causal query

`cause` and `causes_effect()` are always used together: `cause` marks the field to
search an intervention over, `causes_effect()` declares the condition that intervention
should explain.

```python
from krrood.entity_query_language.factories import a, cause
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

pick = a(Pick)(arm=cause, outcome=...)
pick.causes_effect(pick.variable.outcome == Status.SUCCESS)
verbalize_expression(pick)
# 'Generate a Pick and predict its arm and outcome values where its arm causes
#  its outcome to be SUCCESS'
```

### `cause`

`cause` always means *find the value of this field whose intervention (`do(arm=value)`)
best explains the declared effect*. To pin a known value instead -- an ordinary
conditioning assignment, not a causal one -- use a plain literal kwarg (`arm=0.3`).

```{important}
`cause` needs a declared effect to search *for* -- see `causes_effect()` below.
Using `cause` with no `causes_effect(...)` condition anywhere in the query raises
{py:class}`~krrood.entity_query_language.exceptions.NoCausesEffectConditionForCause`.
Declare exactly one effect per query;
{py:class}`~krrood.parametrization.exceptions.MultipleEffectVariablesNotSupported`
raises otherwise -- there is no multi-effect form of the underlying interventional
computation to route several through. Multiple `cause` fields are fine: each candidate
is searched independently and the one that best explains the effect becomes the primary
cause -- see
{py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes`
below to see every candidate's score, not just the primary one. Mark a known
confounder with `confounder` to have it summed out of every candidate's score --
see [Adjusting for a confounder](#adjusting-for-a-confounder) above.
```

### `causes_effect()`

{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` is sugar for
`.where(...)`: it accepts one equality comparator (`attribute == value`) or several
combined with `and_`, declaring exactly one effect variable per query. It filters
results **identically** to an ordinary `.where()` under every backend, including
selective ones, so `causes_effect(...)` never changes what a query would otherwise
select -- only `ProbabilisticBackend` additionally reads it, to know which variable to
compute `P(effect | do(cause))` over.

---

## Backend behaviour

### `ProbabilisticBackend`

Requires a model registry that resolves a
{py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
for the queried class -- {py:class}`~krrood.parametrization.model_registries.CausalCircuitRegistry`
maps classes directly to pre-built causal circuits:

```python
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.model_registries import CausalCircuitRegistry

backend = ProbabilisticBackend(
    model_registry=CausalCircuitRegistry({Pick: pick_causal_circuit}),
)
results = list(pick.evaluate(backend=backend))
assert all(r.outcome == Status.SUCCESS for r in results)
```

If the registry resolves anything other than a `CausalCircuit` for a query containing
`cause`, the backend raises
{py:class}`~krrood.parametrization.exceptions.DoRequiresCausalCircuitModel` --
`cause` needs a registered causal graph to know what to cut when intervening; there
is no fallback to plain conditioning. Confounder adjustment is configured on however
the registered `CausalCircuit` itself was built, not from the query.

#### Ranking every candidate

`evaluate()` only ever narrows to the single best-scoring `cause` candidate. When more
than one field is plausibly responsible -- a pick failure that could trace back to
either `arm` or `grip` -- and how the candidates compare matters, not just which one
wins,
{py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes`
returns every candidate's score instead:

```python
match = a(Pick)(arm=cause, grip=cause, outcome=...)
match.causes_effect(match.variable.outcome == Status.SUCCESS)

ranking = backend.rank_causes(match)
for scored in ranking:
    print(scored.cause_variable.name, scored.effect_probability_given_region)
# arm   0.94
# grip  0.61
```

Every candidate is still searched independently, exactly as `evaluate()` already does
internally -- `rank_causes` runs the same per-candidate search, it just keeps every
result instead of discarding all but the best. It is still not a joint intervention:
this ranks how well each candidate explains the effect on its own, not how well
intervening on several of them together would.

### Selective backends and `EntityQueryLanguageGenerativeBackend`

Neither has a notion of a causal graph. Rather than failing the whole query, they
**warn** (via `krrood.logger`) and treat `cause` exactly like an ordinary
unspecified field (`...`) -- a selective backend naturally selects nothing (nothing
equals `cause`'s wrapped `Ellipsis`), and the generative backend enumerates it if the
field is an enum, or raises the same
{py:class}`~krrood.entity_query_language.exceptions.UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration`
a bare `...` on a non-enum field already raises. Pass
`raise_on_unresolvable_cause=True` to a backend's constructor to fail loudly instead --
useful in tests that want to catch accidental `cause` misuse against a non-causal
backend.

---

## API Reference

- {py:data}`~krrood.entity_query_language.factories.cause`
- {py:data}`~krrood.entity_query_language.operators.causal.cause`
- {py:class}`~krrood.entity_query_language.operators.causal.Cause`
- {py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect`
- {py:class}`~krrood.entity_query_language.operators.causal.CausesEffect`
- {py:class}`~krrood.parametrization.model_registries.CausalCircuitRegistry`
- {py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
- {py:class}`~krrood.parametrization.exceptions.DoRequiresCausalCircuitModel`
- {py:class}`~krrood.parametrization.exceptions.MultipleEffectVariablesNotSupported`
- {py:class}`~krrood.entity_query_language.exceptions.NoCausesEffectConditionForCause`
- {py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes`
- {py:class}`~krrood.entity_query_language.exceptions.NoCauseVariablesForRanking`
- {py:data}`~krrood.entity_query_language.factories.confounder`
- {py:data}`~krrood.entity_query_language.operators.causal.confounder`
- {py:class}`~krrood.entity_query_language.operators.causal.Confounder`
