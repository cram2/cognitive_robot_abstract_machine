import math
from dataclasses import dataclass
from enum import Enum, auto

import pytest
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict
from random_events.interval import closed
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import (
    NoCauseVariablesForRanking,
    NoCausesEffectConditionForCause,
)
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.parametrization.exceptions import (
    DoRequiresCausalCircuitModel,
    MultipleEffectVariablesNotSupported,
)
from krrood.parametrization.model_registries import (
    CausalCircuitRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters


class Outcome(Enum):
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class Pick:
    arm: float
    outcome: Outcome


def _build_two_region_causal_circuit() -> tuple:
    """
    A causal circuit where `arm`'s intervention region determines `outcome`'s value:
    two equal-weight mixture components, `arm`/`outcome` co-varying together.

        Low:  arm in [0, 1], outcome = FAILURE (deterministically)
        High: arm in [2, 3], outcome = SUCCESS (deterministically)

    Ground truth: interventionally forcing `arm` into the high region is what makes
    `outcome` equal SUCCESS -- the query
    `a(Pick)(arm=cause, ...).causes_effect(outcome == Outcome.SUCCESS)` should
    therefore only ever return instances with `arm` in `[2, 3]`.
    """
    arm = Continuous("Pick.arm")
    outcome = Symbolic("Pick.outcome", domain=Set.from_iterable(Outcome))
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for arm_range, outcome_value in [
        ((0, 1), Outcome.FAILURE),
        ((2, 3), Outcome.SUCCESS),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=arm, interval=closed(*arm_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=outcome,
                    probabilities=MissingDict(float, {hash(outcome_value): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph([arm], [outcome]),
        [arm],
        [outcome],
    )
    return causal_circuit, arm, outcome


@dataclass
class TwoCauseCandidatesCircuit:
    """
    A causal circuit with two cause candidates of unequal explanatory strength for the
    same effect, and the variables/circuit needed to query and verify it.
    """

    causal_circuit: CausalCircuit
    decisive_cause: Continuous
    """
    `decisive_cause`'s region alone almost perfectly determines `outcome`.
    """

    uninformative_cause: Continuous
    """
    `uninformative_cause` has the same distribution regardless of `outcome`.
    """

    outcome: Symbolic


def _build_two_cause_candidates_circuit() -> TwoCauseCandidatesCircuit:
    """
    Two equal-weight mixture components over three variables:

        Low:  decisive in [0, 1], uninformative in [0, 2], outcome = FAILURE
        High: decisive in [2, 3], uninformative in [0, 2], outcome = SUCCESS

    `decisive` separates the components perfectly (its regions [0, 1] / [2, 3] each
    occur with exactly one outcome), so restricting it to its best region ([2, 3])
    makes `P(outcome == SUCCESS | do(decisive in [2, 3]))` near-certain (~1.0).
    `uninformative` has the identical range [0, 2] in both components, so its own
    support forms a single, whole-domain region compatible with either outcome --
    restricting it to that region changes nothing, so
    `P(outcome == SUCCESS | do(uninformative in [0, 2]))` stays at the prior, 0.5.
    `decisive` should therefore always win as the primary cause.
    """
    decisive = Continuous("Pick.arm")
    uninformative = Continuous("Pick.grip")
    outcome = Symbolic("Pick.outcome", domain=Set.from_iterable(Outcome))
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for decisive_range, outcome_value in [
        ((0, 1), Outcome.FAILURE),
        ((2, 3), Outcome.SUCCESS),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=decisive, interval=closed(*decisive_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=uninformative, interval=closed(0, 2).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=outcome,
                    probabilities=MissingDict(float, {hash(outcome_value): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph(
            [decisive, uninformative], [outcome]
        ),
        [decisive, uninformative],
        [outcome],
    )
    return TwoCauseCandidatesCircuit(causal_circuit, decisive, uninformative, outcome)


# %% error handling


def test_raises_when_model_registry_does_not_resolve_a_causal_circuit():
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    backend = ProbabilisticBackend(model_registry=FullyFactorizedRegistry())
    with pytest.raises(DoRequiresCausalCircuitModel):
        list(match.evaluate(backend=backend))


def test_raises_when_cause_has_no_causes_effect_condition():
    causal_circuit, _, _ = _build_two_region_causal_circuit()
    match = a(Pick)(arm=cause, outcome=...)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit})
    )
    with pytest.raises(NoCausesEffectConditionForCause):
        list(match.evaluate(backend=backend))


# %% end-to-end correctness


def test_results_satisfy_the_causes_effect_condition():
    causal_circuit, _, _ = _build_two_region_causal_circuit()
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=10,
    )
    results = list(match.evaluate(backend=backend))

    assert len(results) == 10
    for result in results:
        assert result.outcome == Outcome.SUCCESS


def test_results_land_in_the_intervention_region_that_causes_the_effect():
    """
    `arm` must land specifically in [2, 3] -- the region whose intervention causes
    `outcome == SUCCESS` -- not merely anywhere in `arm`'s domain.

    This is the
    cause/effect *correlation* backdoor_adjustment's per-branch ProductUnit structure
    is supposed to preserve; it regressed to marginal-only correctness when its region
    extraction collapsed disjoint branches into one, and is covered directly against
    `CausalCircuit` in `probabilistic_model_test/test_causal/test_causal_circuit.py`
    (`BestDisjointRegionTestCase`).
    """
    causal_circuit, _, _ = _build_two_region_causal_circuit()
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=10,
    )
    results = list(match.evaluate(backend=backend))

    assert len(results) == 10
    for result in results:
        assert 2.0 <= result.arm <= 3.0


def test_pipeline_reproduces_directly_computed_backdoor_adjustment_and_best_region():
    """
    The EQL pipeline's interventional branch is glue over
    `CausalCircuit.backdoor_adjustment` and `CausalCircuit._best_disjoint_region` (both
    untouched, already-tested primitives) -- this checks it wires them together
    correctly by reproducing the same region a direct, hand-written call to those
    primitives selects.
    """
    causal_circuit, arm, outcome = _build_two_region_causal_circuit()
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    parameters = UnderspecifiedParameters(match)
    [cause_variable] = parameters.search_cause_variables
    [effect_variable] = parameters.effect_variables_from_causes_effect
    assert cause_variable == arm
    assert effect_variable == outcome

    # the reference computation: exactly what the plan's design describes as the
    # interventional branch, called directly against the causal circuit
    interventional = causal_circuit.backdoor_adjustment(cause_variable, effect_variable)
    effect_truncated, _ = interventional.truncated(
        parameters.truncation_assignments_from_where_conditions
    )
    expected_best_region = causal_circuit._best_disjoint_region(
        cause_variable, effect_truncated
    )
    expected_narrowed, _ = effect_truncated.truncated(
        expected_best_region.fill_missing_variables_pure(effect_truncated.variables)
    )

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=1,
    )
    scored = backend._score_intervention(
        causal_circuit,
        cause_variable,
        effect_variable,
        parameters.truncation_assignments_from_where_conditions,
    )

    assert scored.cause_variable == cause_variable
    assert scored.narrowed_circuit.probability(
        expected_best_region.fill_missing_variables_pure(
            scored.narrowed_circuit.variables
        )
    ) == pytest.approx(1.0)
    assert expected_narrowed.probability(
        expected_best_region.fill_missing_variables_pure(expected_narrowed.variables)
    ) == pytest.approx(1.0)


# %% multiple cause candidates


def test_multiple_cause_candidates_selects_the_decisive_one_as_primary():
    """
    With two `cause` candidates of unequal explanatory strength for the same effect, the
    primary cause the query narrows to must be the decisive one -- not because joint
    intervention was computed (it wasn't; `backdoor_adjustment` cannot), but because
    each candidate was searched independently and the decisive one scored higher (see
    `_build_two_cause_candidates_circuit`).
    """
    circuit = _build_two_cause_candidates_circuit()
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    parameters = UnderspecifiedParameters(match)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: circuit.causal_circuit})
    )
    primary = backend._resolve_primary_intervention(
        circuit.causal_circuit,
        [circuit.decisive_cause, circuit.uninformative_cause],
        circuit.outcome,
        parameters.truncation_assignments_from_where_conditions,
        match,
    )

    assert primary.cause_variable == circuit.decisive_cause
    assert primary.effect_probability_given_region == pytest.approx(1.0, abs=0.02)


def test_the_uninformative_candidate_scores_lower_than_the_decisive_one():
    circuit = _build_two_cause_candidates_circuit()
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)
    parameters = UnderspecifiedParameters(match)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: circuit.causal_circuit})
    )

    decisive_score = backend._score_intervention(
        circuit.causal_circuit,
        circuit.decisive_cause,
        circuit.outcome,
        parameters.truncation_assignments_from_where_conditions,
    )
    uninformative_score = backend._score_intervention(
        circuit.causal_circuit,
        circuit.uninformative_cause,
        circuit.outcome,
        parameters.truncation_assignments_from_where_conditions,
    )

    assert (
        decisive_score.effect_probability_given_region
        > uninformative_score.effect_probability_given_region
    )


def test_multiple_effect_variables_are_rejected():
    causal_circuit, arm, outcome = _build_two_region_causal_circuit()
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(
        match.variable.outcome == Outcome.SUCCESS, match.variable.arm == 2.5
    )

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit})
    )
    with pytest.raises(MultipleEffectVariablesNotSupported) as excinfo:
        list(match.evaluate(backend=backend))
    assert set(excinfo.value.variables) == {outcome, arm}


# %% rank_causes


@dataclass
class PickAttempt:
    arm: float
    grip: float
    outcome: Outcome


def _build_pick_attempt_ranking_circuit() -> TwoCauseCandidatesCircuit:
    """
    The same two-candidate structure as `_build_two_cause_candidates_circuit`
    (`decisive` separates the outcome perfectly, `uninformative` does not), built for
    :class:`PickAttempt` instead of :class:`Pick` so `rank_causes` tests can query both
    candidates through a real match without touching `Pick`'s own widely-reused fixture.
    """
    decisive = Continuous("PickAttempt.arm")
    uninformative = Continuous("PickAttempt.grip")
    outcome = Symbolic("PickAttempt.outcome", domain=Set.from_iterable(Outcome))
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for decisive_range, outcome_value in [
        ((0, 1), Outcome.FAILURE),
        ((2, 3), Outcome.SUCCESS),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=decisive, interval=closed(*decisive_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=uninformative, interval=closed(0, 2).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=outcome,
                    probabilities=MissingDict(float, {hash(outcome_value): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph(
            [decisive, uninformative], [outcome]
        ),
        [decisive, uninformative],
        [outcome],
    )
    return TwoCauseCandidatesCircuit(causal_circuit, decisive, uninformative, outcome)


def test_rank_causes_returns_every_candidate_ranked_highest_first():
    circuit = _build_pick_attempt_ranking_circuit()
    match = a(PickAttempt)(arm=cause, grip=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({PickAttempt: circuit.causal_circuit})
    )

    ranking = backend.rank_causes(match)

    assert [scored.cause_variable for scored in ranking] == [
        circuit.decisive_cause,
        circuit.uninformative_cause,
    ]
    assert ranking[0].effect_probability_given_region == pytest.approx(1.0, abs=0.02)
    assert ranking[1].effect_probability_given_region == pytest.approx(0.5, abs=0.05)


def test_rank_causes_does_not_change_the_result_of_evaluate():
    """
    `rank_causes` is a read alongside the existing search, not a replacement for it --
    `evaluate()`'s primary-cause selection for the same multi-`cause` match must be
    unaffected by calling `rank_causes` on it.
    """
    circuit = _build_pick_attempt_ranking_circuit()
    match = a(PickAttempt)(arm=cause, grip=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({PickAttempt: circuit.causal_circuit}),
        number_of_samples=10,
    )

    ranking = backend.rank_causes(match)
    results = list(match.evaluate(backend=backend))

    assert ranking[0].cause_variable == circuit.decisive_cause
    assert len(results) == 10
    assert all(2.0 <= result.arm <= 3.0 for result in results)


def test_rank_causes_rejects_a_match_with_no_cause_fields():
    match = a(PickAttempt)(arm=..., grip=..., outcome=...)
    backend = ProbabilisticBackend(model_registry=CausalCircuitRegistry({}))
    with pytest.raises(NoCauseVariablesForRanking):
        backend.rank_causes(match)


# %% whole-query verbalization


@dataclass
class CalibrationAttempt:
    setting: float
    calibrated: bool


def test_a_causal_query_verbalizes_the_causes_effect_clause_in_context():
    match = a(Pick)(arm=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    assert verbalize_expression(match) == (
        "Generate a Pick and predict its arm and outcome values where its arm "
        "causes its outcome to be SUCCESS"
    )


def test_causes_effect_verbalizes_correctly_alongside_an_unrelated_where_condition():
    match = a(PickAttempt)(arm=cause, grip=..., outcome=...)
    match.where(match.variable.grip > 0.5)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    assert verbalize_expression(match) == (
        "Generate a PickAttempt and predict its arm, grip, and outcome values "
        "where its grip is greater than 0.5, and its arm causes its outcome to "
        "be SUCCESS"
    )


def test_causes_effect_verbalizes_a_boolean_effect_naturally():
    # A boolean effect must not fall through the generic "<attribute> to be <value>"
    # template (that reads as a broken double clause, e.g. "a Pick is grasped to be
    # True") -- it reads as "<navigation> to be <predicate>" instead.
    match = a(CalibrationAttempt)(setting=cause, calibrated=...)
    match.causes_effect(match.variable.calibrated == True)

    assert verbalize_expression(match) == (
        "Generate a CalibrationAttempt and predict its setting and calibrated "
        "values where its setting causes the CalibrationAttempt to be calibrated"
    )


def test_causes_effect_verbalization_names_every_cause_candidate():
    match = a(PickAttempt)(arm=cause, grip=cause, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)

    assert verbalize_expression(match) == (
        "Generate a PickAttempt and predict its arm, grip, and outcome values "
        "where its arm and its grip cause its outcome to be SUCCESS"
    )


# %% confounder


class Season(Enum):
    WARM = auto()
    COLD = auto()


class Treatment(Enum):
    LOW = auto()
    HIGH = auto()


@dataclass
class Trial:
    treatment: Treatment
    season: Season
    outcome: Outcome


def _build_confounded_trial_circuit() -> CausalCircuit:
    """
    Season confounds both treatment and outcome; treatment has no causal effect of
    its own. Mirrors `probabilistic_model_test`'s
    `DiscreteConfounderAdjustmentTestCase` fixture, through EQL's own domain classes.

        Warm stratum (p=0.6): P(treatment=HIGH)=0.8, outcome=SUCCESS (deterministic)
        Cold stratum (p=0.4): P(treatment=HIGH)=0.3, outcome=FAILURE (deterministic)

    Ground truth:
        P(outcome=SUCCESS | treatment=HIGH) = 0.8 -- spurious, driven by season.
        P(outcome=SUCCESS | do(treatment=HIGH)) = 0.6 -- causal truth after adjusting
            for season; equal for treatment=LOW too, since treatment has no real
            effect on outcome.
    """
    season = Symbolic("Trial.season", domain=Set.from_iterable(Season))
    treatment = Symbolic("Trial.treatment", domain=Set.from_iterable(Treatment))
    outcome = Symbolic("Trial.outcome", domain=Set.from_iterable(Outcome))

    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for season_value, treatment_high_probability, stratum_weight, outcome_value in [
        (Season.WARM, 0.8, 0.6, Outcome.SUCCESS),
        (Season.COLD, 0.3, 0.4, Outcome.FAILURE),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=season,
                    probabilities=MissingDict(float, {hash(season_value): 1.0}),
                ),
                circuit,
            )
        )
        treatment_distribution = SumUnit(probabilistic_circuit=circuit)
        treatment_distribution.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=treatment,
                    probabilities=MissingDict(float, {hash(Treatment.HIGH): 1.0}),
                ),
                circuit,
            ),
            math.log(treatment_high_probability),
        )
        treatment_distribution.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=treatment,
                    probabilities=MissingDict(float, {hash(Treatment.LOW): 1.0}),
                ),
                circuit,
            ),
            math.log(1 - treatment_high_probability),
        )
        component.add_subcircuit(treatment_distribution)
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=outcome,
                    probabilities=MissingDict(float, {hash(outcome_value): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(stratum_weight))

    return CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph([treatment, season], [outcome]),
        [treatment, season],
        [outcome],
    )


def test_rank_causes_without_confounder_matches_conditioning():
    causal_circuit = _build_confounded_trial_circuit()
    match = a(Trial)(treatment=cause, season=..., outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Trial: causal_circuit})
    )

    [scored] = backend.rank_causes(match)

    # No confounder declared: this is the same confounded 0.8 conditioning would
    # give, not the causal 0.6 -- see test_rank_causes_with_confounder below.
    assert scored.effect_probability_given_region == pytest.approx(0.8, abs=0.02)


def test_rank_causes_with_confounder_recovers_causal_probability():
    causal_circuit = _build_confounded_trial_circuit()
    match = a(Trial)(treatment=cause, season=confounder, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Trial: causal_circuit})
    )

    [scored] = backend.rank_causes(match)

    assert scored.effect_probability_given_region == pytest.approx(0.6, abs=0.02)


def test_evaluate_with_confounder_samples_from_the_deconfounded_region():
    causal_circuit = _build_confounded_trial_circuit()
    match = a(Trial)(treatment=cause, season=confounder, outcome=...)
    match.causes_effect(match.variable.outcome == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Trial: causal_circuit}),
        number_of_samples=10,
    )

    results = list(match.evaluate(backend=backend))

    assert len(results) == 10
    assert all(result.treatment == Treatment.HIGH for result in results)
