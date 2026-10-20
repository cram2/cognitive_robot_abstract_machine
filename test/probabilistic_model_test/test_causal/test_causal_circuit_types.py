"""
Tests that causal inference gives the same answers in both circuit types.

Every circuit of :mod:`test_causal_circuit` is read both as a graph of units and as a
stack of layers, and the two are asked the same questions: whether they are support-
deterministic, what probability they give the interventional distribution, and which
cause they blame for a failure.
"""

from __future__ import annotations

import functools
import operator
import random
from dataclasses import dataclass, field

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Symbolic, Variable
from typing_extensions import Dict, List

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    OverlappingComponentsViolation,
    SupportDeterminismVerificationResult,
    SupportDeterminismViolation,
)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    ProbabilisticCircuitRepresentation,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from .test_causal_circuit import (
    _build_cause_specific_effect_circuit,
    _build_circuit_with_a_rounded_cause_region,
    _build_confounded_circuit,
    _build_correlated_circuit,
    _build_discrete_confounded_circuit,
    _build_independent_circuit,
    _build_nested_overlap_circuit,
    _build_shared_cause_subcircuit_circuit,
    _build_three_variable_circuit,
    _build_unnormalized_circuit,
)

# %% the causal queries both circuit types answer


@dataclass
class CausalQuery:
    """
    A circuit together with the roles its variables play in a causal query over it.
    """

    circuit: ProbabilisticCircuit
    """
    The circuit, built as a graph of units.
    """

    causes: List[Variable]
    """
    The cause Variables to register.
    """

    effect: Variable
    """
    The effect Variable to register.
    """

    adjustment: List[Variable] = field(default_factory=list)
    """
    The Variables to adjust for.
    """

    observations: Dict[Variable, float] = field(default_factory=dict)
    """
    An observed value per cause Variable, for the failure diagnosis.

    Empty for a query whose causes are not numeric, which
    :func:`CausalCircuit.diagnose_failure` skips.
    """


def independent() -> CausalQuery:
    circuit, x, y = _build_independent_circuit()
    return CausalQuery(circuit, [x], y, observations={x: 0.5})


def three_variables_adjusted() -> CausalQuery:
    circuit, x, y, z = _build_three_variable_circuit()
    return CausalQuery(circuit, [x], z, adjustment=[y], observations={x: 1.5})


def correlated() -> CausalQuery:
    circuit, x, w, y = _build_correlated_circuit()
    return CausalQuery(circuit, [x, w], y, observations={x: 0.5, w: 1.5})


def confounded() -> CausalQuery:
    circuit, x, y, z = _build_confounded_circuit()
    return CausalQuery(circuit, [x], y, adjustment=[z], observations={x: 0.5})


def shared_cause_subcircuit() -> CausalQuery:
    circuit, x, y = _build_shared_cause_subcircuit_circuit()
    return CausalQuery(circuit, [x], y, observations={x: 0.5})


def rounded_cause_region() -> CausalQuery:
    circuit, x, y, z = _build_circuit_with_a_rounded_cause_region()
    return CausalQuery(circuit, [x], y, adjustment=[z], observations={x: 1.0})


def discrete_confounded() -> CausalQuery:
    circuit, treatment, season, outcome = _build_discrete_confounded_circuit()
    return CausalQuery(circuit, [treatment], outcome, adjustment=[season])


def cause_specific_effect() -> CausalQuery:
    circuit, treatment, season, outcome = _build_cause_specific_effect_circuit()
    return CausalQuery(circuit, [treatment], outcome, adjustment=[season])


SUPPORT_DETERMINISTIC_QUERIES = [
    independent,
    three_variables_adjusted,
    correlated,
    confounded,
    shared_cause_subcircuit,
    rounded_cause_region,
    discrete_confounded,
    cause_specific_effect,
]
"""
The queries whose circuits pass the support-determinism check.
"""


def nested_overlap() -> CausalQuery:
    circuit, x, y = _build_nested_overlap_circuit()
    return CausalQuery(circuit, [x], y)


def unnormalized() -> CausalQuery:
    circuit, x, y = _build_unnormalized_circuit()
    return CausalQuery(circuit, [x], y)


VIOLATING_QUERIES = [nested_overlap, unnormalized]
"""
The queries whose circuits fail the support-determinism check.
"""


@pytest.fixture(params=SUPPORT_DETERMINISTIC_QUERIES, ids=lambda build: build.__name__)
def query(request) -> CausalQuery:
    """
    Every causal query over a support-deterministic circuit.
    """
    return request.param()


@pytest.fixture(params=VIOLATING_QUERIES, ids=lambda build: build.__name__)
def violating_query(request) -> CausalQuery:
    """
    Every causal query over a circuit that is not support-deterministic.
    """
    return request.param()


# %% the same query, read as either circuit type


def causal_circuit_of(
    query: CausalQuery, circuit: ProbabilisticCircuitRepresentation
) -> CausalCircuit:
    """
    :param query: The query whose roles the circuit is registered with.
    :param circuit: The circuit to register, of either type.
    :return: The causal circuit over it, unverified.
    """
    return CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph(query.causes, [query.effect]),
        query.causes,
        [query.effect],
    )


def as_units(query: CausalQuery) -> CausalCircuit[ProbabilisticCircuit]:
    """
    :param query: The query to read.
    :return: The causal circuit over the graph of units.
    """
    return causal_circuit_of(query, query.circuit)


def as_layers(query: CausalQuery) -> CausalCircuit[LayeredProbabilisticCircuit]:
    """
    :param query: The query to read.
    :return: The causal circuit over the same circuit, converted into layers.
    """
    return causal_circuit_of(
        query, RustworkxCircuitToLayeredCircuitConverter.convert(query.circuit)
    )


def events_over(variables: List[Variable], number_of_events: int) -> List[Event]:
    """
    Events that restrict every variable to a random part of its domain: a sub-interval
    of a continuous one, a random subset of the values of a symbolic one.

    :param variables: The variables to restrict.
    :param number_of_events: How many events to make.
    :return: The events.
    """
    generator = random.Random(0)
    events = []
    for _ in range(number_of_events):
        assignment = {}
        for variable in variables:
            if isinstance(variable, Symbolic):
                elements = list(variable.domain.simple_sets)
                chosen = generator.sample(elements, generator.randint(1, len(elements)))
                assignment[variable] = functools.reduce(
                    operator.or_, [element.as_composite_set() for element in chosen]
                )
            else:
                lower = generator.uniform(-1.0, 3.0)
                assignment[variable] = closed(
                    lower, lower + generator.uniform(0.0, 2.0)
                )
        events.append(SimpleEvent.from_data(assignment).as_composite_set())
    return events


def probabilities_of(circuit: ProbabilisticCircuitRepresentation, events: List[Event]) -> List[float]:
    """
    :param circuit: The circuit to query.
    :param events: The events to query it with.
    :return: The probability of every event under the circuit.
    """
    return [
        float(
            circuit.probability(
                event.__deepcopy__().fill_missing_variables_pure(circuit.variables)
            )
        )
        for event in events
    ]


# %% the support-determinism check


def test_both_circuit_types_pass_the_same_circuit(query):
    assert as_layers(query).verify_support_determinism().passed is True
    assert as_units(query).verify_support_determinism().passed is True


def described(violation: SupportDeterminismViolation) -> str:
    """
    :param violation: A violation of support determinism.
    :return: What it is, and which query variable it is about where it names one.
    """
    match violation:
        case OverlappingComponentsViolation():
            return f"{type(violation).__name__} on {violation.query_variable.name}"
        case _:
            return type(violation).__name__


def violations_of(causal_circuit: CausalCircuit) -> List[str]:
    """
    :param causal_circuit: A causal circuit whose check is expected to fail.
    :return: Every violation as both circuit types describe it, unlike the node a
        violation names, which is of the type the circuit is built from.
    """
    with pytest.raises(SupportDeterminismVerificationResult) as raised:
        causal_circuit.verify_support_determinism()
    return sorted(described(violation) for violation in raised.value.violations)


def test_both_circuit_types_report_the_same_violations(violating_query):
    assert violations_of(as_layers(violating_query)) == violations_of(
        as_units(violating_query)
    )


# %% the interventional distribution


def test_the_interventional_circuit_has_the_type_it_was_read_from(query):
    interventional = as_layers(query).backdoor_adjustment(
        query.causes[0], query.effect, query.adjustment
    )
    assert isinstance(interventional, LayeredProbabilisticCircuit)


def test_both_circuit_types_give_the_interventional_distribution_the_same_probability(
    query,
):
    cause = query.causes[0]
    events = events_over([cause, query.effect], 20)
    assert probabilities_of(
        as_layers(query).backdoor_adjustment(cause, query.effect, query.adjustment),
        events,
    ) == pytest.approx(
        probabilities_of(
            as_units(query).backdoor_adjustment(cause, query.effect, query.adjustment),
            events,
        )
    )


def test_both_circuit_types_weigh_the_cause_regions_the_same(query):
    cause = query.causes[0]
    units = as_units(query)
    layers = as_layers(query)
    interventional = layers.backdoor_adjustment(cause, query.effect, query.adjustment)
    assert probabilities_of(
        interventional,
        [
            region.event
            for region in units._extract_disjoint_regions_for_variable(cause)
        ],
    ) == pytest.approx(
        probabilities_of(
            units.backdoor_adjustment(cause, query.effect, query.adjustment),
            [
                region.event
                for region in units._extract_disjoint_regions_for_variable(cause)
            ],
        )
    )


def best_region_probability(causal_circuit: CausalCircuit, query: CausalQuery) -> float:
    """
    :param causal_circuit: The causal circuit to search.
    :param query: The query whose cause region is searched for.
    :return: The interventional probability of the best cause region. Regions of equal
        probability are what the search picks between arbitrarily, so this is the part
        of its answer that is defined.
    """
    cause = query.causes[0]
    interventional = causal_circuit.backdoor_adjustment(
        cause, query.effect, query.adjustment
    )
    [probability] = probabilities_of(
        interventional,
        [causal_circuit._best_disjoint_region(cause, interventional)],
    )
    return probability


def test_both_circuit_types_find_an_equally_good_best_region(query):
    assert best_region_probability(as_layers(query), query) == pytest.approx(
        best_region_probability(as_units(query), query)
    )


# %% the failure diagnosis


DIAGNOSABLE_QUERIES = [
    independent,
    three_variables_adjusted,
    correlated,
    confounded,
    shared_cause_subcircuit,
    rounded_cause_region,
]
"""
The queries whose causes are numeric, so that a failure can be diagnosed over them.
"""


@pytest.fixture(params=DIAGNOSABLE_QUERIES, ids=lambda build: build.__name__)
def diagnosable_query(request) -> CausalQuery:
    """
    Every causal query that has an observed value per numeric cause to diagnose.
    """
    return request.param()


@dataclass
class DiagnosedBothWays:
    """
    The same failure diagnosed over the graph of units and over the layers.
    """

    over_units: FailureDiagnosisResult
    """
    What the graph of units made of it.
    """

    over_layers: FailureDiagnosisResult
    """
    What the layers made of it.
    """

    @classmethod
    def of(cls, query: CausalQuery) -> DiagnosedBothWays:
        """
        :param query: The query to diagnose the failure of.
        :return: Both diagnoses.
        """
        return cls(
            over_units=diagnosis_of(as_units(query), query),
            over_layers=diagnosis_of(as_layers(query), query),
        )


def diagnosis_of(
    causal_circuit: CausalCircuit, query: CausalQuery
) -> FailureDiagnosisResult:
    """
    :param causal_circuit: The causal circuit to ask.
    :param query: The query whose observed values are to be diagnosed.
    :return: What it blames the failure on.
    """
    return causal_circuit.diagnose_failure(
        query.observations, query.effect, adjustment_variables=query.adjustment
    )


@pytest.fixture
def diagnoses(diagnosable_query) -> DiagnosedBothWays:
    """
    The same failure diagnosed in both circuit types.
    """
    return DiagnosedBothWays.of(diagnosable_query)


def test_both_circuit_types_blame_the_same_cause(diagnoses):
    assert (
        diagnoses.over_layers.primary_cause_variable
        == diagnoses.over_units.primary_cause_variable
    )


def test_both_circuit_types_diagnose_the_same_probabilities(diagnoses):
    np.testing.assert_allclose(
        [
            diagnoses.over_layers.interventional_probability_at_failure,
            diagnoses.over_layers.interventional_probability_at_recommendation,
        ],
        [
            diagnoses.over_units.interventional_probability_at_failure,
            diagnoses.over_units.interventional_probability_at_recommendation,
        ],
        atol=1e-9,
    )
