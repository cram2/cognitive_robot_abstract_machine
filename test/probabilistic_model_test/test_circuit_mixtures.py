"""
Tests for what every circuit type reports about its own mixtures, run against both
circuit types: which of them are unnormalized, what their components model of a
variable, and the restriction that keeps their branches.
"""

from __future__ import annotations

import math

import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous
from sortedcontainers import SortedSet

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    ProbabilisticCircuitRepresentation,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from .circuit_building import mixture_of_intervals, uniform_leaf

# %% circuits of either type


@pytest.fixture(
    params=[ProbabilisticCircuit, LayeredProbabilisticCircuit],
    ids=lambda circuit_type: circuit_type.__name__,
)
def circuit_type(request) -> type[ProbabilisticCircuitRepresentation]:
    """
    Every circuit type that reports about its mixtures.
    """
    return request.param


def as_type(
    circuit: ProbabilisticCircuit,
    circuit_type: type[ProbabilisticCircuitRepresentation],
) -> ProbabilisticCircuitRepresentation:
    """
    :param circuit: A circuit built as a graph of units.
    :param circuit_type: The type to read it as.
    :return: The circuit, converted if the type asks for layers.
    """
    if circuit_type is ProbabilisticCircuit:
        return circuit
    return RustworkxCircuitToLayeredCircuitConverter.convert(circuit)


@pytest.fixture
def two_independent_mixtures(x, y) -> ProbabilisticCircuit:
    """
    A product of a mixture over *x* and a mixture over *y*, both normalized and both
    splitting their variable into two disjoint halves.
    """
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(
        mixture_of_intervals(circuit, x, [(0.6, 0.0, 1.0), (0.4, 1.0, 2.0)])
    )
    root.add_subcircuit(
        mixture_of_intervals(circuit, y, [(0.5, 0.0, 1.0), (0.5, 1.0, 2.0)])
    )
    return circuit


def interval_event(variable: Continuous, lower: float, upper: float) -> Event:
    """
    :param variable: The variable to restrict.
    :param lower: The lower bound.
    :param upper: The upper bound.
    :return: The event that restricts the variable to the bounds.
    """
    return SimpleEvent.from_data({variable: closed(lower, upper)}).as_composite_set()


# %% unnormalized mixtures


def test_a_normalized_circuit_reports_no_unnormalized_mixture(
    circuit_type, two_independent_mixtures
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    assert circuit.unnormalized_mixtures() == []


def test_an_unnormalized_mixture_is_reported_with_what_its_weights_sum_to(
    circuit_type, x
):
    circuit = ProbabilisticCircuit()
    mixture_of_intervals(circuit, x, [(0.9, 0.0, 1.0), (0.9, 1.0, 2.0)])
    [unnormalized] = as_type(circuit, circuit_type).unnormalized_mixtures()
    assert unnormalized.log_weight_sum == pytest.approx(math.log(1.8))


def test_a_mixture_within_the_tolerance_is_not_reported(circuit_type, x):
    circuit = ProbabilisticCircuit()
    mixture_of_intervals(circuit, x, [(0.5, 0.0, 1.0), (0.5000001, 1.0, 2.0)])
    assert as_type(circuit, circuit_type).unnormalized_mixtures() == []


def test_a_circuit_of_one_leaf_has_no_mixture_to_report(circuit_type, x):
    circuit = ProbabilisticCircuit()
    uniform_leaf(x, 0, 1, circuit)
    assert as_type(circuit, circuit_type).unnormalized_mixtures() == []


# %% mixtures over a variable


def supports_per_variable(
    circuit: ProbabilisticCircuitRepresentation, variables: SortedSet
) -> list[tuple[str, list[Event]]]:
    """
    :param circuit: The circuit to read.
    :param variables: The variables to read it through.
    :return: The name of the variable and the component supports of every mixture, so
        that two circuit types can be compared without comparing their node references.
    """
    return sorted(
        (mixture.variable.name, mixture.component_supports)
        for mixture in circuit.mixtures_over(variables)
    )


def test_every_mixture_is_reported_once_per_variable_it_models(
    circuit_type, two_independent_mixtures, x, y
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    assert [name for name, _ in supports_per_variable(circuit, SortedSet([x, y]))] == [
        x.name,
        y.name,
    ]


def test_a_mixture_reports_the_support_of_every_component(
    circuit_type, two_independent_mixtures, x
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    [(_, supports)] = supports_per_variable(circuit, SortedSet([x]))
    assert sorted(supports, key=lambda support: str(support)) == sorted(
        [interval_event(x, 0.0, 1.0), interval_event(x, 1.0, 2.0)],
        key=lambda support: str(support),
    )


def test_a_mixture_of_one_component_is_not_reported(circuit_type, x):
    circuit = ProbabilisticCircuit()
    mixture_of_intervals(circuit, x, [(1.0, 0.0, 1.0)])
    assert as_type(circuit, circuit_type).mixtures_over(SortedSet([x])) == []


def test_a_variable_the_circuit_does_not_model_reports_nothing(
    circuit_type, two_independent_mixtures
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    assert circuit.mixtures_over(SortedSet([Continuous("unmodelled")])) == []


def test_overlapping_components_are_reported_with_their_overlap(circuit_type, x):
    circuit = ProbabilisticCircuit()
    mixture_of_intervals(circuit, x, [(0.5, 0.0, 2.0), (0.5, 1.0, 3.0)])
    [(_, supports)] = supports_per_variable(
        as_type(circuit, circuit_type), SortedSet([x])
    )
    first, second = supports
    assert first.intersection_with(second).size > 0


def test_both_circuit_types_report_the_same_mixtures(two_independent_mixtures, x, y):
    variables = SortedSet([x, y])
    assert supports_per_variable(
        as_type(two_independent_mixtures, LayeredProbabilisticCircuit), variables
    ) == supports_per_variable(
        as_type(two_independent_mixtures, ProbabilisticCircuit), variables
    )


# %% restriction that keeps the branches


def test_restriction_keeps_the_named_variables(
    circuit_type, two_independent_mixtures, x
):
    restricted = as_type(two_independent_mixtures, circuit_type).restrict_to_variables(
        [x]
    )
    assert list(restricted.variables) == [x]


def test_restriction_keeps_the_branches_of_the_mixtures_it_restricts_to(
    circuit_type, two_independent_mixtures, x
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    restricted = circuit.restrict_to_variables([x])
    assert supports_per_variable(restricted, SortedSet([x])) == supports_per_variable(
        circuit, SortedSet([x])
    )


def test_restriction_keeps_the_distribution_of_the_variables_it_keeps(
    circuit_type, two_independent_mixtures, x
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    restricted = circuit.restrict_to_variables([x])
    assert restricted.probability(interval_event(x, 0.0, 1.0)) == pytest.approx(
        circuit.probability(
            interval_event(x, 0.0, 1.0).fill_missing_variables_pure(circuit.variables)
        )
    )


def test_restriction_to_a_variable_the_circuit_does_not_model_gives_nothing(
    circuit_type, two_independent_mixtures
):
    circuit = as_type(two_independent_mixtures, circuit_type)
    assert circuit.restrict_to_variables([Continuous("unmodelled")]) is None
