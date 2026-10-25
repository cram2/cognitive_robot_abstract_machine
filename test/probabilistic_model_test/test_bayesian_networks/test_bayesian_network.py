from __future__ import annotations

import math
from enum import IntEnum

import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.bayesian_network.bayesian_network import (
    BayesianNetwork,
    ConditionalProbabilisticCircuit,
    ConditionalProbabilityTable,
    Root,
)
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)
from probabilistic_model.utils import MissingDict

from ..circuit_building import uniform_leaf


class TwoStates(IntEnum):
    ZERO = 0
    ONE = 1


class ThreeStates(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2


def distribution_over(
    variable: Symbolic, probabilities: list[float]
) -> SymbolicDistribution:
    """
    :param variable: A variable over :class:`TwoStates` or :class:`ThreeStates`.
    :param probabilities: The probability of every state, in the order of the states.
    :return: The distribution.
    """
    return SymbolicDistribution(
        variable=variable,
        probabilities=MissingDict(float, zip(range(len(probabilities)), probabilities)),
    )


def conditional_probability_table(
    network: BayesianNetwork,
    variable: Symbolic,
    probabilities_per_parent_state: list[list[float]],
) -> ConditionalProbabilityTable:
    """
    :param network: The network to add the table to.
    :param variable: The variable the table is over.
    :param probabilities_per_parent_state: The probabilities of the variable for every
        state of its parent.
    :return: The table.
    """
    table = ConditionalProbabilityTable(bayesian_network=network)
    for parent_state, probabilities in enumerate(probabilities_per_parent_state):
        table.conditional_probability_distributions[parent_state] = distribution_over(
            variable, probabilities
        )
    return table


@pytest.fixture
def three_state_variable() -> Symbolic:
    """
    A variable over :class:`ThreeStates`.
    """
    return Symbolic(name="x", domain=Set.from_iterable(ThreeStates))


@pytest.fixture
def two_state_variable() -> Symbolic:
    """
    A variable over :class:`TwoStates`.
    """
    return Symbolic(name="y", domain=Set.from_iterable(TwoStates))


# %% a network of two variables


@pytest.fixture
def minimal_network(three_state_variable, two_state_variable) -> BayesianNetwork:
    """
    A network in which *x* is the parent of *y*.
    """
    network = BayesianNetwork()
    root = Root(
        distribution_over(three_state_variable, [0.5, 0.3, 0.2]),
        bayesian_network=network,
    )
    table = conditional_probability_table(
        network, two_state_variable, [[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
    )
    network.add_edge(root, table)
    return network


def test_structure(minimal_network):
    assert len(minimal_network.nodes()) == 2
    assert len(minimal_network.edges()) == 1
    [root, table] = minimal_network.nodes()
    assert table.parent is root


def test_circuit_of_a_network_multiplies_along_the_edges(
    minimal_network, three_state_variable, two_state_variable
):
    circuit = minimal_network.as_probabilistic_circuit()
    event = SimpleEvent.from_data(
        {three_state_variable: ThreeStates.ONE, two_state_variable: TwoStates.ONE}
    ).as_composite_set()
    assert circuit.probability(event) == pytest.approx(0.3 * 0.7)


# %% a network of four variables


def test_circuit_of_a_tree_shaped_network_is_smaller_than_the_joint_table(
    three_state_variable, two_state_variable
):
    z = Symbolic(name="z", domain=Set.from_iterable(ThreeStates))
    a = Symbolic(name="a", domain=Set.from_iterable(TwoStates))
    network = BayesianNetwork()
    root = Root(
        distribution_over(three_state_variable, [0.5, 0.3, 0.2]),
        bayesian_network=network,
    )
    y_given_x = conditional_probability_table(
        network, two_state_variable, [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
    )
    z_given_x = conditional_probability_table(
        network, z, [[0.1, 0.9], [1, 0], [0.8, 0.2]]
    )
    a_given_z = conditional_probability_table(network, a, [[0.3, 0.7], [0.6, 0.4]])
    network.add_edge(root, y_given_x)
    network.add_edge(root, z_given_x)
    network.add_edge(z_given_x, a_given_z)

    circuit = network.as_probabilistic_circuit().simplify()
    assert circuit.variables == SortedSet(
        [three_state_variable, two_state_variable, z, a]
    )
    assert len(circuit.edges()) < math.prod(
        len(variable.domain.simple_sets) for variable in circuit.variables
    )
    assert circuit.is_valid()


# %% a network with a conditional circuit


def test_circuit_of_a_network_with_a_conditional_circuit(y, z):
    parent = Symbolic(name="x", domain=Set.from_iterable(TwoStates))
    network = BayesianNetwork()
    root = Root(distribution_over(parent, [0.7, 0.3]), bayesian_network=network)
    y_and_z_given_parent = ConditionalProbabilisticCircuit(bayesian_network=network)
    for parent_state, (y_upper, z_upper) in zip(TwoStates, [(1, 1), (2, 3)]):
        circuit = ProbabilisticCircuit()
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(uniform_leaf(y, 0, y_upper, circuit))
        product.add_subcircuit(uniform_leaf(z, 0, z_upper, circuit))
        y_and_z_given_parent.conditional_probability_distributions[parent_state] = (
            circuit
        )
    network.add_nodes_from([root, y_and_z_given_parent])
    network.add_edge(root, y_and_z_given_parent)

    circuit = network.as_probabilistic_circuit()
    assert circuit.probability(
        circuit.universal_simple_event().as_composite_set()
    ) == pytest.approx(1.0)
    event = SimpleEvent.from_data(
        {parent: (TwoStates.ZERO, TwoStates.ONE), y: closed(1.5, 2)}
    )
    assert circuit.probability(event.as_composite_set()) == pytest.approx(0.3 * 0.25)
