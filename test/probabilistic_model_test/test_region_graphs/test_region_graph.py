"""
Tests for region graphs: their random partitions, as in Algorithm 1 of the RAT-SPN
paper, and the structure, normalization and classes of the circuits they build.
"""

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
import numpy as np
import optax
import pytest
import rustworkx as rx
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.learning.region_graph.exceptions import (
    MultipleRootsError,
    UnsupportedRegionVariableError,
)
from probabilistic_model.learning.region_graph.region_graph import (
    Partition,
    Region,
    RegionGraph,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    UnivariateInputLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    UnivariateContinuousLeaf,
    UnivariateDiscreteLeaf,
)


class Target(IntEnum):
    """
    The classes a classification circuit predicts.
    """

    A = 0
    B = 1


@pytest.fixture
def target() -> Symbolic:
    """
    The variable a classification circuit predicts.
    """
    return Symbolic(name="target", domain=Set.from_iterable(Target))


def continuous_variables(number_of_variables: int, prefix: str = "") -> SortedSet:
    """
    :param number_of_variables: The number of variables.
    :param prefix: The prefix of their names, which are followed by their index.
    :return: The variables.
    """
    return SortedSet(
        [Continuous(f"{prefix}{index}") for index in range(number_of_variables)]
    )


# %% structure


def partitions_of(region_graph: RegionGraph) -> list[list[str]]:
    """
    :param region_graph: A region graph.
    :return: The names of the variables of every region, in the order they were created.
    """
    return [
        [variable.name for variable in region.variables]
        for region in region_graph.regions()
    ]


def test_region_graphs_with_the_same_seed_have_the_same_partitions():
    """
    The seed fixes the random partitions.
    """
    first, second = (
        RegionGraph(
            variables=continuous_variables(8), depth=2, repetitions=2, seed=3
        ).create_random_region_graph()
        for _ in range(2)
    )
    assert partitions_of(first) == partitions_of(second)


def test_region_graphs_with_different_seeds_have_different_partitions():
    """
    Another seed draws other partitions.
    """
    first, second = (
        RegionGraph(
            variables=continuous_variables(8), depth=2, repetitions=2, seed=seed
        ).create_random_region_graph()
        for seed in (3, 4)
    )
    assert partitions_of(first) != partitions_of(second)


def test_a_region_graph_with_two_roots_raises():
    """
    A graph of two unconnected regions has no single root.
    """
    variables = continuous_variables(2)
    region_graph = RegionGraph(variables=variables)
    for variable in variables:
        region_graph.add_node(Region(variables=SortedSet([variable])))
    with pytest.raises(MultipleRootsError):
        region_graph.root


def test_a_leaf_region_over_an_integer_variable_raises():
    """
    No input distribution of a region graph circuit models an integer variable.
    """
    region_graph = RegionGraph(
        variables=SortedSet([Integer("count")]), depth=0, repetitions=1
    ).create_random_region_graph()
    with pytest.raises(UnsupportedRegionVariableError):
        region_graph.to_probabilistic_circuit()


def number_of_regions_by_depth(region_graph: RegionGraph) -> dict[int, int]:
    """
    :param region_graph: A region graph.
    :return: The number of regions at every depth below the root region.
    """
    depths = rx.bfs_layers(
        region_graph.graph, [region_graph.index_of(region_graph.root)]
    )
    return {
        depth // 2: len(layer) for depth, layer in enumerate(depths) if depth % 2 == 0
    }


def test_every_repetition_splits_every_region_in_two_until_the_depth():
    """
    Algorithm 1 of the RAT-SPN paper: each of the repetitions splits the root region in
    two and every new region again, until the depth; the leaves are the last regions.
    """
    depth, repetitions = 3, 2
    region_graph = RegionGraph(
        variables=continuous_variables(16),
        partitions=2,
        depth=depth,
        repetitions=repetitions,
    ).create_random_region_graph()
    assert number_of_regions_by_depth(region_graph) == {
        level: 1 if level == 0 else repetitions * 2**level for level in range(depth + 1)
    }
    assert len(list(region_graph.partition_nodes())) == repetitions * (2**depth - 1)


def test_a_region_of_one_variable_is_a_leaf():
    """
    Splitting stops at a region of one variable, even above the depth.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(3), partitions=2, depth=4, repetitions=2
    ).create_random_region_graph()
    assert all(
        not region_graph.successors(region)
        for region in region_graph.regions()
        if len(region.variables) == 1
    )


def test_a_region_is_split_into_at_most_one_part_per_variable():
    """
    A region of two variables split into three parts becomes two regions, none empty.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(2), partitions=3, depth=2, repetitions=1
    ).create_random_region_graph()
    assert all(len(region.variables) > 0 for region in region_graph.regions())
    assert [
        len(region_graph.successors(partition))
        for partition in region_graph.partition_nodes()
    ] == [2]


# %% circuit structure


def number_of_sum_weights(circuit: LayeredProbabilisticCircuit) -> int:
    """
    :param circuit: A layered circuit.
    :return: The number of weights of its sum layers.
    """
    return sum(
        layer.number_of_own_parameters
        for layer in circuit.layers
        if isinstance(layer, SumLayer)
    )


def number_of_input_parameters(circuit: LayeredProbabilisticCircuit) -> int:
    """
    :param circuit: A layered circuit.
    :return: The number of parameters of its input layers.
    """
    return sum(
        layer.number_of_own_parameters
        for layer in circuit.layers
        if isinstance(layer, UnivariateInputLayer)
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_a_region_graph_circuit_has_as_many_sum_weights_as_a_rat_spn(depth):
    """
    Equation (1) of the RAT-SPN paper: every partition holds the cross products of its
    two regions and every sum node of a region is connected to all of them.
    """
    repetitions, sum_units, input_units = 2, 3, 4
    circuit = (
        RegionGraph(
            variables=continuous_variables(16),
            partitions=2,
            depth=depth,
            repetitions=repetitions,
        )
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=input_units, sum_units=sum_units)
    )
    if depth == 1:
        expected = repetitions * input_units**2
    else:
        expected = repetitions * (
            sum_units**2
            + (2 ** (depth - 1) - 2) * sum_units**3
            + 2 ** (depth - 1) * sum_units * input_units**2
        )
    assert number_of_sum_weights(circuit) == expected


def test_every_leaf_region_holds_fully_factorized_input_distributions():
    """
    Equation (2) of the RAT-SPN paper: every repetition has as many univariate input
    distributions per variable as each leaf region has input distributions.
    """
    repetitions, input_units, variables = 3, 4, continuous_variables(8)
    circuit = (
        RegionGraph(variables=variables, partitions=2, depth=1, repetitions=repetitions)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=input_units)
    )
    parameters_of_a_gaussian = 2
    assert number_of_input_parameters(circuit) == (
        repetitions * input_units * len(variables) * parameters_of_a_gaussian
    )


def test_a_region_graph_without_splits_is_a_mixture_of_fully_factorized_inputs():
    """
    Without a split, the root region mixes its fully factorized input distributions.
    """
    circuit = (
        RegionGraph(variables=continuous_variables(3), depth=0, repetitions=1)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=4)
    )
    assert number_of_sum_weights(circuit) == 4
    data = np.random.default_rng(0).normal(size=(10, 3))
    assert np.all(np.isfinite(np.asarray(circuit.log_likelihood(jnp.asarray(data)))))


def test_every_partition_of_a_deep_region_graph_over_few_variables_has_children():
    """
    A depth larger than the variables allow creates no empty partition.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(4), partitions=2, depth=2, repetitions=2
    ).create_random_region_graph()
    assert all(
        len(region_graph.successors(partition)) > 0
        for partition in region_graph.partition_nodes()
    )


def test_deep_region_graph_over_few_variables_becomes_a_circuit():
    """
    A region graph deeper than its variables allow builds a circuit with finite log-
    likelihoods.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(5), partitions=2, depth=2, repetitions=2
    ).create_random_region_graph()
    model = region_graph.to_probabilistic_circuit(input_units=3, sum_units=5)
    data = jnp.array(np.random.normal(size=(20, 5)))
    assert jnp.all(jnp.isfinite(model.log_likelihood(data)))


# %% learning


def test_learned_circuit_has_normalized_parameters(target):
    """
    After fitting, every sum unit and categorical distribution is normalized and every
    scale is positive.
    """
    variables = continuous_variables(8) | SortedSet([target])
    region_graph = RegionGraph(
        variables=variables, partitions=2, depth=2, repetitions=2
    ).create_random_region_graph()
    model = region_graph.to_probabilistic_circuit(input_units=5, sum_units=5)
    data = np.random.uniform(0, 1, (100, len(variables)))
    # the symbolic column holds the hashes of domain elements, which the circuit encodes
    target_column = model.variables.index(target)
    data[:, target_column] = SymbolicEncoding(target).hashes[
        np.random.randint(2, size=len(data))
    ]
    model.fit(data, epochs=10, optimizer=optax.adamw(0.01))

    circuit = CircuitRepresentations().convert(
        model.to_namespace(np), ProbabilisticCircuit
    )
    for node in circuit.nodes():
        if isinstance(node, SumUnit):
            assert logsumexp(node.log_weights) == pytest.approx(0.0)
        elif isinstance(node, UnivariateDiscreteLeaf):
            assert sum(node.distribution.probabilities) == pytest.approx(1.0)
        elif isinstance(node, UnivariateContinuousLeaf):
            assert node.distribution.scale > 0.0


def test_classification_circuit_predicts_a_distribution_over_the_target(target):
    """
    The marginal of a fitted classification circuit over its class variable is a
    distribution.
    """
    features = continuous_variables(4, prefix="x")
    region_graph = RegionGraph(
        variables=features, partitions=2, depth=1, repetitions=6, classes=2
    ).create_random_region_graph()
    model = region_graph.to_probabilistic_circuit(
        input_units=5, sum_units=5, class_variable=target
    )
    assert isinstance(model, LayeredProbabilisticCircuit)
    assert model.variables == features | SortedSet([target])

    data = np.random.uniform(0, 1, (100, len(model.variables)))
    target_column = model.variables.index(target)
    data[:, target_column] = SymbolicEncoding(target).hashes[
        np.random.randint(2, size=len(data))
    ]
    model.fit(data, epochs=10, optimizer=optax.adamw(0.01))

    circuit = CircuitRepresentations().convert(
        model.to_namespace(np), ProbabilisticCircuit
    )
    assert circuit.is_decomposable()
    target_marginal = circuit.marginal(SortedSet([target]))
    total = sum(
        target_marginal.probability_of_simple_event(
            SimpleEvent.from_data({target: element})
        )
        for element in target.domain
    )
    assert total == pytest.approx(1.0)
