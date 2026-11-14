"""
A circuit converts into any representation that a chain of converters of whole circuits
reaches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from random_events.variable import Continuous
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import (
    CircuitRepresentations,
)
from probabilistic_model.adapters.exceptions import NoConversionPathError
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)


@dataclass
class SubclassOfARepresentation(RustworkxProbabilisticCircuit):
    """
    A circuit type derived from a representation, as a learned circuit is.
    """


# %% fixtures


@pytest.fixture
def representations() -> CircuitRepresentations:
    """
    Every representation of circuits and the converters between them.
    """
    return CircuitRepresentations()


@pytest.fixture
def rustworkx_mixture_of_products(x, y) -> RustworkxProbabilisticCircuit:
    """
    A rustworkx circuit over *x* and *y* that mixes two products of Gaussians.
    """
    circuit = RustworkxProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, location in [(0.3, -1.0), (0.7, 2.0)]:
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in [x, y]:
            product.add_subcircuit(
                leaf(
                    GaussianDistribution(
                        variable=variable, location=location, scale=1.0
                    ),
                    circuit,
                )
            )
        root.add_subcircuit(product, np.log(weight))
    return circuit


# %% paths between representations


def test_rustworkx_reaches_layered_in_one_step(representations):
    assert representations.converters_between(
        RustworkxProbabilisticCircuit, LayeredProbabilisticCircuit
    ) == [RustworkxCircuitToLayeredCircuitConverter]


def test_layered_reaches_rustworkx_in_one_step(representations):
    assert representations.converters_between(
        LayeredProbabilisticCircuit, RustworkxProbabilisticCircuit
    ) == [LayeredCircuitToRustworkxCircuitConverter]


def test_subclass_of_a_representation_converts_like_it(representations):
    assert representations.converters_between(
        SubclassOfARepresentation, LayeredProbabilisticCircuit
    ) == [RustworkxCircuitToLayeredCircuitConverter]


def test_unreachable_representation_is_refused(
    representations, rustworkx_mixture_of_products
):
    with pytest.raises(NoConversionPathError):
        representations.convert(rustworkx_mixture_of_products, SortedSet)


# %% conversions


def test_rustworkx_circuit_converts_into_the_same_layered_distribution(
    representations, rustworkx_mixture_of_products
):
    layered = representations.convert(
        rustworkx_mixture_of_products, LayeredProbabilisticCircuit
    )
    assert isinstance(layered, LayeredProbabilisticCircuit)
    events = np.random.normal(size=(20, 2))
    np.testing.assert_allclose(
        layered.log_likelihood(events),
        rustworkx_mixture_of_products.log_likelihood(events),
        rtol=1e-12,
    )


def test_region_graph_circuit_converts_into_the_same_rustworkx_distribution(
    representations,
):
    variables = SortedSet(Continuous(name) for name in "wxyz")
    layered = (
        RegionGraph(variables=variables, partitions=2, depth=1, repetitions=2)
        .create_random_region_graph()
        .to_probabilistic_circuit()
        .to_namespace(np)
    )
    rustworkx = representations.convert(layered, RustworkxProbabilisticCircuit)
    assert isinstance(rustworkx, RustworkxProbabilisticCircuit)
    events = np.random.normal(size=(20, 4))
    np.testing.assert_allclose(
        rustworkx.log_likelihood(events), layered.log_likelihood(events), rtol=1e-9
    )


def test_circuit_in_the_requested_representation_is_returned_as_it_is(
    representations, rustworkx_mixture_of_products
):
    assert (
        representations.convert(
            rustworkx_mixture_of_products, RustworkxProbabilisticCircuit
        )
        is rustworkx_mixture_of_products
    )
