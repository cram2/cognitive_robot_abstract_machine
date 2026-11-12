"""
A circuit converts into any representation that a chain of converters of whole circuits
reaches.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import (
    CircuitRepresentations,
)
from probabilistic_model.adapters.exceptions import NoConversionPathError
from probabilistic_model.adapters.jax_tensorized.jax_to_tensorized import (
    DifferentiableLayeredCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.jax_tensorized.tensorized_to_jax import (
    LayeredCircuitToDifferentiableLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

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


def test_rustworkx_reaches_differentiable_through_layered(representations):
    assert representations.converters_between(
        RustworkxProbabilisticCircuit, DifferentiableLayeredCircuit
    ) == [
        RustworkxCircuitToLayeredCircuitConverter,
        LayeredCircuitToDifferentiableLayeredCircuitConverter,
    ]


def test_differentiable_reaches_rustworkx_through_layered(representations):
    assert representations.converters_between(
        DifferentiableLayeredCircuit, RustworkxProbabilisticCircuit
    ) == [
        DifferentiableLayeredCircuitToLayeredCircuitConverter,
        LayeredCircuitToRustworkxCircuitConverter,
    ]


def test_subclass_of_a_representation_converts_like_it(
    representations, classification_circuit
):
    assert representations.converters_between(
        type(classification_circuit), LayeredProbabilisticCircuit
    ) == [DifferentiableLayeredCircuitToLayeredCircuitConverter]


def test_unreachable_representation_is_refused(
    representations, rustworkx_mixture_of_products
):
    with pytest.raises(NoConversionPathError):
        representations.convert(rustworkx_mixture_of_products, SortedSet)


# %% conversions


def test_rustworkx_circuit_converts_into_the_same_differentiable_distribution(
    representations, rustworkx_mixture_of_products, single_precision_tolerance
):
    differentiable = representations.convert(
        rustworkx_mixture_of_products, DifferentiableLayeredCircuit
    )
    assert isinstance(differentiable, DifferentiableLayeredCircuit)
    events = np.random.normal(size=(20, 2))
    np.testing.assert_allclose(
        np.asarray(differentiable.log_likelihood(jnp.asarray(events))),
        rustworkx_mixture_of_products.log_likelihood(events),
        rtol=single_precision_tolerance,
        atol=single_precision_tolerance,
    )


def test_differentiable_circuit_converts_into_the_same_rustworkx_distribution(
    representations, continuous_region_graph_circuit, single_precision_tolerance
):
    rustworkx = representations.convert(
        continuous_region_graph_circuit, RustworkxProbabilisticCircuit
    )
    assert isinstance(rustworkx, RustworkxProbabilisticCircuit)
    events = np.random.normal(size=(20, 4))
    np.testing.assert_allclose(
        rustworkx.log_likelihood(events),
        np.asarray(continuous_region_graph_circuit.log_likelihood(jnp.asarray(events))),
        rtol=single_precision_tolerance,
        atol=single_precision_tolerance,
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
