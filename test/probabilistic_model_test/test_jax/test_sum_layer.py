from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from random_events.variable import Continuous
from scipy.special import logsumexp

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.learning.nyga_induction import NygaInduction
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableDenseSumLayer,
    DifferentiableSparseSumLayer,
    DifferentiableSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

# %% fixtures


@pytest.fixture
def dirac_deltas() -> list[DifferentiableDiracDeltaLayer]:
    """
    Four layers of Dirac deltas at the locations 0 to 6, whose density caps are the
    location plus one, except at 0 and 6, where they are one.
    """
    return [
        DifferentiableDiracDeltaLayer(0, jnp.array([0.0, 1.0]), jnp.array([1, 2])),
        DifferentiableDiracDeltaLayer(0, jnp.array([2.0]), jnp.array([3])),
        DifferentiableDiracDeltaLayer(
            0, jnp.array([3.0, 4.0, 5.0]), jnp.array([4, 5, 6])
        ),
        DifferentiableDiracDeltaLayer(0, jnp.array([6.0]), jnp.array([1])),
    ]


@pytest.fixture
def weights_per_child() -> list[jnp.ndarray]:
    """
    The weights of two mixtures that sum to two, split by the child layer they weigh.
    """
    return [
        jnp.array([[0, 0.1], [0.4, 0]]) * 2,
        jnp.array([[0.2], [0.3]]) * 2,
        jnp.array([[0.3, 0, 0.4], [0.0, 0.1, 0.2]]) * 2,
        jnp.array([[0], [0]]) * 2,
    ]


@pytest.fixture
def sparse_sum_layer(dirac_deltas, weights_per_child) -> DifferentiableSparseSumLayer:
    """
    The two mixtures of :func:`weights_per_child` over :func:`dirac_deltas` with sparse
    weights.
    """
    log_weights = []
    for weights in weights_per_child:
        sparse_weights = BCOO.fromdense(weights)
        sparse_weights.data = jnp.log(sparse_weights.data)
        log_weights.append(sparse_weights)
    layer = DifferentiableSparseSumLayer(dirac_deltas, log_weights=log_weights)
    layer.validate()
    return layer


@pytest.fixture
def dense_sum_layer(dirac_deltas, weights_per_child) -> DifferentiableDenseSumLayer:
    """
    The two mixtures of :func:`weights_per_child` over :func:`dirac_deltas` with dense
    weights.
    """
    layer = DifferentiableDenseSumLayer(
        dirac_deltas,
        log_weights=[jnp.log(weights) for weights in weights_per_child],
    )
    layer.validate()
    return layer


@pytest.fixture(params=["sparse_sum_layer", "dense_sum_layer"])
def sum_layer(request) -> DifferentiableSumLayer:
    """
    The two mixtures of :func:`weights_per_child`, with sparse and with dense weights.
    """
    return request.getfixturevalue(request.param)


# %% sum layers over Dirac deltas


def test_log_normalization_constants(sum_layer):
    assert jnp.allclose(sum_layer.log_normalization_constants, jnp.log(2))


def test_normalized_weights(sum_layer):
    weights = sum_layer.normalized_weights
    if isinstance(weights, BCOO):
        weights = weights.todense()
    expected = jnp.array(
        [[0, 0.1, 0.2, 0.3, 0, 0.4, 0], [0.4, 0, 0.3, 0.0, 0.1, 0.2, 0]]
    )
    assert jnp.allclose(weights, expected)


def test_log_likelihood(sum_layer):
    log_likelihood = sum_layer.log_likelihood_of_nodes(jnp.arange(7.0).reshape(-1, 1))
    expected = jnp.log(
        jnp.array(
            [
                [0.0, 0.4],
                [0.1 * 2, 0.0],
                [0.2 * 3, 0.3 * 3],
                [0.3 * 4, 0.0],
                [0.0, 0.1 * 5],
                [0.4 * 6, 0.2 * 6],
                [0.0, 0.0],
            ]
        )
    )
    assert jnp.allclose(log_likelihood, expected)


def test_log_likelihood_of_one_event(sum_layer):
    log_likelihood = sum_layer.log_likelihood_of_nodes_single(jnp.array([0]))
    assert jnp.allclose(log_likelihood, jnp.log(jnp.array([0.0, 0.4])))


def test_reset_variables_recomputes_them(sparse_sum_layer):
    sparse_sum_layer.reset_variables()
    assert sparse_sum_layer.variables.item() == 0


# %% unlikely events


@pytest.mark.parametrize(
    "layer_type", [DifferentiableSparseSumLayer, DifferentiableDenseSumLayer]
)
def test_log_likelihood_of_an_unlikely_event_stays_finite(layer_type):
    """
    A sum layer stays finite for an event whose likelihood is below the smallest
    positive single precision number, as the events of a circuit over many variables
    are.
    """
    gaussians = DifferentiableGaussianLayer(
        0, jnp.array([0.0, 1.0]), jnp.zeros(2), jnp.zeros(2)
    )
    event = jnp.array([[20.0]])
    log_weights = jnp.log(jnp.array([[0.25, 0.75]]))
    if layer_type is DifferentiableSparseSumLayer:
        layer = layer_type([gaussians], [BCOO.fromdense(log_weights)])
    else:
        layer = layer_type([gaussians], [log_weights])

    log_likelihood = np.asarray(layer.log_likelihood_of_nodes(event))
    expected = logsumexp(
        np.asarray(gaussians.log_likelihood_of_nodes(event)) + np.asarray(log_weights),
        axis=1,
    )
    assert np.isfinite(log_likelihood).all()
    np.testing.assert_allclose(log_likelihood[:, 0], expected, rtol=1e-5)


# %% Nyga distribution


@pytest.fixture(scope="module")
def nyga_data() -> jax.Array:
    """
    Samples of a standard normal distribution.
    """
    return jax.random.normal(jax.random.PRNGKey(69), (1000, 1))


@pytest.fixture(scope="module")
def nyga_layered_circuit(nyga_data) -> DifferentiableLayeredCircuit:
    """
    A Nyga distribution fitted to :func:`nyga_data`, converted to layers.
    """
    circuit = NygaInduction(Continuous("x"), min_samples_per_quantile=10).fit(nyga_data)
    layered = CircuitRepresentations().convert(circuit, DifferentiableLayeredCircuit)
    layered.root.validate()
    return layered


def test_nyga_distribution_supports_its_training_data(nyga_layered_circuit, nyga_data):
    assert jnp.all(nyga_layered_circuit.log_likelihood(nyga_data) > -jnp.inf)


def test_nyga_distribution_converts_back_to_normalized_weights(nyga_layered_circuit):
    circuit = CircuitRepresentations().convert(
        nyga_layered_circuit, ProbabilisticCircuit
    )
    assert logsumexp(circuit.root.log_weights) == pytest.approx(0.0)
