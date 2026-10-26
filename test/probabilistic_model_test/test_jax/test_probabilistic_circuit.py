from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental.sparse import BCOO
from jax.tree_util import tree_flatten

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import (
    DifferentiableUniformLayer,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    leaf,
)

from ..circuit_building import (
    correlated_gaussian_samples,
    fit_joint_probability_tree,
    mixture_of_products_of_mixtures,
)

# %% a small circuit of Dirac deltas


@pytest.fixture
def small_circuit(x, y) -> ProbabilisticCircuit:
    """
    A mixture of two products, each of a mixture over *x* and a mixture over *y*, that
    share their Dirac delta leaves.
    """
    circuit = ProbabilisticCircuit()
    x_at_0, x_at_1, y_at_2, y_at_3 = (
        leaf(
            DiracDeltaDistribution(
                variable=variable, location=location, density_cap=cap
            ),
            circuit,
        )
        for variable, location, cap in [(x, 0, 1), (x, 1, 2), (y, 2, 3), (y, 3, 4)]
    )
    return mixture_of_products_of_mixtures(
        circuit,
        (x_at_0, x_at_1),
        (y_at_2, y_at_3),
        [((0.8, 0.2), (0.5, 0.5)), ((0.7, 0.3), (0.1, 0.9))],
    )


@pytest.fixture
def small_layered_circuit(small_circuit) -> DifferentiableLayeredCircuit:
    """
    :func:`small_circuit` converted to layers.
    """
    return CircuitRepresentations().convert(small_circuit, DifferentiableLayeredCircuit)


def test_conversion_merges_units_of_a_depth_into_layers(
    small_circuit, small_layered_circuit
):
    assert small_layered_circuit.variables == small_circuit.variables
    root = small_layered_circuit.root
    assert isinstance(root, DifferentiableSparseSumLayer)
    assert root.number_of_nodes == 1
    [product_layer] = root.child_layers
    assert product_layer.number_of_nodes == 2
    assert [layer.number_of_nodes for layer in product_layer.child_layers] == [2, 2]


def test_conversion_keeps_the_log_likelihood(small_circuit, small_layered_circuit):
    samples = small_circuit.sample(1000)
    assert jnp.allclose(
        small_circuit.log_likelihood(samples),
        small_layered_circuit.log_likelihood(samples),
    )


def test_trainable_parameters_are_the_weights(small_layered_circuit):
    parameters, _ = eqx.partition(small_layered_circuit.root, eqx.is_inexact_array)
    flattened_parameters, _ = jax.tree_util.tree_flatten(parameters)
    assert sum(len(parameter) for parameter in flattened_parameters) == 10


# %% a joint probability tree


@pytest.fixture
def joint_probability_tree() -> ProbabilisticCircuit:
    """
    A joint probability tree fitted to samples of a correlated two dimensional Gaussian.
    """
    return fit_joint_probability_tree(
        correlated_gaussian_samples(2), min_samples_per_quantile=100
    )


def test_joint_probability_tree_as_layers_supports_its_samples(
    joint_probability_tree,
):
    layered = CircuitRepresentations().convert(
        joint_probability_tree, DifferentiableLayeredCircuit
    )
    samples = jnp.array(joint_probability_tree.sample(1000))
    assert (layered.log_likelihood(samples) > -jnp.inf).all()


def test_joint_probability_tree_survives_a_round_trip_through_layers(
    joint_probability_tree,
):
    layered = CircuitRepresentations().convert(
        joint_probability_tree, DifferentiableLayeredCircuit
    )
    round_tripped = CircuitRepresentations().convert(layered, ProbabilisticCircuit)
    samples = jnp.array(round_tripped.sample(1000))
    assert (layered.log_likelihood(samples) > -jnp.inf).all()


# %% learning


def test_gradient_descent_learns_the_mixture_weights():
    data = np.vstack(
        (np.random.uniform(0, 1, (100, 1)), np.random.uniform(2, 3, (200, 1)))
    )
    uniform_layer = DifferentiableUniformLayer(
        0, jnp.array([[-0.01, 1.01], [1.99, 3.01]])
    )
    model = DifferentiableSparseSumLayer(
        [uniform_layer],
        [BCOO((jnp.array([0.0, 0.0]), jnp.array([[0, 0], [0, 1]])), shape=(1, 2))],
    )
    model.validate()

    @eqx.filter_jit
    def loss(model, data) -> jax.Array:
        """
        :return: The mean negative log-likelihood of the data under the model.
        """
        return -jnp.mean(model.log_likelihood_of_nodes(data))

    optimizer = optax.adamw(0.01)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    for _ in range(100):
        _, gradients = eqx.filter_value_and_grad(loss)(model, data)
        gradients_of_weights = eqx.filter(
            tree_flatten(gradients), eqx.is_inexact_array
        )[0][0]
        assert jnp.all(jnp.isfinite(gradients_of_weights))
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)

    weights = jnp.exp(model.log_weights[0].data)
    weights /= jnp.sum(weights)
    assert weights[0] == pytest.approx(1 / 3, abs=0.01)
    assert weights[1] == pytest.approx(2 / 3, abs=0.01)
