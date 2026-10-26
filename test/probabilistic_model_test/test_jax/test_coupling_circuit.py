from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental.sparse import BCOO
from jax.tree_util import tree_map

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.probabilistic_circuit.jax.coupling_circuit import (
    Conditioner,
    CouplingCircuit,
    LinearConditioner,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import (
    DifferentiableUniformLayer,
)

from ..circuit_building import correlated_gaussian_samples, fit_joint_probability_tree


class ConstantConditioner(Conditioner):
    """
    A conditioner that weighs two components 0.2 and 0.8, whatever it conditions on.
    """

    def generate_parameters(self, x: jax.Array) -> jax.Array:
        return jnp.log(jnp.array([0.2, 0.8]).repeat(x.shape[0], 0))

    @property
    def output_length(self) -> int:
        return 2


class LinearLayerConditioner(Conditioner):
    """
    A conditioner whose log weights are a linear function of what it conditions on.
    """

    linear = eqx.nn.Linear(1, 2, key=jax.random.PRNGKey(0))

    def generate_parameters(self, x: jax.Array) -> jax.Array:
        return self.linear(x)

    @property
    def output_length(self) -> int:
        return self.linear.out_features


# %% a coupling circuit of one variable


@pytest.fixture
def data() -> jax.Array:
    """
    100 samples in [0, 1] followed by 200 samples in [2, 3].
    """
    return jnp.array(
        np.vstack(
            (np.random.uniform(0, 1, (100, 1)), np.random.uniform(2, 3, (200, 1)))
        )
    )


@pytest.fixture
def mixture_of_uniforms() -> DifferentiableSparseSumLayer:
    """
    An equally weighted mixture of uniform distributions slightly wider than [0, 1] and
    [2, 3].
    """
    uniform_layer = DifferentiableUniformLayer(
        0, jnp.array([[-0.01, 1.01], [1.99, 3.01]])
    )
    return DifferentiableSparseSumLayer(
        [uniform_layer],
        [BCOO((jnp.array([0.0, 0.0]), jnp.array([[0, 0], [0, 1]])), shape=(1, 2))],
    )


def couple(
    conditioner: Conditioner, circuit: DifferentiableSparseSumLayer
) -> CouplingCircuit:
    """
    :param conditioner: The conditioner of the coupling circuit.
    :param circuit: The circuit the conditioner parameterizes.
    :return: A coupling circuit that conditions the first variable on itself.
    """
    return CouplingCircuit(conditioner, jnp.array([0]), circuit, jnp.array([0]))


def test_conditional_log_likelihood_uses_the_conditioner_weights(
    data, mixture_of_uniforms
):
    coupling_circuit = couple(ConstantConditioner(), mixture_of_uniforms)
    coupling_circuit.validate()
    log_likelihood = coupling_circuit.conditional_log_likelihood(data)
    assert log_likelihood[0, 0] == pytest.approx(np.log(0.2 / 1.02), abs=1e-3)
    assert log_likelihood[-1, 0] == pytest.approx(np.log(0.8 / 1.02), abs=1e-3)


def test_conditional_log_likelihood_follows_the_conditioner_per_sample(
    data, mixture_of_uniforms
):
    coupling_circuit = couple(LinearLayerConditioner(), mixture_of_uniforms)
    log_likelihood = coupling_circuit.conditional_log_likelihood(data)
    first_weights = jax.nn.softmax(
        coupling_circuit.conditioner.generate_parameters(data[0])
    )
    second_weights = jax.nn.softmax(
        coupling_circuit.conditioner.generate_parameters(data[1])
    )
    assert (first_weights[0] > second_weights[0]) == (
        log_likelihood[0] > log_likelihood[1]
    )


def test_trainable_parameters_are_the_weights(mixture_of_uniforms):
    coupling_circuit = couple(ConstantConditioner(), mixture_of_uniforms)
    assert coupling_circuit.circuit.number_of_trainable_parameters == 2


def test_circuit_from_parameters_carries_them(mixture_of_uniforms):
    coupling_circuit = couple(LinearConditioner(1, 2), mixture_of_uniforms)
    parameters = coupling_circuit.conditioner.generate_parameters(jnp.array([0.1]))
    circuit = coupling_circuit.create_circuit_from_parameters(parameters)
    assert jnp.allclose(circuit.log_weights[0].data, parameters)


# %% a coupling circuit of four variables


def test_training_lowers_the_negative_log_likelihood():
    number_of_variables = 4
    half = number_of_variables // 2
    dataframe = correlated_gaussian_samples(number_of_variables)
    tree = fit_joint_probability_tree(dataframe, min_samples_per_quantile=30)
    marginal = tree.marginal(tree.variables[half:])
    layered = CircuitRepresentations().convert(marginal, DifferentiableLayeredCircuit)
    model = CouplingCircuit(
        LinearConditioner(half, layered.root.number_of_trainable_parameters),
        jnp.arange(half),
        layered.root,
        jnp.arange(half, number_of_variables),
    )
    model.validate()
    data = jnp.array(dataframe)

    @eqx.filter_jit
    def loss(model, data) -> jax.Array:
        """
        :return: The mean negative conditional log-likelihood of the data under the
            model.
        """
        return -jnp.mean(model.conditional_log_likelihood(data))

    initial_loss = loss(model, data)
    optimizer = optax.adamw(0.01)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    for _ in range(10):
        loss_value, gradients = eqx.filter_value_and_grad(loss)(model, data)
        gradients = tree_map(jnp.nan_to_num, gradients)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
        model = eqx.apply_updates(model, updates)
    assert loss_value < initial_loss
