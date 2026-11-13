"""
Tests for the parameter arrays of a layered circuit as one flat list: copying a circuit
with other arrays, converting it to another array backend, the queries that need numpy
parameters, and training it by gradient descent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import typing_extensions
from scipy.special import logsumexp
from typing_extensions import Any

from probabilistic_model.learning.gradient_descent import LikelihoodMaximization
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    HostOnlyQueryError,
)
from probabilistic_model.probabilistic_circuit.tensorized.has_parameter_arrays import (
    HasParameterArrays,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Layer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)

from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from .test_inner_layers_on_array_backends import (
    mixture_of_products_case,
)

# %% the parameters as one list


def parameters_shifted(
    circuit: LayeredProbabilisticCircuit, layer_type: type, shift: float
) -> list[np.ndarray]:
    """
    :param circuit: A circuit with numpy parameters.
    :param layer_type: The type of layer whose first parameter array is shifted.
    :param shift: The amount to add to that array.
    :return: The parameters of the circuit, the first array of every layer of that type
        shifted.
    """
    return [
        parameter + shift if isinstance(layer, layer_type) and index == 0 else parameter
        for layer in circuit.root.all_layers()
        for index, parameter in enumerate(layer.parameters)
    ]


def test_circuit_parameters_hold_the_parameters_of_every_layer_once(
    mixture_of_products_case,
):
    """
    The parameters of a circuit are those of its layers, a layer that several parents
    share counted once.
    """
    circuit = mixture_of_products_case.circuit(np)
    assert len(circuit.parameters) == sum(
        len(layer.parameters) for layer in circuit.layers
    )


def test_a_copy_with_other_parameters_keeps_shared_layers_shared(
    mixture_of_products_case,
):
    """
    Copying a circuit with other parameters copies every layer once, so the product
    layer both sum layers point to stays one layer.
    """
    circuit = mixture_of_products_case.circuit(np)
    copy = circuit.copy_with_parameters(circuit.parameters)
    mixtures, products = copy.root.child_layers
    assert mixtures.child_layers[0] is products
    assert products is not circuit.root.child_layers[1]


def test_a_copy_with_scaled_weights_answers_with_the_new_weights(
    mixture_of_products_case,
):
    """
    A copy whose log weights are all shifted by one constant has the same log-
    likelihood, since the weights are normalized per node, while a copy with moved
    locations does not.
    """
    case = mixture_of_products_case
    circuit = case.circuit(np)
    np.testing.assert_allclose(
        circuit.copy_with_parameters(
            parameters_shifted(circuit, SumLayer, 2.0)
        ).log_likelihood(case.events(np)),
        circuit.log_likelihood(case.events(np)),
        rtol=1e-12,
    )
    assert not np.allclose(
        circuit.copy_with_parameters(
            parameters_shifted(circuit, GaussianLayer, 1.0)
        ).log_likelihood(case.events(np)),
        circuit.log_likelihood(case.events(np)),
    )


# %% conversion between backends


def test_converting_to_jax_and_back_keeps_the_log_likelihood(mixture_of_products_case):
    """
    A circuit converted to jax answers like the numpy circuit, and converted back it has
    numpy parameters again.
    """
    case = mixture_of_products_case
    circuit = case.circuit(np)
    with jax.enable_x64(True):
        on_jax = circuit.to_namespace(jnp)
        assert all(isinstance(parameter, jax.Array) for parameter in on_jax.parameters)
        np.testing.assert_allclose(
            np.asarray(on_jax.log_likelihood(case.events(jnp))),
            circuit.log_likelihood(case.events(np)),
            rtol=1e-12,
        )
        back = on_jax.to_namespace(np)
    assert all(isinstance(parameter, np.ndarray) for parameter in back.parameters)
    np.testing.assert_allclose(
        back.log_likelihood(case.events(np)),
        circuit.log_likelihood(case.events(np)),
        rtol=1e-12,
    )


def test_a_circuit_in_single_precision_computes_in_single_precision(
    mixture_of_products_case,
):
    """
    The numeric queries of a circuit compute in the floating point type of its
    parameters.
    """
    case = mixture_of_products_case
    circuit = case.circuit(np)
    single = circuit.to_namespace(np, floating_point_type=np.float32)
    result = single.log_likelihood(case.events(np))
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result, circuit.log_likelihood(case.events(np)), rtol=1e-5
    )


# %% host only queries


@pytest.mark.parametrize(
    "query",
    [
        lambda circuit: circuit.sample(3),
        lambda circuit: circuit.support,
        lambda circuit: circuit.simplify(),
    ],
    ids=["sample", "support", "simplify"],
)
def test_a_structural_query_of_a_jax_circuit_raises(mixture_of_products_case, query):
    """
    Structural queries and sampling need numpy parameters.
    """
    with jax.enable_x64(True):
        circuit = mixture_of_products_case.circuit(jnp)
        with pytest.raises(HostOnlyQueryError):
            query(circuit)


# %% projection onto valid parameters


def test_projected_gaussian_scales_are_at_least_the_minimum_scale():
    """
    Projecting the parameters of a Gaussian layer raises every scale below the minimum
    to the minimum and leaves the locations alone.
    """
    layer = GaussianLayer(0, np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    location, scale = layer.projected_parameters(
        [np.array([0.5, -0.5]), np.array([-1.0, 0.5])]
    )
    np.testing.assert_array_equal(location, [0.5, -0.5])
    np.testing.assert_array_equal(scale, [GaussianLayer.minimum_scale, 0.5])


def test_projected_log_probabilities_are_normalized_per_node():
    """
    Projecting the log-probabilities of a dense table normalizes them per node.
    """
    table = DenseProbabilityTable(np.log(np.array([[0.2, 0.8], [0.5, 0.5]])))
    (log_probabilities,) = table.projected_parameters(
        [np.log(np.array([[1.0, 3.0], [2.0, 2.0]]))]
    )
    np.testing.assert_allclose(
        logsumexp(log_probabilities, axis=1), [0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(np.exp(log_probabilities), [[0.25, 0.75], [0.5, 0.5]])


# %% training


def test_training_a_jax_circuit_raises_the_log_likelihood_of_its_data(
    mixture_of_products_case,
):
    """
    Maximizing the likelihood of data sampled from a circuit, starting from a circuit
    with moved locations, raises the average log-likelihood of the data.
    """
    case = mixture_of_products_case
    np.random.seed(0)
    data = case.circuit(np).sample(500)
    start = case.circuit(np)
    moved = parameters_shifted(start, GaussianLayer, 1.5)
    with jax.enable_x64(True):
        circuit = start.copy_with_parameters(moved).to_namespace(jnp)
        before = float(jnp.mean(circuit.log_likelihood(jnp.asarray(data))))
        trained_root = LikelihoodMaximization(circuit.root, optax.adam(5e-2)).minimize(
            200, jnp.asarray(data)
        )
        after = float(
            jnp.mean(trained_root.log_likelihood_of_nodes(jnp.asarray(data))[:, 0])
        )
    assert after > before + 0.1


# %% typing


def test_parameter_arrays_and_gradient_descent_name_their_types():
    """
    The parameter arrays of a part are typed by their array type and the trained root by
    its layer type.
    """
    assert typing_extensions.get_args(HasParameterArrays[jax.Array]) == (jax.Array, Any)
    assert typing_extensions.get_args(HasParameterArrays[np.ndarray, np.dtype]) == (
        np.ndarray,
        np.dtype,
    )
    assert typing_extensions.get_args(LikelihoodMaximization[Layer]) == (Layer,)
