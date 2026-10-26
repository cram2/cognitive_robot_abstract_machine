from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
)

# %% fixtures


@pytest.fixture
def product_layer() -> DifferentiableProductLayer:
    """
    Two products of Dirac deltas over the variables 0, 1 and 2, whose children are given
    in an order other than that of their variables.
    """
    over_z = DifferentiableDiracDeltaLayer(2, jnp.array([6.0]), jnp.array([1]))
    over_x = DifferentiableDiracDeltaLayer(0, jnp.array([0.0, 1.0]), jnp.array([1, 1]))
    over_x_elsewhere = DifferentiableDiracDeltaLayer(
        0, jnp.array([2.0, 3.0]), jnp.array([1, 1])
    )
    over_y = DifferentiableDiracDeltaLayer(1, jnp.array([4.0, 5.0]), jnp.array([1, 1]))
    indices = jnp.array([[0, 0], [0, 1], [1, 0], [2, 1], [3, 0], [3, 1]])
    values = jnp.array([0, 0, 0, 0, 1, 0])
    edges = (
        BCOO((values, indices), shape=(4, 2))
        .sum_duplicates(remove_zeros=False)
        .sort_indices()
    )
    return DifferentiableProductLayer([over_z, over_x, over_x_elsewhere, over_y], edges)


# %% product layer


def test_variables_are_those_of_every_child(product_layer):
    assert jnp.allclose(product_layer.variables, jnp.array([0, 1, 2]))


def test_log_likelihood(product_layer):
    log_likelihood = product_layer.log_likelihood_of_nodes(
        jnp.array([[0.0, 5.0, 6.0], [2, 4, 6]])
    )
    assert log_likelihood[0, 0] > -jnp.inf
    assert log_likelihood[1, 1] > -jnp.inf
    assert log_likelihood[0, 1] == -jnp.inf
    assert log_likelihood[1, 0] == -jnp.inf


def test_inner_product_reads_its_own_variables():
    """
    A product layer below another product layer reads the variables of its scope from
    the whole event, not from the part of the event the outer product selected.
    """
    y_delta = DifferentiableDiracDeltaLayer(1, jnp.array([4.0]), jnp.array([2.0]))
    z_delta = DifferentiableDiracDeltaLayer(2, jnp.array([6.0]), jnp.array([3.0]))
    single_edges = BCOO((jnp.array([0, 0]), jnp.array([[0, 0], [1, 0]])), shape=(2, 1))
    inner_product = DifferentiableProductLayer([y_delta, z_delta], single_edges)
    mixture = DifferentiableSparseSumLayer(
        [inner_product],
        [BCOO((jnp.array([0.0]), jnp.array([[0, 0]])), shape=(1, 1))],
    )
    x_delta = DifferentiableDiracDeltaLayer(0, jnp.array([1.0]), jnp.array([5.0]))
    outer_product = DifferentiableProductLayer([x_delta, mixture], single_edges)

    log_likelihood = outer_product.log_likelihood_of_nodes(jnp.array([[1.0, 4.0, 6.0]]))

    expected = jnp.log(x_delta.density_cap * y_delta.density_cap * z_delta.density_cap)[
        0
    ]
    assert log_likelihood[0, 0].item() == pytest.approx(expected.item(), abs=1e-5)
