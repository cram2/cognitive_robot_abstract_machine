from __future__ import annotations

import math

import jax.numpy as jnp

from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
)

# %% Dirac delta layer


def test_dirac_delta_layer_log_likelihood():
    layer = DifferentiableDiracDeltaLayer(
        0, location=jnp.array([0.0, 1.0]), density_cap=jnp.array([1.0, 2.0])
    )
    log_likelihood = layer.log_likelihood_of_nodes(jnp.array([[0], [1], [2]]))
    expected = [[0, -jnp.inf], [-jnp.inf, math.log(2)], [-jnp.inf, -jnp.inf]]
    assert log_likelihood.shape == (3, layer.number_of_nodes)
    assert jnp.allclose(log_likelihood, jnp.array(expected))
