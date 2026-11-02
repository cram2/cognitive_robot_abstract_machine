from __future__ import annotations

import json
import math

import jax.numpy as jnp
import pytest
from random_events.interval import Bound, SimpleInterval

from probabilistic_model.probabilistic_circuit.jax.uniform_layer import (
    DifferentiableUniformLayer,
)
from probabilistic_model.probabilistic_circuit.jax.utils import (
    simple_interval_to_open_array,
)

# %% fixtures


@pytest.fixture
def uniform_layer() -> DifferentiableUniformLayer:
    """
    Two uniform distributions over (0, 1) and (1, 3).
    """
    return DifferentiableUniformLayer(0, jnp.array([[0, 1], [1, 3]]))


# %% uniform layer


def test_log_likelihood(uniform_layer):
    log_likelihood = uniform_layer.log_likelihood_of_nodes(
        jnp.array([[0.5], [1.5], [4]])
    )
    expected = [
        [0.0, -jnp.inf],
        [-jnp.inf, -math.log(2)],
        [-jnp.inf, -jnp.inf],
    ]
    assert log_likelihood.shape == (3, 2)
    assert jnp.allclose(log_likelihood, jnp.array(expected))


@pytest.mark.skip(reason="Jax next after is inconsistent")
def test_bounds_of_the_interval_decide_whether_they_are_inside():
    intervals = jnp.vstack(
        [
            simple_interval_to_open_array(SimpleInterval.from_data(0, 1, left, right))
            for left, right in [
                (Bound.OPEN, Bound.OPEN),
                (Bound.OPEN, Bound.CLOSED),
                (Bound.CLOSED, Bound.OPEN),
                (Bound.CLOSED, Bound.CLOSED),
            ]
        ]
    )
    layer = DifferentiableUniformLayer(0, intervals)
    likelihood = jnp.exp(layer.log_likelihood_of_nodes(jnp.array([[0.0], [1.0]])))
    expected = jnp.array([[0, 0, 1, 1], [0, 1, 0, 1]]).astype(float)
    assert jnp.allclose(likelihood, expected)


def test_serialization_round_trip(uniform_layer):
    data = json.loads(json.dumps(uniform_layer.to_json()))
    deserialized = DifferentiableUniformLayer.from_json(data)
    assert jnp.allclose(deserialized.interval, uniform_layer.interval)
