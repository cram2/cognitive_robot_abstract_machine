from __future__ import annotations

import jax.numpy as jnp

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    leaf,
)

# %% Gaussian layer


def test_log_likelihood():
    layer = DifferentiableGaussianLayer(
        0, jnp.array([0.0, 1.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.01])
    )
    layer.validate()
    log_likelihood = layer.log_likelihood_of_nodes(jnp.array([[0.0], [1.0]]))
    expected = jnp.array([[-0.91893853, -1.41893853], [-1.41903689, -0.92888886]])
    assert jnp.allclose(log_likelihood, expected, atol=1e-3)


def test_conversion_from_a_mixture_of_gaussian_leaves(x):
    circuit = ProbabilisticCircuit()
    gaussians = [
        GaussianDistribution(variable=x, location=0.0, scale=0.99),
        GaussianDistribution(variable=x, location=1.0, scale=1.0),
    ]
    mixture = SumUnit(probabilistic_circuit=circuit)
    for gaussian in reversed(gaussians):
        mixture.add_subcircuit(leaf(gaussian, circuit), 0.5)

    layered = CircuitRepresentations().convert(circuit, DifferentiableLayeredCircuit)
    gaussian_layer = layered.root.child_layers[0]
    assert isinstance(gaussian_layer, DifferentiableGaussianLayer)
    gaussian_layer.validate()
    assert gaussian_layer.variable == 0
    order = jnp.argsort(gaussian_layer.location)
    assert jnp.allclose(
        gaussian_layer.location[order],
        jnp.array([gaussian.location for gaussian in gaussians]),
    )
    assert jnp.allclose(
        gaussian_layer.scale[order],
        jnp.array([gaussian.scale for gaussian in gaussians]),
    )
