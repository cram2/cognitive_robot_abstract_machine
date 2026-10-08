import unittest

import jax.numpy as jnp
from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from random_events.variable import Continuous

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    UnivariateContinuousLeaf,
    ProbabilisticCircuit as NXProbabilisticCircuit,
)


class GaussianLayerTestCase(unittest.TestCase):
    model: DifferentiableGaussianLayer

    @classmethod
    def setUpClass(cls):
        cls.model = DifferentiableGaussianLayer(
            0, jnp.array([0.0, 1.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.01])
        )
        cls.model.validate()

    def test_log_pdf(self):
        x = jnp.array([[0.0], [1.0]])

        ll = self.model.log_likelihood_of_nodes(x)

        result = jnp.array([[-0.91893853, -1.41893853], [-1.41903689, -0.92888886]])
        self.assertTrue(jnp.allclose(ll, result, atol=1e-3))

    def test_from_nx_circuit(self):
        nx_pc = NXProbabilisticCircuit()
        x = Continuous("x")
        g1 = UnivariateContinuousLeaf(
            GaussianDistribution(variable=x, location=0.0, scale=0.99),
            probabilistic_circuit=nx_pc,
        )
        g2 = UnivariateContinuousLeaf(
            GaussianDistribution(variable=x, location=1.0, scale=1.0),
            probabilistic_circuit=nx_pc,
        )
        s = SumUnit(probabilistic_circuit=nx_pc)
        s.add_subcircuit(g2, 0.5)
        s.add_subcircuit(g1, 0.5)

        jax_pc = CircuitRepresentations().convert(nx_pc, DifferentiableLayeredCircuit)
        gaussian_layer = jax_pc.root.child_layers[0]
        self.assertIsInstance(gaussian_layer, DifferentiableGaussianLayer)
        gaussian_layer.validate()
        self.assertEqual(gaussian_layer.variable, 0)
        distributions = sorted(
            [g1.distribution, g2.distribution],
            key=lambda distribution: distribution.location,
        )
        order = jnp.argsort(gaussian_layer.location)
        self.assertTrue(
            jnp.allclose(
                gaussian_layer.location[order],
                jnp.array([distribution.location for distribution in distributions]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                gaussian_layer.scale[order],
                jnp.array([distribution.scale for distribution in distributions]),
            )
        )


if __name__ == "__main__":
    unittest.main()
