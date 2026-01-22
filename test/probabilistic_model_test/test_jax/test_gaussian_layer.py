import unittest

import jax.numpy as jnp
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    GaussianLayer,
    GaussianDistribution,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    UnivariateContinuousLeaf,
    ProbabilisticCircuit as NXProbabilisticCircuit,
)


class GaussianLayerTestCase(unittest.TestCase):
    model: GaussianLayer

    @classmethod
    def setUpClass(cls):
        cls.model = GaussianLayer(
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
            GaussianDistribution(x, 0.0, 0.99), probabilistic_circuit=nx_pc
        )
        g2 = UnivariateContinuousLeaf(
            GaussianDistribution(x, 1.0, 1.0), probabilistic_circuit=nx_pc
        )
        s = SumUnit(probabilistic_circuit=nx_pc)
        s.add_subcircuit(g2, 0.5)
        s.add_subcircuit(g1, 0.5)

        jax_pc = ProbabilisticCircuit.from_nx(nx_pc)
        gaussian_layer = jax_pc.root.child_layers[0]
        self.assertIsInstance(gaussian_layer, GaussianLayer)
        gaussian_layer.validate()
        self.assertEqual(gaussian_layer.variable, 0)
        self.assertTrue(jnp.allclose(gaussian_layer.location, jnp.array([0.0, 1.0])))
        self.assertTrue(jnp.allclose(gaussian_layer.scale, jnp.array([1.0, 1.01])))


if __name__ == "__main__":
    unittest.main()
