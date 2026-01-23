import math
import unittest

import jax.numpy as jnp
import jax
from random_events.interval import closed

from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer


class DiracDeltaLayerTestCase(unittest.TestCase):
    layer = DiracDeltaLayer(
        0, location=jnp.array([0.0, 1.0]), density_cap=jnp.array([1.0, 2.0])
    )

    def test_likelihood(self):
        data = jnp.array([0, 1, 2]).reshape(-1, 1)
        ll = self.layer.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (len(data), self.layer.number_of_nodes))
        result = [[0, -jnp.inf], [-jnp.inf, math.log(2)], [-jnp.inf, -jnp.inf]]
        assert jnp.allclose(ll, jnp.array(result))


if __name__ == "__main__":
    unittest.main()
