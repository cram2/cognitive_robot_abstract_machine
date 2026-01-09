import json
import math
import unittest

import jax
import jax.numpy as jnp
from random_events.interval import SimpleInterval, Bound
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.jax import simple_interval_to_open_array
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import UniformLayer


class UniformLayerTestCaste(unittest.TestCase):
    x = Continuous("x")
    p_x = UniformLayer(0, jnp.array([[0, 1], [1, 3]]))
    key = jax.random.PRNGKey(69)

    def test_log_likelihood(self):
        data = jnp.array([0.5, 1.5, 4]).reshape(-1, 1)
        ll = self.p_x.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (3, 2))
        result = [
            [0.0, -float("inf")],
            [-float("inf"), -math.log(2)],
            [-float("inf"), -float("inf")],
        ]
        self.assertTrue(jnp.allclose(ll, jnp.array(result)))

    @unittest.skip("Jax next after is inconsistent")
    def test_from_interval(self):
        ioo = SimpleInterval(0, 1)
        ioc = SimpleInterval(0, 1, right=Bound.CLOSED)
        ico = SimpleInterval(0, 1, left=Bound.CLOSED)
        icc = SimpleInterval(0, 1, left=Bound.CLOSED, right=Bound.CLOSED)

        intervals = jnp.vstack(
            [simple_interval_to_open_array(i) for i in [ioo, ioc, ico, icc]]
        )
        p_x = UniformLayer(0, intervals)

        data = jnp.array([[0.0], [1.0]]).astype(float)
        ll = jnp.exp(p_x.log_likelihood_of_nodes(data))
        result = jnp.array([[0, 0, 1, 1], [0, 1, 0, 1]]).astype(float)
        self.assertTrue(jnp.allclose(ll, result))

    def test_to_json(self):
        data = self.p_x.to_json()
        json.dumps(data)
        p_x = UniformLayer.from_json(data)

        self.assertTrue(jnp.allclose(self.p_x.interval, p_x.interval))
