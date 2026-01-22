import unittest

import equinox
import jax
import numpy as np
from jax.experimental.sparse import BCOO
from numpy import dtype
from random_events.interval import closed, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
import jax.numpy as jnp
from sortedcontainers import SortedSet

from probabilistic_model.probabilistic_circuit.jax import (
    embed_sparse_array_in_nan_array,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.jax.inner_layer import ProductLayer
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)

import warnings

warnings.filterwarnings("ignore")


class DiracProductTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    p1_x = DiracDeltaLayer(0, jnp.array([0.0, 1.0]), jnp.array([1, 1]))
    p2_x = DiracDeltaLayer(0, jnp.array([2.0, 3.0]), jnp.array([1, 1]))
    p_y = DiracDeltaLayer(1, jnp.array([4.0, 5.0]), jnp.array([1, 1]))
    p_z = DiracDeltaLayer(2, jnp.array([6.0]), jnp.array([1]))
    product_layer: ProductLayer

    def setUp(self):
        indices = jnp.array([[0, 0], [0, 1], [1, 0], [2, 1], [3, 0], [3, 1]])
        values = jnp.array([0, 0, 0, 0, 1, 0])
        edges = (
            BCOO((values, indices), shape=(4, 2))
            .sum_duplicates(remove_zeros=False)
            .sort_indices()
        )
        self.product_layer = ProductLayer(
            [
                self.p_z,
                self.p1_x,
                self.p2_x,
                self.p_y,
            ],
            edges,
        )

    def test_variables(self):
        self.assertTrue(
            jnp.allclose(self.product_layer.variables, jnp.array([0, 1, 2]))
        )

    def test_likelihood(self):
        data = jnp.array([[0.0, 5.0, 6.0], [2, 4, 6]])
        likelihood = self.product_layer.log_likelihood_of_nodes(data)
        self.assertTrue(likelihood[0, 0] > -jnp.inf)
        self.assertTrue(likelihood[1, 1] > -jnp.inf)
        self.assertTrue(likelihood[0, 1] == -jnp.inf)
        self.assertTrue(likelihood[1, 0] == -jnp.inf)


if __name__ == "__main__":
    unittest.main()
