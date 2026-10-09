import unittest
from jax.experimental.sparse import BCOO
from random_events.variable import Continuous
import jax.numpy as jnp
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
)

import warnings

warnings.filterwarnings("ignore")


class DiracProductTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    p1_x = DifferentiableDiracDeltaLayer(0, jnp.array([0.0, 1.0]), jnp.array([1, 1]))
    p2_x = DifferentiableDiracDeltaLayer(0, jnp.array([2.0, 3.0]), jnp.array([1, 1]))
    p_y = DifferentiableDiracDeltaLayer(1, jnp.array([4.0, 5.0]), jnp.array([1, 1]))
    p_z = DifferentiableDiracDeltaLayer(2, jnp.array([6.0]), jnp.array([1]))
    product_layer: DifferentiableProductLayer

    def setUp(self):
        indices = jnp.array([[0, 0], [0, 1], [1, 0], [2, 1], [3, 0], [3, 1]])
        values = jnp.array([0, 0, 0, 0, 1, 0])
        edges = (
            BCOO((values, indices), shape=(4, 2))
            .sum_duplicates(remove_zeros=False)
            .sort_indices()
        )
        self.product_layer = DifferentiableProductLayer(
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


class NestedProductTestCase(unittest.TestCase):
    """
    A product layer below another product layer reads the variables of its scope from
    the whole event, not from the part of the event the outer product selected.
    """

    def test_inner_product_reads_its_own_variables(self):
        y_delta = DifferentiableDiracDeltaLayer(1, jnp.array([4.0]), jnp.array([2.0]))
        z_delta = DifferentiableDiracDeltaLayer(2, jnp.array([6.0]), jnp.array([3.0]))
        single_edges = BCOO(
            (jnp.array([0, 0]), jnp.array([[0, 0], [1, 0]])), shape=(2, 1)
        )
        inner_product = DifferentiableProductLayer([y_delta, z_delta], single_edges)
        mixture = DifferentiableSparseSumLayer(
            [inner_product],
            [BCOO((jnp.array([0.0]), jnp.array([[0, 0]])), shape=(1, 1))],
        )
        x_delta = DifferentiableDiracDeltaLayer(0, jnp.array([1.0]), jnp.array([5.0]))
        outer_product = DifferentiableProductLayer([x_delta, mixture], single_edges)

        likelihood = outer_product.log_likelihood_of_nodes(jnp.array([[1.0, 4.0, 6.0]]))

        self.assertAlmostEqual(
            likelihood[0, 0].item(),
            jnp.log(x_delta.density_cap * y_delta.density_cap * z_delta.density_cap)[
                0
            ].item(),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
