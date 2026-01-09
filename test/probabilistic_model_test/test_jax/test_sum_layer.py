import unittest

import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
import jax.numpy as jnp
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    SparseSumLayer,
    DenseSumLayer,
)
import jax

from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as NXProbabilisticCircuit,
)


class DiracSumUnitTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    p1_x = DiracDeltaLayer(0, jnp.array([0.0, 1.0]), jnp.array([1, 2]))
    p2_x = DiracDeltaLayer(0, jnp.array([2.0]), jnp.array([3]))
    p3_x = DiracDeltaLayer(0, jnp.array([3.0, 4.0, 5.0]), jnp.array([4, 5, 6]))
    p4_x = DiracDeltaLayer(0, jnp.array([6.0]), jnp.array([1]))
    sum_layer: SparseSumLayer

    @classmethod
    def setUpClass(cls):
        weights_p1 = BCOO.fromdense(jnp.array([[0, 0.1], [0.4, 0]])) * 2
        weights_p1.data = jnp.log(weights_p1.data)

        weights_p2 = BCOO.fromdense(jnp.array([[0.2], [0.3]])) * 2
        weights_p2.data = jnp.log(weights_p2.data)

        weights_p3 = BCOO.fromdense(jnp.array([[0.3, 0, 0.4], [0.0, 0.1, 0.2]])) * 2
        weights_p3.data = jnp.log(weights_p3.data)

        weights_p4 = BCOO.fromdense(jnp.array([[0], [0]])) * 2
        weights_p4.data = jnp.log(weights_p4.data)

        cls.sum_layer = SparseSumLayer(
            [cls.p1_x, cls.p2_x, cls.p3_x, cls.p4_x],
            log_weights=[weights_p1, weights_p2, weights_p3, weights_p4],
        )
        cls.sum_layer.validate()

    def test_normalization_constants(self):
        log_normalization_constants = self.sum_layer.log_normalization_constants
        result = jnp.log(jnp.array([2, 2]))
        self.assertTrue(jnp.allclose(log_normalization_constants, result))

    def test_normalized_weights(self):
        normalized_weights = self.sum_layer.normalized_weights.todense()
        result = jnp.array(
            [[0, 0.1, 0.2, 0.3, 0, 0.4, 0], [0.4, 0, 0.3, 0.0, 0.1, 0.2, 0]]
        )
        self.assertTrue(jnp.allclose(normalized_weights, result))

    def test_ll(self):
        data = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
        # l = self.sum_layer.log_likelihood_of_nodes_single(data[0])

        ll = self.sum_layer.log_likelihood_of_nodes(data)
        result = jnp.log(
            jnp.array(
                [
                    [
                        0.0,
                        0.4,
                    ],
                    [
                        0.1 * 2,
                        0.0,
                    ],
                    [
                        0.2 * 3,
                        0.3 * 3,
                    ],
                    [
                        0.3 * 4,
                        0.0,
                    ],
                    [
                        0.0,
                        0.1 * 5,
                    ],
                    [
                        0.4 * 6,
                        0.2 * 6,
                    ],
                    [
                        0.0,
                        0.0,
                    ],
                ]
            )
        )
        assert jnp.allclose(ll, result)

    def test_ll_single(self):
        data = jnp.array([0])
        l = self.sum_layer.log_likelihood_of_nodes_single(data)
        result = jnp.log(jnp.array([0.0, 0.4]))
        assert jnp.allclose(l, result)

    def test_set_variables(self):
        self.sum_layer.reset_variables()
        self.assertEqual(self.sum_layer.variables.item(), 0)


class DiracDenseSumUnitTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    p1_x = DiracDeltaLayer(0, jnp.array([0.0, 1.0]), jnp.array([1, 2]))
    p2_x = DiracDeltaLayer(0, jnp.array([2.0]), jnp.array([3]))
    p3_x = DiracDeltaLayer(0, jnp.array([3.0, 4.0, 5.0]), jnp.array([4, 5, 6]))
    p4_x = DiracDeltaLayer(0, jnp.array([6.0]), jnp.array([1]))
    sum_layer: DenseSumLayer

    @classmethod
    def setUpClass(cls):
        weights_p1 = (jnp.array([[0, 0.1], [0.4, 0]])) * 2
        weights_p1 = jnp.log(weights_p1)

        weights_p2 = jnp.array([[0.2], [0.3]]) * 2
        weights_p2 = jnp.log(weights_p2)

        weights_p3 = jnp.array([[0.3, 0, 0.4], [0.0, 0.1, 0.2]]) * 2
        weights_p3 = jnp.log(weights_p3)

        weights_p4 = jnp.array([[0], [0]]) * 2
        weights_p4 = jnp.log(weights_p4)

        cls.sum_layer = DenseSumLayer(
            [cls.p1_x, cls.p2_x, cls.p3_x, cls.p4_x],
            log_weights=[weights_p1, weights_p2, weights_p3, weights_p4],
        )
        cls.sum_layer.validate()

    def test_normalization_constants(self):
        log_normalization_constants = self.sum_layer.log_normalization_constants
        result = jnp.log(jnp.array([2, 2]))
        self.assertTrue(jnp.allclose(log_normalization_constants, result))

    def test_normalized_weights(self):
        normalized_weights = self.sum_layer.normalized_weights
        result = jnp.array(
            [[0, 0.1, 0.2, 0.3, 0, 0.4, 0], [0.4, 0, 0.3, 0.0, 0.1, 0.2, 0]]
        )
        self.assertTrue(jnp.allclose(normalized_weights, result))

    def test_ll(self):
        data = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
        # l = self.sum_layer.log_likelihood_of_nodes_single(data[0])

        ll = self.sum_layer.log_likelihood_of_nodes(data)
        result = jnp.log(
            jnp.array(
                [
                    [
                        0.0,
                        0.4,
                    ],
                    [
                        0.1 * 2,
                        0.0,
                    ],
                    [
                        0.2 * 3,
                        0.3 * 3,
                    ],
                    [
                        0.3 * 4,
                        0.0,
                    ],
                    [
                        0.0,
                        0.1 * 5,
                    ],
                    [
                        0.4 * 6,
                        0.2 * 6,
                    ],
                    [
                        0.0,
                        0.0,
                    ],
                ]
            )
        )
        assert jnp.allclose(ll, result)

    def test_ll_single(self):
        data = jnp.array([0])
        l = self.sum_layer.log_likelihood_of_nodes_single(data)
        result = jnp.log(jnp.array([0.0, 0.4]))
        assert jnp.allclose(l, result)


class NygaDistributionTestCase(unittest.TestCase):

    nx_model: NXProbabilisticCircuit
    jax_model: ProbabilisticCircuit
    data: jax.Array

    @classmethod
    def setUpClass(cls):
        cls.data = jax.random.normal(jax.random.PRNGKey(69), (1000, 1))
        model = NygaDistribution(Continuous("x"), min_samples_per_quantile=10)
        cls.nx_model = model.fit(cls.data)
        cls.jax_model = ProbabilisticCircuit.from_nx(cls.nx_model)
        cls.jax_model.root.validate()

    def test_log_likelihood(self):
        ll = self.jax_model.log_likelihood(self.data)
        self.assertTrue(jnp.all(ll > -jnp.inf))

    def test_to_nx(self):
        nx_model = self.jax_model.to_nx()
        self.assertAlmostEqual(logsumexp(nx_model.root.log_weights), 0.0)
