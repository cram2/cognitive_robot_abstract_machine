import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tqdm
from jax.experimental.sparse import BCOO
from jax.tree_util import tree_flatten
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.distributions import DiracDeltaDistribution
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.jax import SparseSumLayer, UniformLayer
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_event,
    leaf,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProductUnit,
    ProbabilisticCircuit as NXProbabilisticCircuit,
)

np.random.seed(69)


class SmallCircuitIntegrationTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    nx_model = SumUnit()
    jax_model: ProbabilisticCircuit
    nx_model: NXProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        nx_model = NXProbabilisticCircuit()
        sum1, sum2, sum3 = (
            SumUnit(probabilistic_circuit=nx_model),
            SumUnit(probabilistic_circuit=nx_model),
            SumUnit(probabilistic_circuit=nx_model),
        )
        sum4, sum5 = SumUnit(probabilistic_circuit=nx_model), SumUnit(
            probabilistic_circuit=nx_model
        )
        prod1, prod2 = ProductUnit(probabilistic_circuit=nx_model), ProductUnit(
            probabilistic_circuit=nx_model
        )

        sum1.add_subcircuit(prod1, np.log(0.5))
        sum1.add_subcircuit(prod2, np.log(0.5))
        prod1.add_subcircuit(sum2)
        prod1.add_subcircuit(sum4)
        prod2.add_subcircuit(sum3)
        prod2.add_subcircuit(sum5)

        d_x1 = leaf(DiracDeltaDistribution(cls.x, 0, 1), nx_model)
        d_x2 = leaf(DiracDeltaDistribution(cls.x, 1, 2), nx_model)
        d_y1 = leaf(DiracDeltaDistribution(cls.y, 2, 3), nx_model)
        d_y2 = leaf(DiracDeltaDistribution(cls.y, 3, 4), nx_model)

        sum2.add_subcircuit(d_x1, np.log(0.8))
        sum2.add_subcircuit(d_x2, np.log(0.2))
        sum3.add_subcircuit(d_x1, np.log(0.7))
        sum3.add_subcircuit(d_x2, np.log(0.3))

        sum4.add_subcircuit(d_y1, np.log(0.5))
        sum4.add_subcircuit(d_y2, np.log(0.5))
        sum5.add_subcircuit(d_y1, np.log(0.1))
        sum5.add_subcircuit(d_y2, np.log(0.9))

        cls.nx_model = nx_model
        cls.jax_model = ProbabilisticCircuit.from_nx(cls.nx_model)

    def test_creation(self):
        self.assertEqual(self.jax_model.variables, self.nx_model.variables)
        self.assertIsInstance(self.jax_model.root, SparseSumLayer)
        self.assertEqual(self.jax_model.root.number_of_nodes, 1)
        self.assertEqual(len(self.jax_model.root.child_layers), 1)
        product_layer = self.jax_model.root.child_layers[0]
        self.assertEqual(product_layer.number_of_nodes, 2)
        self.assertEqual(len(product_layer.child_layers), 2)
        sum_layer1 = product_layer.child_layers[0]
        sum_layer2 = product_layer.child_layers[1]
        self.assertEqual(sum_layer1.number_of_nodes, 2)
        self.assertEqual(sum_layer2.number_of_nodes, 2)

    def test_ll(self):
        samples = self.nx_model.sample(1000)
        nx_ll = self.nx_model.log_likelihood(samples)
        jax_ll = self.jax_model.log_likelihood(samples)
        self.assertTrue(jnp.allclose(nx_ll, jax_ll))

    def test_trainable_parameters(self):
        params, _ = eqx.partition(self.jax_model.root, eqx.is_inexact_array)
        flattened_params, _ = jax.tree_util.tree_flatten(params)
        number_of_parameters = sum([len(p) for p in flattened_params])
        self.assertEqual(number_of_parameters, 10)


class JPTIntegrationTestCase(unittest.TestCase):
    number_of_variables = 2
    number_of_samples = 1000

    jpt: NXProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        mean = np.full(cls.number_of_variables, 0)
        cov = np.random.uniform(
            0, 1, (cls.number_of_variables, cls.number_of_variables)
        )
        cov = np.dot(cov, cov.T)
        samples = np.random.multivariate_normal(mean, cov, cls.number_of_samples)
        df = pd.DataFrame(
            samples, columns=[f"x_{i}" for i in range(cls.number_of_variables)]
        )
        variables = infer_variables_from_dataframe(df, min_samples_per_quantile=100)
        jpt = JPT(variables, min_samples_leaf=0.1)

        cls.jpt = jpt.fit(df)

    def test_from_jpt(self):
        model = ProbabilisticCircuit.from_nx(self.jpt, False)
        samples = jnp.array(self.jpt.sample(1000))
        jax_ll = model.log_likelihood(samples)
        self.assertTrue((jax_ll > -jnp.inf).all())

    def test_to_nx_pc(self):
        model = ProbabilisticCircuit.from_nx(self.jpt, False)
        model_nx = model.to_nx(True)
        # import matplotlib.pyplot as plt
        # model_nx.root.plot_structure()
        # plt.show()
        samples = jnp.array(model_nx.sample(1000))
        jax_ll = model.log_likelihood(samples)
        self.assertTrue((jax_ll > -jnp.inf).all())


class LearningTestCase(unittest.TestCase):
    data = np.vstack(
        (np.random.uniform(0, 1, (100, 1)), np.random.uniform(2, 3, (200, 1)))
    )
    uniform_layer = UniformLayer(0, jnp.array([[-0.01, 1.01], [1.99, 3.01]]))
    sum_layer = SparseSumLayer(
        [uniform_layer],
        [BCOO((jnp.array([0.0, 0.0]), jnp.array([[0, 0], [0, 1]])), shape=(1, 2))],
    )
    sum_layer.validate()

    def test_learning(self):
        @eqx.filter_jit
        def loss(model, x):
            ll = model.log_likelihood_of_nodes(x)
            return -jnp.mean(ll)

        optim = optax.adamw(0.01)
        opt_state = optim.init(eqx.filter(self.sum_layer, eqx.is_inexact_array))
        model = self.sum_layer
        for _ in tqdm.trange(100):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model, self.data)
            grads_of_sum_layer = eqx.filter(tree_flatten(grads), eqx.is_inexact_array)[
                0
            ][0]
            self.assertTrue(jnp.all(jnp.isfinite(grads_of_sum_layer)))

            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, updates)

        weights = jnp.exp(model.log_weights[0].data)
        weights /= jnp.sum(weights)
        self.assertAlmostEqual(weights[0], 1 / 3, delta=0.01)
        self.assertAlmostEqual(weights[1], 2 / 3, delta=0.01)


class NanGradientTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")

    nx_model: NXProbabilisticCircuit
    jax_model: ProbabilisticCircuit
    event: Event

    @classmethod
    def setUpClass(cls):
        event1 = SimpleEvent(
            {cls.x: closed(0, 1) | closed(2, 3), cls.y: closed(0, 1) | closed(2, 3)}
        ).as_composite_set()
        event2 = SimpleEvent(
            {cls.x: closed(1, 2) | closed(3, 4), cls.y: closed(1, 2) | closed(3, 4)}
        ).as_composite_set()
        cls.event = event1 | event2
        cls.nx_model = uniform_measure_of_event(cls.event)
        cls.jax_model = ProbabilisticCircuit.from_nx(cls.nx_model)

    def test_nan_gradient(self):
        """
        This is the entry point for debugging nans.
        """
        data = jnp.array([[2.5, 2.5]])

        @eqx.filter_jit
        def loss(model, x):
            ll = model.log_likelihood_of_nodes(x)
            return -jnp.mean(ll)

        model = self.jax_model.root

        loss_value, grads = eqx.filter_value_and_grad(loss)(model, data)

        grads_of_sum_layer = eqx.filter(tree_flatten(grads), eqx.is_inexact_array)[0][0]
        # self.assertFalse(jnp.all(jnp.isfinite(grads_of_sum_layer)))


if __name__ == "__main__":
    unittest.main()
