"""
Tests for the individual layer classes, the sparse array they are built on, the helpers
and the integration with a learned circuit.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from krrood.adapters.json_serializer import from_json, to_json
from random_events.interval import Bound, SimpleInterval, closed, open, reals, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous, Integer

from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.tensorized.discrete_layer import IntegerLayer
from probabilistic_model.probabilistic_circuit.tensorized.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.helper import (
    fully_factorized,
    mixture_of,
    product_of,
    uniform_measure_of_event,
    uniform_measure_of_simple_event,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import (
    ProductLayer,
    SparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.uniform_layer import UniformLayer
from .test_layered_probabilistic_circuit import shared_children_circuit
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    SparseArray,
    embedded_logsumexp,
)

from probabilistic_model.utils import MissingDict

x = Continuous("x")
y = Continuous("y")
n = Integer("n")


def uniform_layer_of(variable_index: int, intervals) -> UniformLayer:
    return UniformLayer.from_distributions(
        variable_index,
        [
            UniformDistribution(
                variable=x, interval=closed(lower, upper).simple_sets[0]
            )
            for lower, upper in intervals
        ],
    )


class SparseArrayTestCase(unittest.TestCase):

    def test_dense_round_trip_keeps_explicit_zeros(self):
        dense = np.array([[0, 3], [2, 0]])
        sparse = SparseArray.from_dense(dense)
        np.testing.assert_array_equal(sparse.to_dense(), dense)

        # a product layer stores child node indices, where zero is a real edge
        edges = SparseArray.from_coordinates([0, 1], [0, 0], [0, 0], (2, 1))
        self.assertEqual(edges.number_of_stored_entries, 2)
        np.testing.assert_array_equal(edges.data, np.array([0, 0]))

    def test_sort_indices(self):
        sparse = SparseArray.from_coordinates(
            [1, 0, 1], [0, 2, 1], [10.0, 20.0, 30.0], (2, 3)
        )
        sorted_sparse = sparse.sort_indices()
        np.testing.assert_array_equal(sorted_sparse.rows, np.array([0, 1, 1]))
        np.testing.assert_array_equal(sorted_sparse.columns, np.array([2, 0, 1]))
        np.testing.assert_array_equal(
            sorted_sparse.data, np.array([20.0, 10.0, 30.0])
        )

    def test_json_round_trip(self):
        sparse = SparseArray.from_coordinates([0, 1], [1, 0], [1.5, -2.5], (2, 2))
        restored = SparseArray.from_json(sparse.to_json())
        np.testing.assert_array_equal(restored.to_dense(), sparse.to_dense())
        self.assertEqual(restored.shape, sparse.shape)

    def test_copy_shares_no_memory(self):
        sparse = SparseArray.from_coordinates([0], [0], [1.0], (1, 1))
        copy = sparse.copy()
        copy.data[0] = 5.0
        self.assertEqual(sparse.data[0], 1.0)

    def test_embedded_logsumexp(self):
        np.testing.assert_allclose(
            embedded_logsumexp(
                np.array([[np.log(0.2), np.log(0.3)], [np.log(0.5), -np.inf]]), axis=1
            ),
            np.array([np.log(0.5), np.log(0.5)]),
        )

    def test_logsumexp_of_only_minus_infinity_is_minus_infinity(self):
        # the padding of the edge gather is -inf, so a node without edges reduces to a
        # whole row of -inf and must not turn into a nan
        np.testing.assert_array_equal(
            embedded_logsumexp(np.array([[-np.inf, -np.inf]]), axis=1),
            np.array([-np.inf]),
        )


class TruncationOfInputLayersTestCase(unittest.TestCase):
    """
    Truncating an input layer has to keep the number of nodes stable, even when a node
    splits into several pieces or changes its type.
    """

    def circuit_of(self, layer) -> LayeredProbabilisticCircuit:
        return LayeredProbabilisticCircuit(
            [x], mixture_of([layer], [0.0]) if layer.number_of_nodes == 1 else layer
        )

    def test_truncating_to_a_composite_interval_introduces_a_selecting_sum_layer(self):
        layer = uniform_layer_of(0, [(0, 4)])
        circuit = LayeredProbabilisticCircuit([x], mixture_of([layer], [0.0]))

        event = SimpleEvent.from_data(
            {x: closed(0.0, 1.0) | closed(3.0, 4.0)}
        ).as_composite_set()
        truncated, probability = circuit.truncated(event)

        self.assertAlmostEqual(probability, 0.5)

        # the single uniform node became two, mixed back together by a sum layer
        uniform_layers = [
            candidate
            for candidate in truncated.layers
            if isinstance(candidate, UniformLayer)
        ]
        self.assertEqual(len(uniform_layers), 1)
        self.assertEqual(uniform_layers[0].number_of_nodes, 2)

        np.testing.assert_allclose(
            truncated.likelihood(np.array([[0.5], [2.0], [3.5]])),
            np.array([0.5, 0.0, 0.5]),
        )

    def test_truncating_to_a_singleton(self):
        layer = uniform_layer_of(0, [(0, 2)])
        circuit = LayeredProbabilisticCircuit([x], mixture_of([layer], [0.0]))

        event = SimpleEvent.from_data({x: singleton(1.0)}).as_composite_set()
        truncated, probability = circuit.truncated(event, singleton_allowed=True)

        self.assertAlmostEqual(probability, 0.5)
        self.assertTrue(
            any(isinstance(layer, DiracDeltaLayer) for layer in truncated.layers)
        )

    def test_an_impossible_node_is_removed_but_its_siblings_survive(self):
        layer = uniform_layer_of(0, [(0, 1), (2, 3)])
        root = SparseSumLayer(
            [layer],
            [
                SparseArray.from_coordinates(
                    [0, 0], [0, 1], np.log([0.25, 0.75]), (1, 2)
                )
            ],
        )
        circuit = LayeredProbabilisticCircuit([x], root)

        event = SimpleEvent.from_data({x: closed(2.0, 3.0)}).as_composite_set()
        truncated, probability = circuit.truncated(event)

        self.assertAlmostEqual(probability, 0.75)
        remaining = [
            candidate
            for candidate in truncated.layers
            if isinstance(candidate, UniformLayer)
        ]
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].number_of_nodes, 1)
        np.testing.assert_allclose(
            truncated.likelihood(np.array([[2.5], [0.5]])), np.array([1.0, 0.0])
        )


class VectorizedTruncationTestCase(unittest.TestCase):
    """
    The input layers truncate all of their nodes with array arithmetic instead of one
    python call per node. That fast path has to agree with the distribution classes it
    replaces, in every combination of open and closed bounds.
    """

    def test_uniform_layer_agrees_with_the_scalar_truncation(self):
        bound_pairs = [
            (Bound.CLOSED, Bound.CLOSED),
            (Bound.CLOSED, Bound.OPEN),
            (Bound.OPEN, Bound.CLOSED),
            (Bound.OPEN, Bound.OPEN),
        ]
        # supports that overlap the event fully, partially, at a point and not at all
        node_ranges = [(0.0, 1.0), (0.5, 2.5), (2.0, 3.0), (3.0, 4.0), (-1.0, 5.0)]
        event_ranges = [(0.5, 2.5), (0.0, 4.0), (2.0, 2.0), (10.0, 11.0)]

        for node_bounds in bound_pairs:
            for event_bounds in bound_pairs:
                distributions = [
                    UniformDistribution(
                        variable=x,
                        interval=SimpleInterval.from_data(lower, upper, *node_bounds),
                    )
                    for lower, upper in node_ranges
                ]
                layer = UniformLayer.from_distributions(0, distributions)

                for lower, upper in event_ranges:
                    event_interval = SimpleInterval.from_data(
                        lower, upper, *event_bounds
                    )
                    with self.subTest(
                        node_bounds=node_bounds,
                        event_bounds=event_bounds,
                        event=(lower, upper),
                    ):
                        vectorized = layer.log_truncated_of_assignment(
                            event_interval.as_composite_set(), False
                        )
                        self.assertIsNotNone(vectorized)
                        truncated_layer, log_probabilities = vectorized

                        for node, distribution in enumerate(distributions):
                            expected, expected_log_probability = (
                                distribution.log_conditional_from_simple_interval(
                                    event_interval, False
                                )
                            )
                            if expected is None:
                                self.assertEqual(log_probabilities[node], -np.inf)
                                continue

                            self.assertAlmostEqual(
                                float(log_probabilities[node]),
                                float(expected_log_probability),
                            )
                            self.assertEqual(
                                truncated_layer.simple_interval_of(node),
                                expected.interval,
                            )

    def test_a_singleton_event_falls_back_to_the_scalar_path(self):
        layer = uniform_layer_of(0, [(0, 2)])
        self.assertIsNone(
            layer.log_truncated_of_assignment(singleton(1.0), True)
        )

    def test_a_composite_assignment_falls_back_to_the_scalar_path(self):
        layer = uniform_layer_of(0, [(0, 4)])
        self.assertIsNone(
            layer.log_truncated_of_assignment(closed(0, 1) | closed(3, 4), False)
        )

    def test_dirac_delta_layer_agrees_with_the_scalar_truncation(self):
        layer = DiracDeltaLayer(
            0, np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])
        )
        for assignment in (
            closed(0.5, 1.5),
            closed(1.0, 1.0),
            open(1.0, 2.0),
            closed(0.0, 2.0),
            closed(5.0, 6.0),
        ):
            with self.subTest(str(assignment)):
                _, log_probabilities = layer.log_truncated_of_assignment(
                    assignment, False
                )
                for node in range(layer.number_of_nodes):
                    distribution = layer.node_distribution(node, x)
                    _, expected = distribution.log_truncated(
                        SimpleEvent.from_data({x: assignment}).as_composite_set()
                    )
                    self.assertEqual(
                        float(log_probabilities[node]), float(expected)
                    )

    def test_discrete_layer_agrees_with_the_scalar_truncation(self):
        distributions = [
            IntegerDistribution(
                variable=n, probabilities=MissingDict(float, {0: 0.2, 1: 0.3, 2: 0.5})
            ),
            IntegerDistribution(
                variable=n, probabilities=MissingDict(float, {0: 1.0})
            ),
        ]
        layer = IntegerLayer.from_distributions(0, distributions)

        for assignment in (closed(0, 1), closed(2, 2), closed(5, 6)):
            with self.subTest(str(assignment)):
                truncated_layer, log_probabilities = (
                    layer.log_truncated_of_assignment(assignment, False)
                )
                for node, distribution in enumerate(distributions):
                    expected, expected_log_probability = distribution.log_truncated(
                        SimpleEvent.from_data({n: assignment}).as_composite_set()
                    )
                    if expected is None:
                        self.assertEqual(log_probabilities[node], -np.inf)
                        continue
                    self.assertAlmostEqual(
                        float(log_probabilities[node]),
                        float(expected_log_probability),
                    )
                    self.assertEqual(
                        truncated_layer.probabilities_of_node(node),
                        expected.probabilities,
                    )

    def test_gaussian_layer_agrees_with_the_scalar_truncation(self):
        bound_pairs = [
            (Bound.CLOSED, Bound.CLOSED),
            (Bound.CLOSED, Bound.OPEN),
            (Bound.OPEN, Bound.CLOSED),
            (Bound.OPEN, Bound.OPEN),
        ]
        # locations and scales that overlap the event fully, partially and barely at all
        node_parameters = [(0.0, 1.0), (2.0, 0.5), (-3.0, 2.0), (100.0, 0.01)]
        event_ranges = [(-1.0, 1.0), (-10.0, 10.0), (5.0, 6.0)]

        distributions = [
            GaussianDistribution(variable=x, location=location, scale=scale)
            for location, scale in node_parameters
        ]
        layer = GaussianLayer.from_distributions(0, distributions)

        for event_bounds in bound_pairs:
            for lower, upper in event_ranges:
                event_interval = SimpleInterval.from_data(lower, upper, *event_bounds)
                with self.subTest(event_bounds=event_bounds, event=(lower, upper)):
                    vectorized = layer.log_truncated_of_assignment(
                        event_interval.as_composite_set(), False
                    )
                    self.assertIsNotNone(vectorized)
                    truncated_layer, log_probabilities = vectorized

                    for node, distribution in enumerate(distributions):
                        expected, expected_log_probability = (
                            distribution.log_conditional_from_simple_interval(
                                event_interval, False
                            )
                        )
                        if expected is None:
                            self.assertEqual(log_probabilities[node], -np.inf)
                            continue

                        self.assertAlmostEqual(
                            float(log_probabilities[node]),
                            float(expected_log_probability),
                        )
                        self.assertIsInstance(
                            expected, TruncatedGaussianDistribution
                        )
                        self.assertEqual(
                            truncated_layer.simple_interval_of(node),
                            expected.interval,
                        )

    def test_gaussian_layer_truncated_to_the_real_line_stays_gaussian(self):
        distributions = [
            GaussianDistribution(variable=x, location=0.0, scale=1.0),
            GaussianDistribution(variable=x, location=5.0, scale=2.0),
        ]
        layer = GaussianLayer.from_distributions(0, distributions)

        vectorized = layer.log_truncated_of_assignment(reals(), False)
        self.assertIsNotNone(vectorized)
        truncated_layer, log_probabilities = vectorized

        self.assertIsInstance(truncated_layer, GaussianLayer)
        self.assertNotIsInstance(truncated_layer, TruncatedGaussianLayer)
        np.testing.assert_allclose(log_probabilities, 0.0, atol=1e-9)
        np.testing.assert_array_equal(truncated_layer.location, layer.location)
        np.testing.assert_array_equal(truncated_layer.scale, layer.scale)

    def test_gaussian_layer_marks_a_node_with_no_probability_left_as_impossible(self):
        layer = GaussianLayer.from_distributions(
            0,
            [
                GaussianDistribution(variable=x, location=0.0, scale=0.001),
                GaussianDistribution(variable=x, location=100.0, scale=1.0),
            ],
        )
        _, log_probabilities = layer.log_truncated_of_assignment(
            closed(99.0, 101.0), False
        )
        self.assertEqual(log_probabilities[0], -np.inf)
        self.assertGreater(log_probabilities[1], -np.inf)

    def test_a_singleton_event_falls_back_to_the_scalar_path_for_a_gaussian_layer(self):
        layer = GaussianLayer.from_distributions(
            0, [GaussianDistribution(variable=x, location=0.0, scale=1.0)]
        )
        self.assertIsNone(layer.log_truncated_of_assignment(singleton(1.0), True))

    def test_a_composite_assignment_falls_back_to_the_scalar_path_for_a_gaussian_layer(
        self,
    ):
        layer = GaussianLayer.from_distributions(
            0, [GaussianDistribution(variable=x, location=0.0, scale=1.0)]
        )
        self.assertIsNone(
            layer.log_truncated_of_assignment(closed(0, 1) | closed(3, 4), False)
        )

    def test_a_structural_pass_does_not_write_into_the_circuit_it_reads(self):
        """
        Truncating a composite event reuses one circuit for every simple set instead of
        copying it, which is only sound because the pass builds new layers.
        """
        layered = LayeredProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        layers_before = list(layered.layers)
        parameters_before = [
            (type(layer).__name__, layer.number_of_nodes, to_json(layer))
            for layer in layers_before
        ]
        points = np.array([[0.5, 0.5], [1.5, 1.5]])
        likelihood_before = layered.log_likelihood(points)

        event = (
            SimpleEvent.from_data(
                {x: closed(0.0, 0.5), y: closed(0.0, 0.5)}
            ).as_composite_set()
            | SimpleEvent.from_data(
                {x: closed(1.5, 2.0), y: closed(1.5, 2.0)}
            ).as_composite_set()
        )
        layered.truncated(event.__deepcopy__())

        self.assertEqual(
            [id(layer) for layer in layered.layers],
            [id(layer) for layer in layers_before],
        )
        self.assertEqual(
            [
                (type(layer).__name__, layer.number_of_nodes, to_json(layer))
                for layer in layered.layers
            ],
            parameters_before,
        )
        np.testing.assert_array_equal(
            layered.log_likelihood(points), likelihood_before
        )


class HelperTestCase(unittest.TestCase):

    def test_uniform_measure_of_a_simple_event(self):
        event = SimpleEvent.from_data({x: closed(0.0, 2.0), y: closed(0.0, 4.0)})
        circuit = uniform_measure_of_simple_event(event)
        self.assertEqual(list(circuit.variables), [x, y])
        np.testing.assert_allclose(
            circuit.likelihood(np.array([[1.0, 2.0]])), np.array([1 / 8])
        )
        self.assertAlmostEqual(circuit.probability_of_simple_event(event), 1.0)

    def test_uniform_measure_of_a_composite_event(self):
        event = (
            SimpleEvent.from_data(
                {x: closed(0.0, 1.0), y: closed(0.0, 1.0)}
            ).as_composite_set()
            | SimpleEvent.from_data(
                {x: closed(2.0, 3.0), y: closed(2.0, 3.0)}
            ).as_composite_set()
        )
        circuit = uniform_measure_of_event(event.__deepcopy__())
        self.assertAlmostEqual(circuit.probability(event.__deepcopy__()), 1.0)

    def test_fully_factorized(self):
        circuit = fully_factorized([x, y], means={x: 1.0}, variances={y: 2.0})
        self.assertEqual(list(circuit.variables), [x, y])
        self.assertAlmostEqual(circuit.expectation()[x], 1.0)
        self.assertAlmostEqual(circuit.expectation()[y], 0.0)

    def test_product_of_and_mixture_of(self):
        layer_x = uniform_layer_of(0, [(0, 1)])
        layer_y = UniformLayer.from_distributions(
            1, [UniformDistribution(variable=y, interval=closed(0, 2).simple_sets[0])]
        )
        product = product_of([x, y], [layer_x, layer_y])
        self.assertIsInstance(product, ProductLayer)

        circuit = LayeredProbabilisticCircuit([x, y], product)
        np.testing.assert_allclose(
            circuit.likelihood(np.array([[0.5, 1.0]])), np.array([0.5])
        )

        mixture = mixture_of([product], [np.log(1.0)])
        self.assertIsInstance(mixture, SparseSumLayer)
        np.testing.assert_allclose(
            LayeredProbabilisticCircuit([x, y], mixture).likelihood(
                np.array([[0.5, 1.0]])
            ),
            np.array([0.5]),
        )

    def test_mixture_of_rejects_a_wrong_number_of_weights(self):
        with self.assertRaises(ValueError):
            mixture_of([uniform_layer_of(0, [(0, 1)])], [0.0, 0.0])


class JointProbabilityTreeIntegrationTestCase(unittest.TestCase):
    """
    A learned circuit is a much larger and more irregular graph than the hand built ones,
    so it exercises the conversion and the queries on a realistic structure.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        number_of_variables = 3
        covariance = np.random.uniform(0, 1, (number_of_variables, number_of_variables))
        covariance = covariance @ covariance.T
        samples = np.random.multivariate_normal(
            np.zeros(number_of_variables), covariance, 1000
        )
        frame = pd.DataFrame(
            samples, columns=[f"x_{index}" for index in range(number_of_variables)]
        )
        variables = infer_variables_from_dataframe(frame, min_samples_per_quantile=100)
        cls.rx_circuit = JointProbabilityTree(
            annotated_variables=variables, min_samples_per_leaf=0.1
        ).fit(frame)
        cls.layered = LayeredProbabilisticCircuit.from_rustworkx(cls.rx_circuit)

    def setUp(self):
        np.random.seed(69)

    def test_conversion_is_valid(self):
        self.layered.validate()
        self.assertEqual(
            list(self.layered.variables), list(self.rx_circuit.variables)
        )
        self.assertTrue(self.layered.is_decomposable())

    def test_log_likelihood(self):
        samples = self.rx_circuit.sample(500)
        np.testing.assert_allclose(
            self.layered.log_likelihood(samples),
            self.rx_circuit.log_likelihood(samples),
        )

    def test_probability_of_a_simple_event(self):
        event = self.rx_circuit.support.bounding_box()
        self.assertAlmostEqual(
            self.layered.probability_of_simple_event(event),
            self.rx_circuit.probability_of_simple_event(event),
        )

    def test_expectation(self):
        for variable in self.layered.variables:
            self.assertAlmostEqual(
                self.layered.expectation()[variable],
                self.rx_circuit.expectation()[variable],
            )

    def test_sampling(self):
        samples = self.layered.sample(2000)
        self.assertFalse(np.any(np.isnan(samples)))
        self.assertTrue(np.all(self.layered.log_likelihood(samples) > -np.inf))

    def test_truncation(self):
        bounding_box = self.rx_circuit.support.bounding_box()
        variable = self.layered.variables[0]
        interval = bounding_box[variable].simple_sets[0]
        half = SimpleInterval.from_data(
            interval.lower,
            (interval.lower + interval.upper) / 2,
            interval.left,
            interval.right,
        )
        event = SimpleEvent.from_data(
            {variable: half.as_composite_set()}
        ).as_composite_set()

        rx_truncated, rx_probability = self.rx_circuit.truncated(event.__deepcopy__())
        truncated, probability = self.layered.truncated(event.__deepcopy__())

        self.assertAlmostEqual(probability, rx_probability)
        truncated.validate()

        samples = truncated.sample(300)
        np.testing.assert_allclose(
            truncated.log_likelihood(samples), rx_truncated.log_likelihood(samples)
        )

    def test_round_trip_through_rustworkx(self):
        back = self.layered.to_rustworkx()
        samples = self.rx_circuit.sample(300)
        np.testing.assert_allclose(
            back.log_likelihood(samples), self.rx_circuit.log_likelihood(samples)
        )

    def test_json_round_trip(self):
        restored = from_json(to_json(self.layered))
        samples = self.rx_circuit.sample(300)
        np.testing.assert_allclose(
            restored.log_likelihood(samples), self.layered.log_likelihood(samples)
        )

    def test_the_layered_circuit_has_fewer_layers_than_nodes(self):
        # the point of the layout: many nodes are folded into few parameter blocks
        self.assertLess(len(self.layered.layers), self.layered.number_of_nodes)


if __name__ == "__main__":
    unittest.main()
