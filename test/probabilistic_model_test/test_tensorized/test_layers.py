"""
Tests for the individual layer classes, the helpers and the integration with a learned
circuit.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from krrood.adapters.json_serializer import from_json, to_json
from random_events.interval import Bound, SimpleInterval, closed, open, reals, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous, Integer
from scipy.sparse import coo_array
from sortedcontainers import SortedSet

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.helper import (
    fully_factorized,
    uniform_measure_of_event,
    uniform_measure_of_simple_event,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    NumberOfWeightsMismatchError,
)
from probabilistic_model.probabilistic_circuit.tensorized.forward_sample_assignment import (
    SampleRowsOfNode,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdge,
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.layer_with_depth import (
    LayerWithDepth,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
    LogProbabilitiesOfLayers,
)
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    RowGroupedSparseArray,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    IntegerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    embedded_logsumexp,
)
from probabilistic_model.utils import MissingDict
from .test_layered_probabilistic_circuit import shared_children_circuit


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


class LogSumExpTestCase(unittest.TestCase):

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


class DecomposabilityTestCase(unittest.TestCase):

    def test_decomposability_is_decided_per_node(self):
        leaf_x = uniform_layer_of(0, [(0, 1)])
        other_leaf_x = uniform_layer_of(0, [(1, 2)])
        leaf_y = uniform_layer_of(1, [(0, 1)])
        # node 0 multiplies x and y, node 1 multiplies x twice
        edges = coo_array(
            (np.zeros(4, dtype=np.int64), ([0, 1, 0, 2], [0, 0, 1, 1])), shape=(3, 2)
        )
        layer = ProductLayer([leaf_x, leaf_y, other_leaf_x], edges)
        np.testing.assert_array_equal(
            layer.is_decomposable_of_nodes(), np.array([True, False])
        )
        self.assertFalse(layer.is_decomposable())


class TruncationOfInputLayersTestCase(unittest.TestCase):
    """
    Truncating an input layer has to keep the number of nodes stable, even when a node
    splits into several pieces or changes its type.
    """

    def circuit_of(self, layer) -> LayeredProbabilisticCircuit:
        return LayeredProbabilisticCircuit(
            SortedSet([x]),
            (
                SumLayer.mixture_of([layer], [0.0])
                if layer.number_of_nodes == 1
                else layer
            ),
        )

    def test_truncating_to_a_composite_interval_introduces_a_selecting_sum_layer(self):
        layer = uniform_layer_of(0, [(0, 4)])
        circuit = LayeredProbabilisticCircuit(
            SortedSet([x]), SumLayer.mixture_of([layer], [0.0])
        )

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
        circuit = LayeredProbabilisticCircuit(
            SortedSet([x]), SumLayer.mixture_of([layer], [0.0])
        )

        event = SimpleEvent.from_data({x: singleton(1.0)}).as_composite_set()
        truncated, probability = circuit.truncated(event, singleton_allowed=True)

        self.assertAlmostEqual(probability, 0.5)
        self.assertTrue(
            any(isinstance(layer, DiracDeltaLayer) for layer in truncated.layers)
        )

    def test_an_impossible_node_is_removed_but_its_siblings_survive(self):
        layer = uniform_layer_of(0, [(0, 1), (2, 3)])
        root = SumLayer(
            [layer],
            RowGroupedSparseArray.from_entries(
                SparseEntries(np.log([0.25, 0.75]), [0, 0], [0, 1]), (1, 2)
            ),
        )
        circuit = LayeredProbabilisticCircuit(SortedSet([x]), root)

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
    python call per node.

    That fast path has to agree with the distribution classes it replaces, in every
    combination of open and closed bounds.
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
                        truncated = layer.log_truncated_of_assignment(
                            event_interval.as_composite_set(), False
                        )
                        truncated_layer = truncated.layer
                        log_probabilities = truncated.log_probabilities

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

    def test_a_singleton_turns_every_node_into_a_dirac_delta(self):
        layer = uniform_layer_of(0, [(0, 2), (1, 5)])
        assignment = singleton(1.0)
        truncated = layer.log_truncated_of_assignment(assignment, True)
        truncated_layer = truncated.layer
        log_probabilities = truncated.log_probabilities
        self.assertIsInstance(truncated_layer, DiracDeltaLayer)
        for node in range(layer.number_of_nodes):
            expected, expected_log_probability = layer.node_distribution(
                node, x
            ).log_conditional_from_simple_interval(assignment.simple_sets[0], True)
            self.assertAlmostEqual(
                float(log_probabilities[node]), float(expected_log_probability)
            )
            self.assertEqual(
                truncated_layer.node_distribution(node, x).location, expected.location
            )

    def test_a_composite_assignment_mixes_the_truncations_to_its_simple_intervals(
        self,
    ):
        layer = uniform_layer_of(0, [(0, 4), (0, 8)])
        assignment = closed(0, 1) | closed(3, 4)
        truncated = layer.log_truncated_of_assignment(assignment, False)
        truncated_layer = truncated.layer
        log_probabilities = truncated.log_probabilities
        self.assertIsInstance(truncated_layer, SumLayer)
        self.assertEqual(truncated_layer.number_of_nodes, layer.number_of_nodes)
        for node in range(layer.number_of_nodes):
            expected = layer.node_distribution(node, x).probability(
                SimpleEvent.from_data({x: assignment}).as_composite_set()
            )
            self.assertAlmostEqual(float(np.exp(log_probabilities[node])), expected)

    def test_dirac_delta_layer_agrees_with_the_scalar_truncation(self):
        layer = DiracDeltaLayer(0, np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        for assignment in (
            closed(0.5, 1.5),
            closed(1.0, 1.0),
            open(1.0, 2.0),
            closed(0.0, 2.0),
            closed(5.0, 6.0),
        ):
            with self.subTest(str(assignment)):
                log_probabilities = layer.log_truncated_of_assignment(
                    assignment, False
                ).log_probabilities
                for node in range(layer.number_of_nodes):
                    distribution = layer.node_distribution(node, x)
                    _, expected = distribution.log_truncated(
                        SimpleEvent.from_data({x: assignment}).as_composite_set()
                    )
                    self.assertEqual(float(log_probabilities[node]), float(expected))

    def test_a_structural_pass_does_not_write_into_the_circuit_it_reads(self):
        """
        Truncating a composite event reuses one circuit for every simple set instead of
        copying it, which is only sound because the pass builds new layers.
        """
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(
            shared_children_circuit()
        )
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
        np.testing.assert_array_equal(layered.log_likelihood(points), likelihood_before)

    def test_discrete_layer_agrees_with_the_scalar_truncation(self):
        distributions = [
            IntegerDistribution(
                variable=n, probabilities=MissingDict(float, {0: 0.2, 1: 0.3, 2: 0.5})
            ),
            IntegerDistribution(variable=n, probabilities=MissingDict(float, {0: 1.0})),
        ]
        layer = IntegerLayer.from_distributions(0, distributions)

        for assignment in (closed(0, 1), closed(2, 2), closed(5, 6)):
            with self.subTest(str(assignment)):
                truncated = layer.log_truncated_of_assignment(assignment, False)
                truncated_layer = truncated.layer
                log_probabilities = truncated.log_probabilities
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
                    truncated = layer.log_truncated_of_assignment(
                        event_interval.as_composite_set(), False
                    )
                    truncated_layer = truncated.layer
                    log_probabilities = truncated.log_probabilities

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
                        self.assertIsInstance(expected, TruncatedGaussianDistribution)
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

        truncated = layer.log_truncated_of_assignment(reals(), False)
        truncated_layer = truncated.layer
        log_probabilities = truncated.log_probabilities

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
        log_probabilities = layer.log_truncated_of_assignment(
            closed(99.0, 101.0), False
        ).log_probabilities
        self.assertEqual(log_probabilities[0], -np.inf)
        self.assertGreater(log_probabilities[1], -np.inf)

    def test_a_singleton_turns_every_gaussian_node_into_a_dirac_delta(self):
        layer = GaussianLayer.from_distributions(
            0, [GaussianDistribution(variable=x, location=0.0, scale=1.0)]
        )
        truncated = layer.log_truncated_of_assignment(singleton(1.0), True)
        self.assertIsInstance(truncated.layer, DiracDeltaLayer)
        self.assertAlmostEqual(
            float(truncated.log_probabilities[0]),
            float(layer.log_likelihood_of_nodes_from_column(np.array([1.0]))[0, 0]),
        )

    def test_a_composite_assignment_mixes_the_truncated_gaussians(self):
        distribution = GaussianDistribution(variable=x, location=0.0, scale=1.0)
        layer = GaussianLayer.from_distributions(0, [distribution])
        assignment = closed(-1, 0) | closed(1, 2)
        truncated = layer.log_truncated_of_assignment(assignment, False)
        self.assertIsInstance(truncated.layer, SumLayer)
        self.assertIsInstance(truncated.layer.child_layers[0], TruncatedGaussianLayer)
        self.assertAlmostEqual(
            float(np.exp(truncated.log_probabilities[0])),
            distribution.probability(
                SimpleEvent.from_data({x: assignment}).as_composite_set()
            ),
        )

    def test_truncated_gaussian_layer_agrees_with_the_scalar_truncation(self):
        """
        The expected distribution is built on the intersection of the support and the
        assignment directly: ``TruncatedGaussianDistribution.log_truncated`` keeps the
        interval of the event instead of that intersection.
        """
        support = closed(-1.0, 3.0)
        distributions = [
            TruncatedGaussianDistribution(
                variable=x,
                interval=support.simple_sets[0],
                location=location,
                scale=scale,
            )
            for location, scale in ((0.0, 1.0), (2.5, 0.5))
        ]
        layer = TruncatedGaussianLayer.from_distributions(0, distributions)
        points = np.linspace(-1.5, 3.5, 11).reshape(-1, 1)
        for assignment in (closed(0.0, 1.0), open(-5.0, 0.5), closed(2.0, 10.0)):
            with self.subTest(str(assignment)):
                truncated = layer.log_truncated_of_assignment(assignment, False)
                self.assertIsInstance(truncated.layer, TruncatedGaussianLayer)
                [intersection] = (support & assignment).simple_sets
                for node, distribution in enumerate(distributions):
                    event = SimpleEvent.from_data({x: assignment}).as_composite_set()
                    self.assertAlmostEqual(
                        float(np.exp(truncated.log_probabilities[node])),
                        distribution.probability(event),
                    )
                    expected = TruncatedGaussianDistribution(
                        variable=x,
                        interval=intersection,
                        location=distribution.location,
                        scale=distribution.scale,
                    )
                    self.assertEqual(
                        truncated.layer.simple_interval_of(node), intersection
                    )
                    np.testing.assert_allclose(
                        np.exp(
                            truncated.layer.log_likelihood_of_nodes(points)[:, node]
                        ),
                        expected.likelihood(points),
                        atol=1e-9,
                    )

    def test_conditioning_a_discrete_layer_on_a_value(self):
        layer = IntegerLayer.from_distributions(
            0,
            [
                IntegerDistribution(
                    variable=n, probabilities=MissingDict(float, {0: 0.2, 2: 0.8})
                ),
                IntegerDistribution(
                    variable=n, probabilities=MissingDict(float, {0: 1.0})
                ),
            ],
        )
        conditioned = layer.log_conditional_of_value(2)
        np.testing.assert_allclose(conditioned.log_probabilities, np.log([0.8, 0.0]))
        self.assertEqual(
            conditioned.layer.probabilities_of_node(0), MissingDict(float, {2: 1.0})
        )
        np.testing.assert_array_equal(
            layer.log_conditional_of_value(7).log_probabilities, [-np.inf, -np.inf]
        )


class LayerGraphTraversalTestCase(unittest.TestCase):
    """
    Walking the layer graph of a circuit in which one layer has two parents that sit at
    different distances from the root.
    """

    def setUp(self):
        self.leaf_x = uniform_layer_of(0, [(0, 1)])
        self.leaf_y = UniformLayer.from_distributions(
            1, [UniformDistribution(variable=y, interval=closed(0, 1).simple_sets[0])]
        )
        self.product = ProductLayer.product_of([self.leaf_x, self.leaf_y])
        self.wrapper = SumLayer.mixture_of([self.product], [np.log(1.0)])
        # the product layer is both a child of the root and, through the wrapper, its
        # grandchild, so it is reachable at two different depths
        self.root = SumLayer.mixture_of(
            [self.product, self.wrapper], np.log([0.5, 0.5])
        )

    def test_an_input_layer_has_no_child_layers(self):
        self.assertEqual([], self.leaf_x.child_layers)

    def test_all_layers_reports_a_shared_layer_once(self):
        layers = self.root.all_layers()
        self.assertEqual(
            {id(self.root), id(self.product), id(self.wrapper)}
            | {id(self.leaf_x), id(self.leaf_y)},
            {id(layer) for layer in layers},
        )
        self.assertEqual(len({id(layer) for layer in layers}), len(layers))

    def test_all_layers_with_depth_pairs_every_layer_with_its_depth(self):
        self.assertEqual(
            LayerWithDepth(0, self.root), self.root.all_layers_with_depth()[0]
        )

    def test_all_layers_with_depth_reports_a_shared_layer_once_per_path(self):
        self.assertEqual(
            [1, 2],
            sorted(
                entry.depth
                for entry in self.root.all_layers_with_depth()
                if entry.layer is self.product
            ),
        )

    def test_all_layers_visits_every_parent_before_the_layer(self):
        order = self.root.all_layers()
        positions = {id(layer): index for index, layer in enumerate(order)}

        self.assertEqual(len(self.root.all_layers()), len(order))
        for layer in order:
            for child_layer in layer.child_layers:
                self.assertLess(positions[id(layer)], positions[id(child_layer)])


class HelperTestCase(unittest.TestCase):

    def test_uniform_measure_of_a_simple_event(self):
        event = SimpleEvent.from_data({x: closed(0.0, 2.0), y: closed(0.0, 4.0)})
        circuit = RustworkxCircuitToLayeredCircuitConverter.convert(
            uniform_measure_of_simple_event(event)
        )
        self.assertEqual(list(circuit.variables), [x, y])
        np.testing.assert_allclose(
            circuit.likelihood(np.array([[1.0, 2.0]])), np.array([1 / 8])
        )
        self.assertAlmostEqual(circuit.probability_of_simple_event(event), 1.0)

    def test_fully_factorized(self):
        circuit = RustworkxCircuitToLayeredCircuitConverter.convert(
            fully_factorized([x, y], means={x: 1.0}, variances={y: 2.0})
        )
        self.assertEqual(list(circuit.variables), [x, y])
        self.assertAlmostEqual(circuit.expectation()[x], 1.0)
        self.assertAlmostEqual(circuit.expectation()[y], 0.0)

    def test_uniform_measure_of_a_composite_event(self):
        event = (
            SimpleEvent.from_data(
                {x: closed(0.0, 1.0), y: closed(0.0, 1.0)}
            ).as_composite_set()
            | SimpleEvent.from_data(
                {x: closed(2.0, 3.0), y: closed(2.0, 3.0)}
            ).as_composite_set()
        )
        circuit = RustworkxCircuitToLayeredCircuitConverter.convert(
            uniform_measure_of_event(event.__deepcopy__())
        )
        self.assertAlmostEqual(circuit.probability(event.__deepcopy__()), 1.0)


class SingleNodeLayerTestCase(unittest.TestCase):

    def test_product_of_and_mixture_of(self):
        layer_x = uniform_layer_of(0, [(0, 1)])
        layer_y = UniformLayer.from_distributions(
            1, [UniformDistribution(variable=y, interval=closed(0, 2).simple_sets[0])]
        )
        product = ProductLayer.product_of([layer_x, layer_y])
        self.assertIsInstance(product, ProductLayer)

        circuit = LayeredProbabilisticCircuit(SortedSet([x, y]), product)
        np.testing.assert_allclose(
            circuit.likelihood(np.array([[0.5, 1.0]])), np.array([0.5])
        )

        mixture = SumLayer.mixture_of([product], [np.log(1.0)])
        self.assertIsInstance(mixture, SumLayer)
        np.testing.assert_allclose(
            LayeredProbabilisticCircuit(SortedSet([x, y]), mixture).likelihood(
                np.array([[0.5, 1.0]])
            ),
            np.array([0.5]),
        )

    def test_mixture_of_rejects_a_wrong_number_of_weights(self):
        with self.assertRaises(NumberOfWeightsMismatchError):
            SumLayer.mixture_of([uniform_layer_of(0, [(0, 1)])], [0.0, 0.0])


class QueryDatastructureTestCase(unittest.TestCase):

    def test_edges_into_one_child_layer(self):
        edges = InnerLayerEdges(
            np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0]), np.array([2, 0, 1, 3])
        )
        into_first = edges.into_child_layer(0)
        self.assertEqual(
            list(into_first),
            [InnerLayerEdge(0, 0, 2), InnerLayerEdge(1, 0, 1), InnerLayerEdge(1, 0, 3)],
        )
        self.assertFalse(into_first.every_node_at_most_once)
        self.assertTrue(edges.into_child_layer(1).every_node_at_most_once)

    def test_product_layer_edges_per_child_layer(self):
        layer = ProductLayer.product_of(
            [uniform_layer_of(0, [(0, 1)]), uniform_layer_of(1, [(0, 1)])]
        )
        self.assertEqual(
            [list(edges) for edges in layer.edges_per_child_layer],
            [[InnerLayerEdge(0, 0, 0)], [InnerLayerEdge(0, 1, 0)]],
        )

    def test_sparse_entries_concatenate_into_one_array(self):
        entries = SparseEntries.concatenate(
            [SparseEntries([1.0], [0], [1]), SparseEntries([2.0], [1], [0])]
        )
        np.testing.assert_array_equal(
            entries.to_coo_array((2, 2)).toarray(), np.array([[0.0, 1.0], [2.0, 0.0]])
        )

    def test_sample_rows_of_a_node_join_the_chunks_of_all_parents(self):
        rows = SampleRowsOfNode()
        self.assertTrue(rows.is_empty)
        rows.add(np.array([0, 2]))
        rows.add(np.array([5]))
        self.assertFalse(rows.is_empty)
        np.testing.assert_array_equal(rows.rows, np.array([0, 2, 5]))

    def test_moment_query_from_maps(self):
        query = MomentQuery.from_maps({y: 2}, {x: 1.0, y: 3.0}, SortedSet([x, y]))
        np.testing.assert_array_equal(query.order, np.array([0, 2]))
        np.testing.assert_array_equal(query.center, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(query.requested, np.array([False, True]))
        self.assertEqual(query.number_of_variables, 2)

    def test_alive_nodes_of_recorded_and_unrecorded_layers(self):
        recorded = uniform_layer_of(0, [(0, 1), (1, 2)])
        unrecorded = uniform_layer_of(0, [(0, 1)])
        log_probabilities = LogProbabilitiesOfLayers()
        log_probabilities.record(
            LayerWithLogProbabilities(recorded, np.array([-np.inf, 0.0]))
        )
        np.testing.assert_array_equal(
            log_probabilities.alive_nodes_of(recorded), np.array([False, True])
        )
        np.testing.assert_array_equal(
            log_probabilities.alive_nodes_of(unrecorded), np.array([True])
        )

    def test_uniform_moment_is_the_difference_of_the_antiderivative(self):
        layer = uniform_layer_of(0, [(0, 2), (1, 5)])
        np.testing.assert_allclose(
            layer.moment_of_nodes_own(1, 0.0, x), np.array([1.0, 3.0])
        )
        np.testing.assert_allclose(
            layer.antiderivative_of_moment_at(np.array([2.0, 5.0]), 1, 0.0),
            np.array([1.0, 25 / 8]),
        )


class SupportWithoutCopiesTestCase(unittest.TestCase):
    """
    The support and mode queries combine the events of the children with operations
    that return new events, so they must not alter the events of the children.
    """

    def test_the_support_of_a_sum_does_not_alter_the_support_of_its_children(self):
        child = uniform_layer_of(0, [(0, 1), (2, 3)])
        root = SumLayer(
            [child],
            RowGroupedSparseArray.from_entries(
                SparseEntries(np.log([0.5, 0.5]), [0, 0], [0, 1]), (1, 2)
            ),
        )
        variables = SortedSet([x])
        [support] = root.support_of_nodes(variables)
        self.assertEqual(
            support,
            SimpleEvent.from_data({x: closed(0, 1) | closed(2, 3)}).as_composite_set(),
        )
        self.assertEqual(
            child.support_of_nodes(variables)[0],
            SimpleEvent.from_data({x: closed(0, 1)}).as_composite_set(),
        )

    def test_the_mode_of_a_product_fills_in_the_variables_of_the_other_factors(self):
        leaf_x = uniform_layer_of(0, [(0, 1)])
        leaf_y = UniformLayer.from_distributions(
            1, [UniformDistribution(variable=y, interval=closed(0, 2).simple_sets[0])]
        )
        product = ProductLayer.product_of([leaf_x, leaf_y])
        [mode], values = product.log_mode_of_nodes(SortedSet([x, y]))
        self.assertEqual(
            mode,
            SimpleEvent.from_data(
                {x: closed(0, 1), y: closed(0, 2)}
            ).as_composite_set(),
        )
        self.assertAlmostEqual(float(values[0]), np.log(0.5))


class JointProbabilityTreeIntegrationTestCase(unittest.TestCase):
    """
    A learned circuit is a much larger and more irregular graph than the hand built
    ones, so it exercises the conversion and the queries on a realistic structure.
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
        cls.layered = RustworkxCircuitToLayeredCircuitConverter.convert(cls.rx_circuit)

    def setUp(self):
        np.random.seed(69)

    def test_conversion_is_valid(self):
        self.layered.validate()
        self.assertEqual(list(self.layered.variables), list(self.rx_circuit.variables))
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
        back = LayeredCircuitToRustworkxCircuitConverter.convert(self.layered)
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
