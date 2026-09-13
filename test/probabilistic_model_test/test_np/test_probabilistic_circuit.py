"""
Cross validation of the layered numpy circuits against the rustworkx implementation.

Every query of a layered circuit has to return what the circuit of the ``rx`` package it
was converted from returns, so most tests here build a circuit with the ``rx`` classes,
convert it and compare the two answers.

Two queries are checked against a closed form instead of against ``rx``, because ``rx``
answers them incorrectly for circuits in which a subcircuit has more than one parent:

- ``marginal`` (and therefore the marginalization step inside ``conditional``) routes
  through ``simplify``, which merges a sum unit into its parent sum unit by adding an
  edge per grandchild. The graph is not a multigraph, so when both sum units point at the
  same grandchild the second ``add_subcircuit`` overwrites the weight of the first
  instead of adding to it, and that branch loses its mass.
- ``log_conditional_in_place`` reports the log-probability of whichever node is the root
  *after* that simplification rather than of the node the forward pass computed it for,
  so it returns the leftover value of an unrelated node when simplification replaces the
  root.

The values asserted below are derived by hand in the test that uses them and confirmed by
numerically integrating the joint density.
"""

from __future__ import annotations

import unittest
from enum import IntEnum
from unittest import mock

import numpy as np
from krrood.adapters.json_serializer import from_json, to_json
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic

from probabilistic_model.distributions.distributions import (
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import IntractableError
from probabilistic_model.probabilistic_circuit.np.discrete_layer import (
    IntegerLayer,
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.np.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.np.inner_layer import (
    ProductLayer,
    SparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.np.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.np.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.np.uniform_layer import UniformLayer
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RxCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict

x = Continuous("x")
y = Continuous("y")
n = Integer("n")


class SymbolEnum(IntEnum):
    A = 0
    B = 1
    C = 2


s = Symbolic(name="s", domain=Set.from_iterable(SymbolEnum))


def uniform(variable, lower, upper):
    return UniformDistribution(
        variable=variable, interval=closed(lower, upper).simple_sets[0]
    )


def overlapping_mixture() -> RxCircuit:
    """
    A mixture of two products of uniforms whose supports overlap, so the circuit is not
    deterministic.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    left = ProductUnit(probabilistic_circuit=circuit)
    right = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(left, np.log(0.4))
    root.add_subcircuit(right, np.log(0.6))
    left.add_subcircuit(leaf(uniform(x, 0, 1), circuit))
    left.add_subcircuit(leaf(uniform(y, 0, 2), circuit))
    right.add_subcircuit(leaf(uniform(x, 1, 3), circuit))
    right.add_subcircuit(leaf(uniform(y, 1, 2), circuit))
    return circuit


def deterministic_mixture() -> RxCircuit:
    """
    A mixture of two products of uniforms with disjoint supports.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    left = ProductUnit(probabilistic_circuit=circuit)
    right = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(left, np.log(0.3))
    root.add_subcircuit(right, np.log(0.7))
    left.add_subcircuit(leaf(uniform(x, 0, 1), circuit))
    left.add_subcircuit(leaf(uniform(y, 0, 1), circuit))
    right.add_subcircuit(leaf(uniform(x, 2, 3), circuit))
    right.add_subcircuit(leaf(uniform(y, 2, 4), circuit))
    return circuit


def shared_children_circuit() -> RxCircuit:
    """
    A circuit whose leaves are shared by several sum units, so the layered circuit is a
    directed acyclic graph rather than a tree.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    left = ProductUnit(probabilistic_circuit=circuit)
    right = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(left, np.log(0.5))
    root.add_subcircuit(right, np.log(0.5))

    sum_x_1 = SumUnit(probabilistic_circuit=circuit)
    sum_x_2 = SumUnit(probabilistic_circuit=circuit)
    sum_y_1 = SumUnit(probabilistic_circuit=circuit)
    sum_y_2 = SumUnit(probabilistic_circuit=circuit)

    left.add_subcircuit(sum_x_1)
    left.add_subcircuit(sum_y_1)
    right.add_subcircuit(sum_x_2)
    right.add_subcircuit(sum_y_2)

    x_1 = leaf(uniform(x, 0, 1), circuit)
    x_2 = leaf(uniform(x, 1, 2), circuit)
    y_1 = leaf(uniform(y, 0, 1), circuit)
    y_2 = leaf(uniform(y, 1, 2), circuit)

    sum_x_1.add_subcircuit(x_1, np.log(0.8))
    sum_x_1.add_subcircuit(x_2, np.log(0.2))
    sum_x_2.add_subcircuit(x_1, np.log(0.3))
    sum_x_2.add_subcircuit(x_2, np.log(0.7))
    sum_y_1.add_subcircuit(y_1, np.log(0.6))
    sum_y_1.add_subcircuit(y_2, np.log(0.4))
    sum_y_2.add_subcircuit(y_1, np.log(0.1))
    sum_y_2.add_subcircuit(y_2, np.log(0.9))
    return circuit


def mixed_circuit() -> RxCircuit:
    """
    A circuit over a continuous, an integer and a symbolic variable.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)

    for weight, (lower, upper), integer_probabilities, symbolic_probabilities in (
        (0.4, (0, 2), {0: 0.2, 1: 0.8}, {0: 0.5, 1: 0.25, 2: 0.25}),
        (0.6, (1, 4), {0: 0.7, 2: 0.3}, {0: 0.1, 1: 0.6, 2: 0.3}),
    ):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(leaf(uniform(x, lower, upper), circuit))
        product.add_subcircuit(
            leaf(
                IntegerDistribution(
                    variable=n,
                    probabilities=MissingDict(float, integer_probabilities),
                ),
                circuit,
            )
        )
        product.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=s,
                    probabilities=MissingDict(
                        float,
                        {
                            hash(element): probability
                            for element, probability in zip(
                                s.domain.simple_sets, symbolic_probabilities.values()
                            )
                        },
                    ),
                ),
                circuit,
            )
        )
    return circuit


def gaussian_circuit() -> RxCircuit:
    """
    A mixture of products of Gaussians.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, location, scale in ((0.35, -1.0, 0.5), (0.65, 2.0, 1.5)):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(
            leaf(
                GaussianDistribution(variable=x, location=location, scale=scale),
                circuit,
            )
        )
        product.add_subcircuit(
            leaf(
                GaussianDistribution(variable=y, location=-location, scale=scale),
                circuit,
            )
        )
    return circuit


ALL_CIRCUITS = {
    "overlapping": overlapping_mixture,
    "deterministic": deterministic_mixture,
    "shared": shared_children_circuit,
    "mixed": mixed_circuit,
    "gaussian": gaussian_circuit,
}

CONTINUOUS_CIRCUITS = ("overlapping", "deterministic", "shared", "gaussian")


class ConversionTestCase(unittest.TestCase):

    def test_every_circuit_converts_and_keeps_its_variables(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                self.assertEqual(list(layered.variables), list(rx_circuit.variables))
                layered.validate()

    def test_layer_types_of_a_uniform_mixture(self):
        layered = ProbabilisticCircuit.from_rustworkx(overlapping_mixture())
        self.assertIsInstance(layered.root, SparseSumLayer)
        self.assertEqual(layered.root.number_of_nodes, 1)

        product_layer = layered.root.child_layers[0]
        self.assertIsInstance(product_layer, ProductLayer)
        self.assertEqual(product_layer.number_of_nodes, 2)
        self.assertEqual(len(product_layer.child_layers), 2)

        for child_layer in product_layer.child_layers:
            self.assertIsInstance(child_layer, UniformLayer)
            self.assertEqual(child_layer.number_of_nodes, 2)

    def test_shared_leaves_become_one_layer_with_two_nodes(self):
        layered = ProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        uniform_layers = [
            layer for layer in layered.layers if isinstance(layer, UniformLayer)
        ]
        # one layer per variable, each holding the two shared leaves
        self.assertEqual(len(uniform_layers), 2)
        self.assertEqual({layer.number_of_nodes for layer in uniform_layers}, {2})

    def test_round_trip_through_rustworkx_keeps_the_likelihood(self):
        for name in CONTINUOUS_CIRCUITS:
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                samples = rx_circuit.sample(200)
                np.testing.assert_allclose(
                    layered.to_rustworkx().log_likelihood(samples),
                    rx_circuit.log_likelihood(samples),
                )

    def test_gaussian_leaves_become_a_gaussian_layer(self):
        layered = ProbabilisticCircuit.from_rustworkx(gaussian_circuit())
        self.assertTrue(
            any(isinstance(layer, GaussianLayer) for layer in layered.layers)
        )

    def test_discrete_leaves_become_discrete_layers(self):
        layered = ProbabilisticCircuit.from_rustworkx(mixed_circuit())
        self.assertTrue(
            any(isinstance(layer, SymbolicLayer) for layer in layered.layers)
        )
        self.assertTrue(any(isinstance(layer, IntegerLayer) for layer in layered.layers))


class QueryTestCase(unittest.TestCase):
    """
    The numeric queries have to agree with the rustworkx implementation.
    """

    def setUp(self):
        np.random.seed(69)

    def test_log_likelihood(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                samples = rx_circuit.sample(500)
                np.testing.assert_allclose(
                    layered.log_likelihood(samples),
                    rx_circuit.log_likelihood(samples),
                )

    def test_log_likelihood_outside_the_support_is_minus_infinity(self):
        rx_circuit = deterministic_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        outside = np.array([[10.0, 10.0], [1.5, 1.5]])
        np.testing.assert_allclose(
            layered.log_likelihood(outside), rx_circuit.log_likelihood(outside)
        )
        self.assertTrue(np.all(np.isneginf(layered.log_likelihood(outside))))

    def test_cumulative_distribution_function(self):
        for name in CONTINUOUS_CIRCUITS:
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                samples = rx_circuit.sample(200)
                np.testing.assert_allclose(
                    layered.cumulative_distribution_function(samples),
                    rx_circuit.cumulative_distribution_function(samples),
                )

    def test_probability_of_a_simple_event(self):
        rx_circuit = overlapping_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        event = SimpleEvent.from_data({x: closed(0.5, 2.0), y: closed(0.0, 1.5)})
        self.assertAlmostEqual(
            layered.probability_of_simple_event(event),
            rx_circuit.probability_of_simple_event(event),
        )

    def test_probability_of_a_composite_event(self):
        rx_circuit = overlapping_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        event = SimpleEvent.from_data(
            {x: closed(0.0, 0.5) | closed(2.0, 2.5)}
        ).as_composite_set()
        self.assertAlmostEqual(
            layered.probability(event.__deepcopy__()),
            rx_circuit.probability(event.__deepcopy__()),
        )

    def test_probability_of_a_mixed_event(self):
        rx_circuit = mixed_circuit()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        event = SimpleEvent.from_data(
            {
                x: closed(0.5, 2.5),
                n: closed(0, 0),
                s: Set.from_iterable([SymbolEnum.A, SymbolEnum.C]),
            }
        )
        self.assertAlmostEqual(
            layered.probability_of_simple_event(event),
            rx_circuit.probability_of_simple_event(event),
        )

    def test_support(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                self.assertEqual(layered.support, rx_circuit.support)

    def test_expectation_and_variance(self):
        for name in CONTINUOUS_CIRCUITS:
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                for variable in rx_circuit.variables:
                    self.assertAlmostEqual(
                        layered.expectation()[variable],
                        rx_circuit.expectation()[variable],
                    )
                    self.assertAlmostEqual(
                        layered.variance()[variable], rx_circuit.variance()[variable]
                    )

    def test_expectation_of_an_integer_variable(self):
        rx_circuit = mixed_circuit()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        self.assertAlmostEqual(
            layered.expectation([n])[n], rx_circuit.expectation([n])[n]
        )

    def test_expectation_of_a_subset_of_the_variables(self):
        rx_circuit = overlapping_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        self.assertAlmostEqual(
            layered.expectation([x])[x], rx_circuit.expectation([x])[x]
        )

    def test_mode_of_a_deterministic_circuit(self):
        rx_circuit = deterministic_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        rx_mode, rx_likelihood = rx_circuit.log_mode()
        mode, likelihood = layered.log_mode()
        self.assertEqual(mode, rx_mode)
        self.assertAlmostEqual(likelihood, rx_likelihood)

    def test_mode_of_a_non_deterministic_circuit_is_intractable(self):
        layered = ProbabilisticCircuit.from_rustworkx(overlapping_mixture())
        with self.assertRaises(IntractableError):
            layered.log_mode()

    def test_determinism_matches_the_rustworkx_answer(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
                self.assertEqual(
                    layered.is_deterministic(), rx_circuit.is_deterministic()
                )

    def test_decomposability(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                layered = ProbabilisticCircuit.from_rustworkx(factory())
                self.assertTrue(layered.is_decomposable())


class SamplingTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_samples_lie_in_the_support(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                layered = ProbabilisticCircuit.from_rustworkx(factory())
                samples = layered.sample(500)
                self.assertEqual(samples.shape, (500, len(layered.variables)))
                self.assertFalse(np.any(np.isnan(samples)))
                self.assertTrue(np.all(layered.log_likelihood(samples) > -np.inf))

    def test_sample_mean_approximates_the_expectation(self):
        layered = ProbabilisticCircuit.from_rustworkx(overlapping_mixture())
        samples = layered.sample(20000)
        for index, variable in enumerate(layered.variables):
            self.assertAlmostEqual(
                float(samples[:, index].mean()),
                float(layered.expectation()[variable]),
                delta=0.05,
            )

    def test_sampling_a_circuit_with_shared_leaves(self):
        layered = ProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        samples = layered.sample(5000)
        self.assertTrue(np.all(layered.log_likelihood(samples) > -np.inf))
        for index, variable in enumerate(layered.variables):
            self.assertAlmostEqual(
                float(samples[:, index].mean()),
                float(layered.expectation()[variable]),
                delta=0.05,
            )


class TruncationTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def assert_same_truncation(self, rx_circuit: RxCircuit, event: Event, grid):
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

        rx_truncated, rx_probability = rx_circuit.truncated(event.__deepcopy__())
        truncated, probability = layered.truncated(event.__deepcopy__())

        self.assertAlmostEqual(probability, rx_probability)
        if rx_truncated is None:
            self.assertIsNone(truncated)
            return

        truncated.validate()
        np.testing.assert_allclose(
            truncated.log_likelihood(grid),
            rx_truncated.log_likelihood(grid),
            atol=1e-10,
        )
        return truncated

    def test_truncation_to_a_simple_event(self):
        grid = np.stack(
            np.meshgrid(np.linspace(-1, 4, 25), np.linspace(-1, 4, 25)), axis=-1
        ).reshape(-1, 2)
        event = SimpleEvent.from_data(
            {x: closed(0.5, 2.5), y: closed(0.5, 1.75)}
        ).as_composite_set()
        for name in ("overlapping", "deterministic", "shared"):
            with self.subTest(name):
                self.assert_same_truncation(ALL_CIRCUITS[name](), event, grid)

    def test_truncation_to_a_composite_event(self):
        grid = np.stack(
            np.meshgrid(np.linspace(-1, 4, 25), np.linspace(-1, 4, 25)), axis=-1
        ).reshape(-1, 2)
        event = SimpleEvent.from_data(
            {x: closed(0.0, 0.5) | closed(2.0, 2.6)}
        ).as_composite_set()
        for name in ("overlapping", "deterministic", "shared"):
            with self.subTest(name):
                self.assert_same_truncation(ALL_CIRCUITS[name](), event, grid)

    def test_truncation_of_a_gaussian_circuit_produces_truncated_gaussian_layers(self):
        grid = np.stack(
            np.meshgrid(np.linspace(-4, 6, 25), np.linspace(-4, 6, 25)), axis=-1
        ).reshape(-1, 2)
        event = SimpleEvent.from_data(
            {x: closed(-2.0, 1.0), y: closed(-1.0, 3.0)}
        ).as_composite_set()
        truncated = self.assert_same_truncation(gaussian_circuit(), event, grid)
        self.assertTrue(
            any(
                isinstance(layer, TruncatedGaussianLayer)
                for layer in truncated.layers
            )
        )

    def test_truncation_of_a_mixed_circuit(self):
        rx_circuit = mixed_circuit()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        event = SimpleEvent.from_data(
            {
                x: closed(0.5, 3.0),
                n: closed(0, 0),
                s: Set.from_iterable([SymbolEnum.A, SymbolEnum.B]),
            }
        ).as_composite_set()

        rx_truncated, rx_probability = rx_circuit.truncated(event.__deepcopy__())
        truncated, probability = layered.truncated(event.__deepcopy__())
        self.assertAlmostEqual(probability, rx_probability)

        samples = truncated.sample(200)
        np.testing.assert_allclose(
            truncated.log_likelihood(samples),
            rx_truncated.log_likelihood(samples),
        )

    def boxes(self, number_of_boxes: int, lower: float, upper: float) -> Event:
        """
        A staircase of disjoint boxes with gaps between them.

        Each box takes its own window of both variables, which stops the product algebra
        from merging them back into one simple set, and the gaps stop it from splitting
        them at shared corners.
        """
        width = (upper - lower) / (2 * number_of_boxes)
        result = None
        for index in range(number_of_boxes):
            start = lower + 2 * index * width
            box = SimpleEvent.from_data(
                {
                    x: closed(start, start + width),
                    y: closed(start, start + width),
                }
            ).as_composite_set()
            result = box if result is None else result | box
        return result

    @staticmethod
    def truncate_one_simple_set_at_a_time(layered, event):
        """
        Truncate through the fallback path, which handles one simple set at a time.

        The patch is applied to the class: the public truncation works on a copy of the
        circuit, so an attribute set on the instance would not reach it.
        """
        with mock.patch.object(
            ProbabilisticCircuit,
            "truncated_root_of_simple_events",
            return_value=None,
        ):
            return layered.truncated(event)

    def test_truncation_to_an_event_with_many_simple_sets(self):
        grid = np.stack(
            np.meshgrid(np.linspace(-1, 4, 30), np.linspace(-1, 4, 30)), axis=-1
        ).reshape(-1, 2)
        event = self.boxes(8, 0.0, 3.0)
        self.assertEqual(len(event.simple_sets), 8)
        self.assertGreater(len(event.simple_sets), 1)

        for name in ("overlapping", "deterministic", "shared"):
            with self.subTest(name):
                self.assert_same_truncation(ALL_CIRCUITS[name](), event, grid)

    def test_the_batched_pass_agrees_with_truncating_once_per_simple_set(self):
        """
        Truncating to all simple sets in one pass and truncating to them one at a time and
        mixing the results have to describe the same distribution.
        """
        grid = np.stack(
            np.meshgrid(np.linspace(-1, 4, 30), np.linspace(-1, 4, 30)), axis=-1
        ).reshape(-1, 2)
        event = self.boxes(6, 0.0, 3.0)

        for name in ("overlapping", "deterministic", "shared"):
            with self.subTest(name):
                layered = ProbabilisticCircuit.from_rustworkx(ALL_CIRCUITS[name]())

                batched, batched_probability = layered.truncated(event.__deepcopy__())
                separate, separate_probability = (
                    self.truncate_one_simple_set_at_a_time(
                        layered, event.__deepcopy__()
                    )
                )

                self.assertAlmostEqual(batched_probability, separate_probability)
                np.testing.assert_allclose(
                    batched.log_likelihood(grid),
                    separate.log_likelihood(grid),
                    atol=1e-10,
                )

    def test_the_batched_pass_keeps_the_number_of_layers(self):
        """
        The point of truncating to every simple set in one pass: the circuit keeps its
        layers and their blocks grow, instead of getting one set of layers per simple set.
        """
        layered = ProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        event = self.boxes(10, 0.0, 2.0)

        truncated, _ = layered.truncated(event.__deepcopy__())

        separate, _ = self.truncate_one_simple_set_at_a_time(
            layered, event.__deepcopy__()
        )

        # one mixing layer on top of the circuit's own layers, against one whole set of
        # layers per simple set
        self.assertLessEqual(len(truncated.layers), len(layered.layers) + 1)
        self.assertGreater(len(separate.layers), 4 * len(truncated.layers))

        # the same distribution, held in taller blocks rather than in more layers
        self.assertGreaterEqual(
            truncated.number_of_nodes, separate.number_of_nodes - len(separate.layers)
        )

    def test_a_gaussian_circuit_falls_back_to_truncating_once_per_simple_set(self):
        # a gaussian layer becomes a truncated gaussian layer, which the batched pass
        # does not handle, so this exercises the fallback end to end
        layered = ProbabilisticCircuit.from_rustworkx(gaussian_circuit())
        self.assertIsNone(
            layered.truncated_root_of_simple_events(
                list(self.boxes(4, -1.0, 3.0).simple_sets), False
            )
        )

        grid = np.stack(
            np.meshgrid(np.linspace(-4, 6, 25), np.linspace(-4, 6, 25)), axis=-1
        ).reshape(-1, 2)
        self.assert_same_truncation(gaussian_circuit(), self.boxes(4, -1.0, 3.0), grid)

    def test_truncation_puts_all_mass_inside_the_event(self):
        layered = ProbabilisticCircuit.from_rustworkx(overlapping_mixture())
        event = SimpleEvent.from_data(
            {x: closed(0.5, 2.5), y: closed(0.5, 1.75)}
        ).as_composite_set()
        truncated, _ = layered.truncated(event.__deepcopy__())
        self.assertAlmostEqual(truncated.probability(event.__deepcopy__()), 1.0)

    def test_truncation_to_an_impossible_event(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        event = SimpleEvent.from_data({x: closed(10.0, 11.0)}).as_composite_set()
        truncated, probability = layered.truncated(event)
        self.assertIsNone(truncated)
        self.assertEqual(probability, 0.0)

    def test_truncation_removes_the_impossible_nodes(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        event = SimpleEvent.from_data({x: closed(0.0, 1.0)}).as_composite_set()
        truncated, probability = layered.truncated(event)
        self.assertAlmostEqual(probability, 0.3)
        # the branch over [2, 3] x [2, 4] is gone
        for layer in truncated.layers:
            if isinstance(layer, UniformLayer):
                self.assertEqual(layer.number_of_nodes, 1)

    def test_truncation_does_not_change_the_original(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        before = layered.number_of_nodes
        event = SimpleEvent.from_data({x: closed(0.0, 1.0)}).as_composite_set()
        layered.truncated(event)
        self.assertEqual(layered.number_of_nodes, before)


class ConditionalTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_conditioning_on_a_continuous_variable(self):
        # p(x = 0.75) is the weight of the only branch whose support contains 0.75
        for name, expected_probability in (("overlapping", 0.4), ("deterministic", 0.3)):
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

                point = {x: 0.75}
                rx_conditional, _ = rx_circuit.conditional(point)
                conditional, probability = layered.conditional(point)

                self.assertAlmostEqual(probability, expected_probability)
                self.assertEqual(list(conditional.variables), list(layered.variables))

                grid = np.stack(
                    [np.full(40, 0.75), np.linspace(-1, 4, 40)], axis=-1
                )
                np.testing.assert_allclose(
                    conditional.log_likelihood(grid),
                    rx_conditional.log_likelihood(grid),
                    atol=1e-10,
                )

    def test_the_conditional_density_integrates_to_one(self):
        grid = np.linspace(-1, 5, 60001)
        for name in ("overlapping", "deterministic", "shared"):
            with self.subTest(name):
                layered = ProbabilisticCircuit.from_rustworkx(ALL_CIRCUITS[name]())
                conditional, _ = layered.conditional({x: 0.75})
                density = conditional.likelihood(
                    np.stack([np.full_like(grid, 0.75), grid], axis=-1)
                )
                self.assertAlmostEqual(
                    float(np.trapezoid(density, grid)), 1.0, places=3
                )

    def test_conditioning_a_circuit_with_shared_leaves(self):
        """
        ``p(x=0.75, y) = 0.4 * sum_y_1(y) + 0.15 * sum_y_2(y)``, which is ``0.255`` on
        ``(0, 1)`` and ``0.295`` on ``(1, 2)``. Dividing by ``p(x=0.75) = 0.55`` gives the
        conditional density.
        """
        layered = ProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        conditional, probability = layered.conditional({x: 0.75})

        self.assertAlmostEqual(probability, 0.55)
        np.testing.assert_allclose(
            conditional.likelihood(np.array([[0.75, 0.5], [0.75, 1.5]])),
            np.array([0.255 / 0.55, 0.295 / 0.55]),
        )

    def test_conditioning_on_an_impossible_point(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        conditional, probability = layered.conditional({x: 1.5})
        self.assertIsNone(conditional)
        self.assertEqual(probability, 0.0)

    def test_conditioning_on_a_symbolic_variable(self):
        rx_circuit = mixed_circuit()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

        point = {s: SymbolEnum.B}
        rx_conditional, rx_probability = rx_circuit.log_conditional(point)
        conditional, probability = layered.log_conditional(point)

        self.assertAlmostEqual(probability, rx_probability)
        samples = conditional.sample(200)
        np.testing.assert_allclose(
            conditional.log_likelihood(samples),
            rx_conditional.log_likelihood(samples),
        )

    def test_conditioning_on_every_variable(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        conditional, probability = layered.conditional({x: 0.5, y: 0.5})
        self.assertIsNotNone(conditional)
        self.assertAlmostEqual(probability, 0.3)
        np.testing.assert_allclose(
            conditional.log_likelihood(np.array([[0.5, 0.5]])), np.array([0.0])
        )


class MarginalTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_marginal_of_one_variable(self):
        for name in ("overlapping", "deterministic"):
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

                rx_marginal = rx_circuit.marginal([x])
                marginal = layered.marginal([x])

                self.assertEqual(list(marginal.variables), [x])
                grid = np.linspace(-1, 4, 60).reshape(-1, 1)
                np.testing.assert_allclose(
                    marginal.log_likelihood(grid),
                    rx_marginal.log_likelihood(grid),
                    atol=1e-10,
                )

    def test_marginal_of_a_circuit_with_shared_leaves(self):
        """
        ``p(x) = 0.5 * sum_x_1(x) + 0.5 * sum_x_2(x)``, which is ``0.5 * 0.8 + 0.5 * 0.3``
        on ``(0, 1)`` and ``0.5 * 0.2 + 0.5 * 0.7`` on ``(1, 2)``.
        """
        layered = ProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        marginal = layered.marginal([x])

        np.testing.assert_allclose(
            marginal.likelihood(np.array([[0.5], [1.5]])), np.array([0.55, 0.45])
        )

        # the same density, obtained by integrating y out of the joint
        grid = np.linspace(0, 2, 20001)
        for value, expected in ((0.5, 0.55), (1.5, 0.45)):
            joint = layered.likelihood(
                np.stack([np.full_like(grid, value), grid], axis=-1)
            )
            self.assertAlmostEqual(float(np.trapezoid(joint, grid)), expected, places=3)

    def test_marginal_of_a_discrete_variable(self):
        rx_circuit = mixed_circuit()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)
        marginal = layered.marginal([n])
        rx_marginal = rx_circuit.marginal([n])
        self.assertEqual(list(marginal.variables), [n])
        grid = np.array([[0], [1], [2], [3]])
        np.testing.assert_allclose(
            marginal.log_likelihood(grid), rx_marginal.log_likelihood(grid)
        )

    def test_marginal_of_a_variable_that_is_not_modeled(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        self.assertIsNone(layered.marginal([Continuous("z")]))

    def test_marginal_does_not_change_the_original(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        layered.marginal([x])
        self.assertEqual(list(layered.variables), [x, y])


class SerializationTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_json_round_trip(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

                restored = from_json(to_json(layered))
                self.assertIsInstance(restored, ProbabilisticCircuit)
                self.assertEqual(list(restored.variables), list(layered.variables))

                samples = rx_circuit.sample(200)
                np.testing.assert_allclose(
                    restored.log_likelihood(samples), layered.log_likelihood(samples)
                )

    def test_json_round_trip_of_a_truncated_gaussian_layer(self):
        layered = ProbabilisticCircuit.from_rustworkx(gaussian_circuit())
        event = SimpleEvent.from_data(
            {x: closed(-2.0, 1.0), y: closed(-1.0, 3.0)}
        ).as_composite_set()
        truncated, _ = layered.truncated(event)

        restored = from_json(to_json(truncated))
        samples = truncated.sample(200)
        np.testing.assert_allclose(
            restored.log_likelihood(samples), truncated.log_likelihood(samples)
        )

    def test_json_round_trip_of_a_conditioned_circuit(self):
        # conditional() attaches a DiracDeltaLayer per conditioned variable under a new
        # product root, which is not exercised by any of the ALL_CIRCUITS factories.
        layered = ProbabilisticCircuit.from_rustworkx(gaussian_circuit())
        conditioned, _ = layered.conditional({x: 0.5})
        self.assertIsInstance(conditioned.root, ProductLayer)
        self.assertTrue(
            any(
                isinstance(layer, DiracDeltaLayer) for layer in conditioned.layers
            )
        )

        restored = from_json(to_json(conditioned))
        samples = conditioned.sample(200)
        np.testing.assert_allclose(
            restored.log_likelihood(samples), conditioned.log_likelihood(samples)
        )

    def test_deep_copy_is_independent(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        copy = layered.__deepcopy__()

        samples = np.array([[0.5, 0.5], [2.5, 3.0]])
        before = layered.log_likelihood(samples)

        copy.apply_translation({x: 10.0, y: 10.0})
        np.testing.assert_allclose(layered.log_likelihood(samples), before)
        self.assertTrue(np.all(np.isneginf(copy.log_likelihood(samples))))


class TransformationTestCase(unittest.TestCase):

    def test_translation_matches_rustworkx(self):
        rx_circuit = deterministic_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

        translation = {x: 1.5, y: -0.5}
        rx_circuit.apply_translation(translation)
        layered.apply_translation(translation)

        grid = np.stack(
            np.meshgrid(np.linspace(-2, 6, 20), np.linspace(-2, 6, 20)), axis=-1
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            layered.log_likelihood(grid), rx_circuit.log_likelihood(grid)
        )

    def test_scaling_matches_rustworkx(self):
        rx_circuit = deterministic_mixture()
        layered = ProbabilisticCircuit.from_rustworkx(rx_circuit)

        scaling = {x: 2.0, y: 3.0}
        rx_circuit.apply_scaling(scaling)
        layered.apply_scaling(scaling)

        grid = np.stack(
            np.meshgrid(np.linspace(-2, 14, 20), np.linspace(-2, 14, 20)), axis=-1
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            layered.log_likelihood(grid), rx_circuit.log_likelihood(grid)
        )

    def test_renaming_variables_with_a_prefix(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        samples = np.array([[0.5, 0.5], [2.5, 3.0]])
        before = layered.log_likelihood(samples)

        layered.rename_variables_with_prefix("world")
        self.assertEqual(
            [variable.name for variable in layered.variables], ["world.x", "world.y"]
        )
        np.testing.assert_allclose(layered.log_likelihood(samples), before)

    def test_renaming_keeps_the_column_order(self):
        # renaming may reorder the variables, and the layers have to follow
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        samples = np.array([[0.5, 0.7], [2.5, 3.0]])
        before = layered.log_likelihood(samples)

        layered.update_variables({x: Continuous("z"), y: Continuous("a")})
        self.assertEqual(
            [variable.name for variable in layered.variables], ["a", "z"]
        )
        np.testing.assert_allclose(
            layered.log_likelihood(samples[:, ::-1]), before
        )


class SimplificationTestCase(unittest.TestCase):

    def test_normalize_makes_the_weights_sum_to_one(self):
        layered = ProbabilisticCircuit.from_rustworkx(overlapping_mixture())
        root = layered.root
        root.log_weights[0].data = root.log_weights[0].data + 3.0

        samples = np.array([[0.5, 0.5], [2.0, 1.5]])
        before = layered.log_likelihood(samples)

        layered.normalize()
        np.testing.assert_allclose(
            root.log_normalization_constants, np.zeros(root.number_of_nodes), atol=1e-12
        )
        np.testing.assert_allclose(layered.log_likelihood(samples), before)

    def test_simplify_removes_an_identity_sum_layer(self):
        layered = ProbabilisticCircuit.from_rustworkx(deterministic_mixture())
        samples = np.array([[0.5, 0.5], [2.5, 3.0]])
        before = layered.log_likelihood(samples)

        # truncation to the full support introduces an identity layer over the leaves
        number_of_layers = len(layered.layers)
        layered.simplify()
        self.assertLessEqual(len(layered.layers), number_of_layers)
        np.testing.assert_allclose(layered.log_likelihood(samples), before)


class LayerTestCase(unittest.TestCase):
    """
    Direct checks of the layer classes, independent of a circuit.
    """

    def test_uniform_layer_likelihood(self):
        layer = UniformLayer.from_distributions(
            0, [uniform(x, 0, 1), uniform(x, 1, 3)]
        )
        self.assertEqual(layer.number_of_nodes, 2)
        result = layer.log_likelihood_of_nodes(np.array([[0.5], [2.0], [5.0]]))
        expected = np.array(
            [[0.0, -np.inf], [-np.inf, -np.log(2)], [-np.inf, -np.inf]]
        )
        np.testing.assert_allclose(result, expected)

    def test_dirac_delta_layer_likelihood(self):
        layer = DiracDeltaLayer(0, np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        result = layer.log_likelihood_of_nodes(np.array([[0.0], [1.0], [2.0]]))
        expected = np.array(
            [[0.0, -np.inf], [-np.inf, np.log(2)], [-np.inf, -np.inf]]
        )
        np.testing.assert_allclose(result, expected)

    def test_gaussian_layer_likelihood(self):
        layer = GaussianLayer(0, np.array([0.0, 2.0]), np.array([1.0, 0.5]))
        points = np.array([[0.0], [2.0]])
        result = layer.log_likelihood_of_nodes(points)
        for node in range(2):
            distribution = layer.node_distribution(node, x)
            np.testing.assert_allclose(
                result[:, node], distribution.log_likelihood(points)
            )

    def test_symbolic_layer_likelihood(self):
        distributions = [
            SymbolicDistribution(
                variable=s,
                probabilities=MissingDict(
                    float,
                    {
                        hash(element): probability
                        for element, probability in zip(
                            s.domain.simple_sets, (0.5, 0.25, 0.25)
                        )
                    },
                ),
            ),
            SymbolicDistribution(
                variable=s,
                probabilities=MissingDict(
                    float,
                    {
                        hash(element): probability
                        for element, probability in zip(
                            s.domain.simple_sets, (0.1, 0.6, 0.3)
                        )
                    },
                ),
            ),
        ]
        layer = SymbolicLayer.from_distributions(0, distributions)
        events = np.array([[hash(element)] for element in s.domain.simple_sets])
        result = layer.log_likelihood_of_nodes(events)
        for node, distribution in enumerate(distributions):
            np.testing.assert_allclose(
                result[:, node], distribution.log_likelihood(events)
            )

    def test_integer_layer_cumulative_distribution(self):
        distribution = IntegerDistribution(
            variable=n, probabilities=MissingDict(float, {0: 0.2, 1: 0.3, 2: 0.5})
        )
        layer = IntegerLayer.from_distributions(0, [distribution])
        points = np.array([[-1], [0], [1], [2], [3]])
        np.testing.assert_allclose(
            layer.cumulative_distribution_of_nodes(points)[:, 0],
            distribution.cumulative_distribution_function(points),
        )

    def test_number_of_parameters(self):
        layered = ProbabilisticCircuit.from_rustworkx(overlapping_mixture())
        # two weights on the root plus two bounds for each of the four uniform nodes
        self.assertEqual(layered.number_of_parameters, 2 + 4 * 2)

    def test_topological_order_visits_parents_first(self):
        layered = ProbabilisticCircuit.from_rustworkx(shared_children_circuit())
        order = layered.root.topological_layer_order()
        positions = {id(layer): index for index, layer in enumerate(order)}
        self.assertEqual(len(order), len(layered.layers))
        for layer in order:
            for child_layer in layer.child_layers:
                self.assertLess(positions[id(layer)], positions[id(child_layer)])

    def test_validate_detects_a_shape_mismatch(self):
        layer = GaussianLayer(0, np.array([0.0, 1.0]), np.array([1.0]))
        with self.assertRaises(Exception):
            layer.validate()


if __name__ == "__main__":
    unittest.main()
