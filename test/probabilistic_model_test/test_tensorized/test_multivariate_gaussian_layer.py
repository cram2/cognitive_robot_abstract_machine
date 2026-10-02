"""
Cross validation of the multivariate Gaussian layers against the multivariate leaves of
the ``rx`` package.

The probabilities of boxes under a correlated Gaussian are integrated numerically by
:mod:`scipy.stats.multivariate_normal`, whose quasi Monte Carlo estimates differ from call
to call by about ``1e-5``, so every comparison that involves one allows for that.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import pandas as pd
from krrood.adapters.json_serializer import from_json, to_json
from random_events.interval import closed, open as open_interval, reals, singleton
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.variable import Continuous

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.distributions.truncated_multivariate_gaussian import (
    MomentIntegration,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_model import (
    GaussianMixtureModel,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RxCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.exceptions import NoClosedFormError
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.multivariate_gaussian_layer import (
    MultivariateGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.truncated_multivariate_gaussian_layer import (
    TruncatedMultivariateGaussianLayer,
)

x = Continuous("x")
y = Continuous("y")
z = Continuous("z")

NUMERICAL_INTEGRATION_TOLERANCE = 1e-4
"""
The absolute error allowed for a probability that was integrated numerically.
"""

# %% circuits


def first_component() -> MultivariateGaussianDistribution:
    """
    A correlated Gaussian whose variables are given in the reverse of the order the
    circuit sorts them in, so that a conversion that confused the two orders would swap
    them.
    """
    return MultivariateGaussianDistribution(
        variables=(y, x),
        mean=np.array([1.0, -0.5]),
        covariance=Covariance.from_matrix([[1.0, 0.6], [0.6, 2.0]]),
    )


def second_component() -> MultivariateGaussianDistribution:
    """
    A Gaussian that is negatively correlated, with its variables in the circuit's order.
    """
    return MultivariateGaussianDistribution(
        variables=(x, y),
        mean=np.array([2.0, 1.5]),
        covariance=Covariance.from_matrix([[0.5, -0.2], [-0.2, 0.8]]),
    )


def gaussian_mixture() -> RxCircuit:
    """
    A mixture of two correlated Gaussians over ``x`` and ``y``.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(leaf(first_component(), circuit), np.log(0.3))
    root.add_subcircuit(leaf(second_component(), circuit), np.log(0.7))
    return circuit


def single_gaussian() -> RxCircuit:
    """
    The first component on its own, which is deterministic.
    """
    circuit = RxCircuit()
    leaf(first_component(), circuit)
    return circuit


def gaussian_mixture_times_uniform() -> RxCircuit:
    """
    A mixture of products of a correlated Gaussian over ``x`` and ``y`` and a uniform
    distribution over ``z``.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, component, (lower, upper) in (
        (0.4, first_component(), (0.0, 1.0)),
        (0.6, second_component(), (0.5, 3.0)),
    ):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(leaf(component, circuit))
        product.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=z, interval=closed(lower, upper).simple_sets[0]
                ),
                circuit,
            )
        )
    return circuit


def three_variable_gaussian() -> RxCircuit:
    """
    One Gaussian over three correlated variables.
    """
    circuit = RxCircuit()
    leaf(
        MultivariateGaussianDistribution(
            variables=(z, x, y),
            mean=np.array([0.5, -1.0, 2.0]),
            covariance=Covariance.from_matrix(
                [[1.0, 0.3, -0.2], [0.3, 1.5, 0.4], [-0.2, 0.4, 0.7]]
            ),
        ),
        circuit,
    )
    return circuit


ALL_CIRCUITS = {
    "mixture": gaussian_mixture,
    "single": single_gaussian,
    "mixture times uniform": gaussian_mixture_times_uniform,
    "three variables": three_variable_gaussian,
}


def box(**intervals) -> SimpleEvent:
    """
    :param intervals: The interval of every constrained variable, by name.
    :return: The box.
    """
    variables = {variable.name: variable for variable in (x, y, z)}
    return SimpleEvent.from_data(
        {variables[name]: interval for name, interval in intervals.items()}
    )


# %% conversion


class MultivariateGaussianConversionTestCase(unittest.TestCase):

    def test_multivariate_gaussian_leaves_become_one_layer(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        [gaussian_layer] = layered.root.child_layers
        self.assertIsInstance(gaussian_layer, MultivariateGaussianLayer)
        self.assertEqual(gaussian_layer.number_of_nodes, 2)
        np.testing.assert_array_equal(gaussian_layer.variables, [0, 1])

    def test_the_parameters_are_laid_out_by_the_variables_of_the_circuit(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(single_gaussian())
        gaussian_layer = layered.root
        component = first_component()
        in_circuit_order = [component.index_of(x), component.index_of(y)]
        np.testing.assert_allclose(
            gaussian_layer.mean[0], component.mean[in_circuit_order]
        )
        np.testing.assert_allclose(
            gaussian_layer.covariance.matrices[0],
            component.covariance.matrix[np.ix_(in_circuit_order, in_circuit_order)],
        )

    def test_round_trip_through_rustworkx_keeps_the_likelihood(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                samples = rx_circuit.sample(200)
                np.testing.assert_allclose(
                    LayeredCircuitToRustworkxCircuitConverter.convert(
                        layered
                    ).log_likelihood(samples),
                    rx_circuit.log_likelihood(samples),
                )

    def test_a_learned_gaussian_mixture_converts(self):
        generator = np.random.default_rng(0)
        data = pd.DataFrame(
            np.concatenate(
                [
                    generator.multivariate_normal(
                        [0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=200
                    ),
                    generator.multivariate_normal(
                        [6.0, -4.0], [[0.5, -0.2], [-0.2, 2.0]], size=100
                    ),
                ]
            ),
            columns=["y", "x"],
        )
        rx_circuit = GaussianMixtureModel().fit(data)
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        samples = rx_circuit.sample(300)
        np.testing.assert_allclose(
            layered.log_likelihood(samples), rx_circuit.log_likelihood(samples)
        )


# %% queries that leave the circuit as it is


class MultivariateGaussianQueryTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_log_likelihood(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                samples = rx_circuit.sample(500)
                np.testing.assert_allclose(
                    layered.log_likelihood(samples),
                    rx_circuit.log_likelihood(samples),
                )

    def test_cumulative_distribution_function(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                samples = rx_circuit.sample(20)
                np.testing.assert_allclose(
                    layered.cumulative_distribution_function(samples),
                    rx_circuit.cumulative_distribution_function(samples),
                    atol=NUMERICAL_INTEGRATION_TOLERANCE,
                )

    def test_probability_of_a_box(self):
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5), z=closed(0.2, 2.0))
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                event.fill_missing_variables(set(rx_circuit.variables))
                self.assertAlmostEqual(
                    layered.probability_of_simple_event(event),
                    rx_circuit.probability_of_simple_event(event),
                    delta=NUMERICAL_INTEGRATION_TOLERANCE,
                )

    def test_probability_of_a_box_that_bounds_only_one_variable(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(x=closed(0.0, 1.0), y=reals())
        self.assertAlmostEqual(
            layered.probability_of_simple_event(event),
            rx_circuit.probability_of_simple_event(event),
            delta=NUMERICAL_INTEGRATION_TOLERANCE,
        )

    def test_probability_of_several_intervals_per_variable(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(x=closed(-2.0, 0.0) | closed(1.0, 3.0), y=closed(-1.0, 2.0))
        self.assertAlmostEqual(
            layered.probability_of_simple_event(event),
            rx_circuit.probability_of_simple_event(event),
            delta=NUMERICAL_INTEGRATION_TOLERANCE,
        )

    def test_a_singleton_has_no_probability(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        self.assertEqual(
            layered.probability_of_simple_event(box(x=singleton(0.0), y=reals())), 0.0
        )

    def test_moment(self):
        for name, factory in ALL_CIRCUITS.items():
            with self.subTest(name):
                rx_circuit = factory()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                expectation = rx_circuit.expectation(rx_circuit.variables)
                order = VariableMap({variable: 2 for variable in rx_circuit.variables})
                expected = rx_circuit.moment(order, expectation)
                moment = layered.moment(order, expectation)
                for variable in rx_circuit.variables:
                    self.assertAlmostEqual(moment[variable], expected[variable])

    def test_expectation_next_to_another_leaf(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(
            gaussian_mixture_times_uniform()
        )
        expectation = layered.expectation([x, z])
        first, second = first_component(), second_component()
        self.assertAlmostEqual(
            expectation[x],
            0.4 * first.mean[first.index_of(x)] + 0.6 * second.mean[second.index_of(x)],
        )
        # the means of the uniform distributions over [0, 1] and [0.5, 3]
        self.assertAlmostEqual(expectation[z], 0.4 * 0.5 + 0.6 * 1.75)

    def test_marginal(self):
        rx_circuit = three_variable_gaussian()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        samples = rx_circuit.sample(100)
        for kept in ([x], [z], [x, y], [y, z]):
            with self.subTest([variable.name for variable in kept]):
                rx_marginal = rx_circuit.marginal(kept)
                layered_marginal = layered.marginal(kept)
                columns = [rx_circuit.variables.index(variable) for variable in kept]
                np.testing.assert_allclose(
                    layered_marginal.log_likelihood(samples[:, columns]),
                    rx_marginal.log_likelihood(samples[:, columns]),
                )

    def test_mode_is_the_mean(self):
        rx_circuit = single_gaussian()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        mode, log_likelihood = layered.log_mode()
        expected_mode, expected_log_likelihood = rx_circuit.log_mode()
        self.assertEqual(mode, expected_mode)
        self.assertAlmostEqual(log_likelihood, expected_log_likelihood)

    def test_samples_have_the_mean_of_the_circuit(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        expectation = rx_circuit.expectation(rx_circuit.variables)
        np.testing.assert_allclose(
            layered.sample(20000).mean(axis=0),
            [expectation[variable] for variable in rx_circuit.variables],
            atol=0.05,
        )

    def test_samples_have_the_covariance_of_the_gaussian(self):
        component = first_component()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(single_gaussian())
        in_circuit_order = [component.index_of(x), component.index_of(y)]
        np.testing.assert_allclose(
            np.cov(layered.sample(20000).T),
            component.covariance.matrix[np.ix_(in_circuit_order, in_circuit_order)],
            atol=0.05,
        )

    def test_the_support_is_the_whole_space(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        self.assertEqual(layered.support, rx_circuit.support)


# %% truncation


class MultivariateGaussianTruncationTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_truncation_to_a_box(self):
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        for name in ("mixture", "single", "mixture times uniform"):
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                rx_truncated, rx_probability = rx_circuit.truncated(
                    event.__deepcopy__()
                )
                truncated, probability = layered.truncated(event.__deepcopy__())
                self.assertAlmostEqual(
                    probability, rx_probability, delta=NUMERICAL_INTEGRATION_TOLERANCE
                )
                self.assertTrue(
                    any(
                        isinstance(layer, TruncatedMultivariateGaussianLayer)
                        for layer in truncated.layers
                    )
                )
                samples = rx_truncated.sample(200)
                np.testing.assert_allclose(
                    truncated.log_likelihood(samples),
                    rx_truncated.log_likelihood(samples),
                    atol=1e-3,
                )

    def test_likelihood_outside_the_box_is_minus_infinity(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, _ = layered.truncated(
            box(x=closed(-1.0, 2.0), y=open_interval(0.0, 1.5)).as_composite_set()
        )
        outside = np.array([[3.0, 1.0], [0.0, 1.5], [0.0, 0.0]])
        self.assertTrue(np.all(np.isneginf(truncated.log_likelihood(outside))))

    def test_truncation_to_the_whole_space_keeps_the_gaussians(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, probability = layered.truncated(
            box(x=reals(), y=reals()).as_composite_set()
        )
        np.testing.assert_allclose(probability, 1.0)
        self.assertTrue(
            any(
                isinstance(layer, MultivariateGaussianLayer)
                for layer in truncated.layers
            )
        )

    def test_truncation_to_an_impossible_event(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, probability = layered.truncated(
            box(x=singleton(0.0), y=reals()).as_composite_set()
        )
        self.assertIsNone(truncated)
        self.assertEqual(probability, 0.0)

    def test_truncation_to_several_intervals_of_one_variable(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(
            x=closed(-2.0, 0.0) | closed(1.0, 3.0), y=closed(-1.0, 2.0)
        ).as_composite_set()
        rx_truncated, rx_probability = rx_circuit.truncated(event.__deepcopy__())
        truncated, probability = layered.truncated(event.__deepcopy__())
        self.assertAlmostEqual(
            probability, rx_probability, delta=NUMERICAL_INTEGRATION_TOLERANCE
        )
        samples = rx_truncated.sample(200)
        np.testing.assert_allclose(
            truncated.log_likelihood(samples),
            rx_truncated.log_likelihood(samples),
            atol=1e-3,
        )

    def test_truncation_to_several_boxes_in_one_batch(self):
        rx_circuit = gaussian_mixture_times_uniform()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = (
            box(x=closed(-1.0, 0.0), y=closed(-1.0, 0.0), z=closed(0.0, 1.0))
            .as_composite_set()
            .union_with(
                box(
                    x=closed(1.0, 2.0), y=closed(1.0, 2.0), z=closed(0.5, 2.5)
                ).as_composite_set()
            )
        )
        simple_events = list(event.__deepcopy__().simple_sets)
        self.assertTrue(layered.can_truncate_in_one_batch(simple_events))

        rx_truncated, rx_probability = rx_circuit.truncated(event.__deepcopy__())
        truncated, probability = layered.truncated(event.__deepcopy__())
        self.assertAlmostEqual(
            probability, rx_probability, delta=NUMERICAL_INTEGRATION_TOLERANCE
        )
        samples = rx_truncated.sample(200)
        np.testing.assert_allclose(
            truncated.log_likelihood(samples),
            rx_truncated.log_likelihood(samples),
            atol=1e-3,
        )

    def test_truncating_twice_intersects_the_boxes(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        first = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        second = box(x=closed(0.0, 3.0), y=closed(-1.0, 1.0)).as_composite_set()

        rx_truncated, _ = rx_circuit.truncated(first.__deepcopy__())
        rx_twice, rx_probability = rx_truncated.truncated(second.__deepcopy__())
        truncated, _ = layered.truncated(first.__deepcopy__())
        twice, probability = truncated.truncated(second.__deepcopy__())

        self.assertAlmostEqual(
            probability, rx_probability, delta=NUMERICAL_INTEGRATION_TOLERANCE
        )
        self.assertEqual(twice.support, rx_twice.support)

    def test_queries_of_a_truncated_circuit(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        rx_truncated, _ = rx_circuit.truncated(event.__deepcopy__())
        truncated, _ = layered.truncated(event.__deepcopy__())

        points = np.array([[0.5, 0.5], [1.5, 1.0], [-2.0, 1.0], [3.0, 3.0]])
        np.testing.assert_allclose(
            truncated.cumulative_distribution_function(points),
            rx_truncated.cumulative_distribution_function(points),
            atol=NUMERICAL_INTEGRATION_TOLERANCE,
        )
        inner = box(x=closed(0.0, 1.0), y=closed(0.5, 3.0))
        self.assertAlmostEqual(
            truncated.probability_of_simple_event(inner),
            rx_truncated.probability_of_simple_event(inner),
            delta=NUMERICAL_INTEGRATION_TOLERANCE,
        )
        self.assertEqual(truncated.support, rx_truncated.support)

    def test_mode_of_a_truncated_gaussian(self):
        rx_circuit = single_gaussian()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        # the box excludes the mean, so the mode lies on its boundary
        event = box(x=closed(1.0, 2.0), y=closed(-1.0, 0.0)).as_composite_set()
        rx_truncated, _ = rx_circuit.truncated(event.__deepcopy__())
        truncated, _ = layered.truncated(event.__deepcopy__())
        mode, log_likelihood = truncated.log_mode()
        expected_mode, expected_log_likelihood = rx_truncated.log_mode()
        self.assertEqual(mode, expected_mode)
        self.assertAlmostEqual(log_likelihood, expected_log_likelihood, delta=1e-3)

    def test_samples_of_a_truncated_circuit_lie_in_the_box(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        rx_truncated, _ = rx_circuit.truncated(event.__deepcopy__())
        truncated, _ = layered.truncated(event.__deepcopy__())

        samples = truncated.sample(5000)
        self.assertEqual(samples.shape, (5000, 2))
        self.assertTrue(np.all(np.isfinite(truncated.log_likelihood(samples))))
        np.testing.assert_allclose(
            samples.mean(axis=0), rx_truncated.sample(5000).mean(axis=0), atol=0.05
        )

    def test_a_truncated_gaussian_has_no_marginal_over_some_of_its_variables(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, _ = layered.truncated(
            box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        )
        with self.assertRaises(NoClosedFormError):
            truncated.marginal([x])

    def test_moments_of_a_truncated_gaussian_agree_with_rustworkx(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        rx_truncated, _ = rx_circuit.truncated(event.__deepcopy__())
        truncated, _ = layered.truncated(event.__deepcopy__())

        expectation = truncated.expectation([x, y])
        variance = truncated.variance([x, y])
        expected_expectation = rx_truncated.expectation([x, y])
        expected_variance = rx_truncated.variance([x, y])
        for variable in (x, y):
            self.assertAlmostEqual(
                expectation[variable], expected_expectation[variable]
            )
            self.assertAlmostEqual(variance[variable], expected_variance[variable])

    def test_the_expectation_of_a_truncated_gaussian_is_the_mean_of_its_samples(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, _ = layered.truncated(
            box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        )
        expectation = truncated.expectation([x, y])
        np.testing.assert_allclose(
            [expectation[x], expectation[y]],
            truncated.sample(20000).mean(axis=0),
            atol=0.03,
        )

    def test_the_way_to_integrate_moments_follows_the_distributions_through_the_layer(
        self,
    ):
        rx_circuit = single_gaussian()
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        rx_truncated, _ = rx_circuit.truncated(event)
        [leaf] = rx_truncated.leaves
        moment_integration = MomentIntegration(
            deviations_integrated_over=6.0, panels=8, nodes_per_panel=12
        )
        leaf.distribution = dataclasses.replace(
            leaf.distribution, moment_integration=moment_integration
        )
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_truncated)
        further, _ = layered.truncated(
            box(x=closed(-0.5, 1.0), y=closed(0.0, 1.5)).as_composite_set()
        )
        [restored] = LayeredCircuitToRustworkxCircuitConverter.convert(further).leaves
        self.assertEqual(restored.distribution.moment_integration, moment_integration)

    def test_a_higher_moment_of_a_truncated_gaussian_agrees_with_rustworkx(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        rx_truncated, _ = rx_circuit.truncated(event.__deepcopy__())
        truncated, _ = layered.truncated(event.__deepcopy__())
        order = VariableMap({x: 3, y: 4})
        center = VariableMap({x: 0.5, y: 0.0})
        moment = truncated.moment(order, center)
        expected = rx_truncated.moment(order, center)
        for variable in (x, y):
            self.assertAlmostEqual(moment[variable], expected[variable])


# %% conditioning


class MultivariateGaussianConditioningTestCase(unittest.TestCase):

    def test_conditioning_on_some_of_the_variables(self):
        for name in ("mixture", "mixture times uniform", "three variables"):
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                point = {x: 0.5}
                rx_conditioned, rx_log_probability = rx_circuit.log_conditional(point)
                conditioned, log_probability = layered.log_conditional(point)
                self.assertAlmostEqual(log_probability, rx_log_probability)

                free = [variable for variable in rx_circuit.variables if variable != x]
                samples = rx_conditioned.marginal(free).sample(200)
                np.testing.assert_allclose(
                    conditioned.marginal(free).log_likelihood(samples),
                    rx_conditioned.marginal(free).log_likelihood(samples),
                )

    def test_conditioning_on_every_variable_is_the_likelihood(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        conditioned, log_probability = layered.log_conditional({x: 0.5, y: -0.25})
        self.assertAlmostEqual(
            log_probability, float(layered.log_likelihood(np.array([[0.5, -0.25]]))[0])
        )
        self.assertEqual(conditioned.sample(3).tolist(), [[0.5, -0.25]] * 3)

    def test_conditioning_a_truncated_circuit_agrees_with_rustworkx(self):
        event = box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        point = {x: 0.5}
        values = np.array([[0.1], [0.7], [1.4]])
        for name in ("single", "mixture"):
            with self.subTest(name):
                rx_circuit = ALL_CIRCUITS[name]()
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
                rx_truncated, _ = rx_circuit.truncated(event.__deepcopy__())
                truncated, _ = layered.truncated(event.__deepcopy__())

                expected, expected_log_probability = rx_truncated.log_conditional(point)
                conditioned, log_probability = truncated.log_conditional(point)
                self.assertAlmostEqual(
                    log_probability, expected_log_probability, delta=1e-3
                )
                np.testing.assert_allclose(
                    conditioned.marginal([y]).log_likelihood(values),
                    expected.marginal([y]).log_likelihood(values),
                    atol=1e-3,
                )

    def test_conditioning_a_truncated_mixture_is_the_density_of_the_marginal(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, _ = layered.truncated(
            box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        )
        _, log_probability = truncated.log_conditional({x: 0.5})

        # the density of x at 0.5 integrates the joint density over the slice of y
        grid = np.linspace(0.0, 1.5, 20001)
        density = np.exp(
            truncated.log_likelihood(np.column_stack([np.full_like(grid, 0.5), grid]))
        )
        self.assertAlmostEqual(
            np.exp(log_probability), np.trapezoid(density, grid), delta=1e-3
        )

    def test_conditioning_outside_the_box_is_impossible(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, _ = layered.truncated(
            box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        )
        conditioned, log_probability = truncated.log_conditional({x: 5.0})
        self.assertIsNone(conditioned)
        self.assertEqual(log_probability, -np.inf)


# %% changing the circuit in place


class MultivariateGaussianTransformationTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(69)

    def test_translation_moves_the_density(self):
        for truncate in (False, True):
            with self.subTest(truncated=truncate):
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(
                    gaussian_mixture()
                )
                if truncate:
                    layered, _ = layered.truncated(
                        box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
                    )
                samples = layered.sample(50)
                expected = layered.log_likelihood(samples)
                layered.apply_translation({x: 1.0, y: -2.0})
                np.testing.assert_allclose(
                    layered.log_likelihood(samples + np.array([1.0, -2.0])), expected
                )

    def test_scaling_scales_the_density(self):
        for truncate in (False, True):
            with self.subTest(truncated=truncate):
                layered = RustworkxCircuitToLayeredCircuitConverter.convert(
                    gaussian_mixture()
                )
                if truncate:
                    layered, _ = layered.truncated(
                        box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
                    )
                samples = layered.sample(50)
                expected = layered.log_likelihood(samples) - np.log(2.0 * 3.0)
                layered.apply_scaling({x: 2.0, y: 3.0})
                np.testing.assert_allclose(
                    layered.log_likelihood(samples * np.array([2.0, 3.0])), expected
                )

    def test_renaming_that_reorders_the_variables(self):
        rx_circuit = gaussian_mixture()
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        samples = rx_circuit.sample(50)
        expected = layered.log_likelihood(samples)

        # a name that sorts after y moves x to the last column
        renamed_x = Continuous("z_x")
        layered.update_variables(VariableMap({x: renamed_x}))
        self.assertEqual(list(layered.variables), [y, renamed_x])
        np.testing.assert_allclose(layered.log_likelihood(samples[:, ::-1]), expected)


# %% serialization


class MultivariateGaussianSerializationTestCase(unittest.TestCase):

    def test_json_round_trip(self):
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture())
        truncated, _ = layered.truncated(
            box(x=closed(-1.0, 2.0), y=closed(0.0, 1.5)).as_composite_set()
        )
        points = np.array([[0.5, 0.5], [1.5, 1.0]])
        for circuit in (layered, truncated):
            with self.subTest(type(circuit.layers[-1]).__name__):
                restored = from_json(to_json(circuit.root))
                self.assertIsInstance(restored, SumLayer)
                np.testing.assert_allclose(
                    restored.log_likelihood_of_nodes(points)[:, 0],
                    circuit.log_likelihood(points),
                )


if __name__ == "__main__":
    unittest.main()
