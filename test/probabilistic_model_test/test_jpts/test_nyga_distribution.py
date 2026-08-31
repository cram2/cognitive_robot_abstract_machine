import unittest

import numpy as np
import plotly.graph_objects as go
from numpy import testing
from random_events.interval import closed, closed_open, reals, Bound, SimpleInterval
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from random_events.variable import Continuous
from scipy.special import logsumexp

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.nyga_induction import (
    NygaInduction,
    InductionStep,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    UnivariateContinuousLeaf,
    leaf,
    ProbabilisticCircuit,
)


class InductionStepTestCase(unittest.TestCase):
    variable: Continuous = Continuous("x")
    sorted_data: np.array = np.array([1, 2, 3, 4, 7, 9])
    weights: np.array = np.ones((len(sorted_data),))
    induction_step: InductionStep

    def setUp(self) -> None:
        nyga_distribution = NygaInduction(
            self.variable, min_samples_per_quantile=1, min_likelihood_improvement=0.01
        )
        cumulative_log_weights = np.cumsum(np.log(self.weights))
        cumulative_log_weights = np.append(0, cumulative_log_weights)
        cumulative_weights = np.cumsum(self.weights)
        cumulative_weights = np.append(
            0,
            cumulative_weights,
        )
        self.induction_step = InductionStep(
            self.sorted_data,
            cumulative_weights,
            cumulative_log_weights,
            0,
            len(self.sorted_data),
            nyga_distribution,
        )

    def test_variable(self):
        self.assertEqual(self.induction_step.variable, self.variable)

    def test_left_connecting_point_edge_case(self):
        self.assertEqual(self.induction_step.left_connecting_point(), 1)

    def test_right_connecting_point_edge_case(self):
        self.assertEqual(self.induction_step.right_connecting_point(), 9)

    def test_left_connecting_point(self):
        self.assertEqual(self.induction_step.left_connecting_point_from_index(3), 3.5)

    def test_right_connecting_point(self):
        self.assertEqual(self.induction_step.right_connecting_point_from_index(5), 8.0)

    def test_create_uniform_distribution_edge_case(self):
        # a single piece spans the whole (unset) support of the induction, i.e. the
        # entire real line, rather than being clipped to the data's own bounds
        distribution = self.induction_step.create_uniform_distribution()
        self.assertEqual(
            distribution,
            UniformDistribution(
                variable=self.variable, interval=reals().simple_sets[0]
            ),
        )

    def test_create_uniform_distribution(self):
        distribution = self.induction_step.create_uniform_distribution_from_indices(
            3, 5
        )
        self.assertEqual(
            distribution,
            UniformDistribution(
                variable=self.variable, interval=closed_open(3.5, 8.0).simple_sets[0]
            ),
        )

    def test_sum_weights(self):
        self.assertAlmostEqual(self.induction_step.sum_weights(), 6)

    def test_sum_weights_from_indices(self):
        self.assertAlmostEqual(self.induction_step.sum_weights_from_indices(3, 5), 2)

    def test_likelihood_of_split(self):
        """
        Test that the calculation of the likelihood of a split is correct and as it is
        in the notebook.
        """
        likelihood_without_split = self.induction_step.log_likelihood_without_split()
        self.assertAlmostEqual(likelihood_without_split, -12.48, delta=0.01)

        # k = 1
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(
            1, 1
        )
        self.assertAlmostEqual(likelihood_of_split_left, -1.1, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(
            1, 9
        )
        self.assertAlmostEqual(likelihood_of_split_right, -10.99, delta=0.01)

        # k = 2
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(
            2, 1
        )
        self.assertAlmostEqual(likelihood_of_split_left, -3.01, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(
            2, 9
        )
        self.assertAlmostEqual(likelihood_of_split_right, -9.11, delta=0.01)

        # k = 3
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(
            3, 1
        )
        self.assertAlmostEqual(likelihood_of_split_left, -4.83, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(
            3, 9
        )
        self.assertAlmostEqual(likelihood_of_split_right, -7.19, delta=0.01)

        # k = 4
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(
            4, 1
        )
        self.assertAlmostEqual(likelihood_of_split_left, -7.64, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(
            4, 9
        )
        self.assertAlmostEqual(likelihood_of_split_right, -4.7, delta=0.01)

        # k = 5
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(
            5, 1
        )
        self.assertAlmostEqual(likelihood_of_split_left, -10.64, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(
            5, 9
        )
        self.assertAlmostEqual(likelihood_of_split_right, -1.79, delta=0.01)

    def test_compute_best_split(self):
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 3)

    def test_compute_best_split_without_result(self):
        self.induction_step.nyga_induction.min_samples_per_quantile = 4
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, None)
        self.assertEqual(maximum, -float("inf"))

    def test_compute_best_split_with_induced_indices(self):
        self.induction_step.begin_index = 3
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 5)

    def test_construct_left_induction_step(self):
        induction_step = self.induction_step.construct_left_induction_step(1)
        self.assertEqual(induction_step.begin_index, 0)
        self.assertEqual(induction_step.end_index, 1)
        testing.assert_equal(induction_step.data, self.induction_step.data)
        testing.assert_equal(
            induction_step.cumulative_log_weights,
            self.induction_step.cumulative_log_weights,
        )

    def test_construct_right_induction_step(self):
        induction_step = self.induction_step.construct_right_induction_step(1)
        self.assertEqual(induction_step.begin_index, 1)
        self.assertEqual(induction_step.end_index, 6)
        testing.assert_equal(induction_step.data, self.induction_step.data)
        testing.assert_equal(
            induction_step.cumulative_log_weights,
            self.induction_step.cumulative_log_weights,
        )

    def test_fit(self):
        np.random.seed(69)
        self.induction_step.nyga_induction.min_samples_per_quantile = 20
        self.induction_step.nyga_induction.min_likelihood_improvement = 0
        data = np.random.normal(0, 1, 500).tolist()
        distribution = self.induction_step.nyga_induction
        pc = distribution.fit(data)
        self.assertLessEqual(
            len(pc.root.subcircuits),
            int(
                len(data) / self.induction_step.nyga_induction.min_samples_per_quantile
            ),
        )
        self.assertAlmostEqual(logsumexp(pc.root.log_weights), 0.0)

    def test_domain(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = self.induction_step.nyga_induction
        pc = distribution.fit(data)
        domain = pc.support
        self.assertEqual(len(domain.simple_sets), 1)

        lowest = min(data)
        highest = max(data)

        self.assertAlmostEqual(
            domain.simple_sets[0][self.variable].simple_sets[0].lower,
            lowest,
            delta=10e-5,
        )
        self.assertAlmostEqual(
            domain.simple_sets[0][self.variable].simple_sets[-1].upper,
            highest,
            delta=10e-5,
        )

    def test_plot(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100)
        distribution = self.induction_step.nyga_induction
        pc = distribution.fit(data)
        fig = go.Figure(pc.plot())
        self.assertIsNotNone(fig)
        # fig.show()

    def test_fit_from_singular_data(self):
        data = [1.0, 1.0]
        distribution = self.induction_step.nyga_induction
        pc = distribution.fit(data)
        self.assertEqual(len(pc.nodes()), 1)
        self.assertIsInstance(pc.root.distribution, DiracDeltaDistribution)

    def test_serialization(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = NygaInduction(self.variable, min_likelihood_improvement=0.01)
        distribution.fit(data)
        serialized = to_json(distribution)
        deserialized = from_json(serialized)
        self.assertIsInstance(deserialized, NygaInduction)
        self.assertEqual(distribution, deserialized)

    def test_from_mixture_of_uniform_distributions(self):
        pc1 = ProbabilisticCircuit()
        u1 = leaf(
            UniformDistribution(
                variable=self.variable, interval=closed(0, 5).simple_sets[0]
            ),
            pc1,
        )
        u2 = leaf(
            UniformDistribution(
                variable=self.variable, interval=closed(2, 3).simple_sets[0]
            ),
            pc1,
        )
        sum_unit = SumUnit(probabilistic_circuit=pc1)
        sum_unit.add_subcircuit(u1, np.log(0.5))
        sum_unit.add_subcircuit(u2, np.log(0.5))
        distribution = NygaInduction.from_uniform_mixture(
            sum_unit.probabilistic_circuit
        )

        solution_by_hand = ProbabilisticCircuit()
        root_of_solution = SumUnit(probabilistic_circuit=solution_by_hand)
        leaf_1 = UniformDistribution(
            variable=self.variable, interval=closed_open(0, 2).simple_sets[0]
        )
        leaf_2 = UniformDistribution(
            variable=self.variable, interval=closed_open(2, 3).simple_sets[0]
        )
        leaf_3 = UniformDistribution(
            variable=self.variable, interval=closed(3, 5).simple_sets[0]
        )

        root_of_solution.add_subcircuit(leaf(leaf_1, solution_by_hand), np.log(0.2))
        root_of_solution.add_subcircuit(leaf(leaf_2, solution_by_hand), np.log(0.6))
        root_of_solution.add_subcircuit(leaf(leaf_3, solution_by_hand), np.log(0.2))

        self.assertEqual(len(distribution.leaves), 3)
        self.assertSetEqual(
            set([leaf.distribution for leaf in distribution.leaves]),
            {leaf_1, leaf_2, leaf_3},
        )
        self.assertTrue(
            np.allclose(
                distribution.root.log_weights, solution_by_hand.root.log_weights
            )
        )


class FittedNygaDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    model: NygaInduction
    data: np.array

    def setUp(self) -> None:
        np.random.seed(420)
        self.model = NygaInduction(
            self.x, min_likelihood_improvement=0.001, min_samples_per_quantile=300
        )
        data = np.random.normal(0.0, 1.0, 1000).astype(np.float32)
        data.sort()
        self.model.fit(data)
        self.data = data

    def test_plot(self):
        fig = go.Figure(self.model.probabilistic_circuit.plot())
        self.assertIsNotNone(fig)
        # fig.show()

    def test_determinism(self):
        self.assertTrue(self.model.probabilistic_circuit.is_deterministic())

    def test_likelihood(self):
        likelihood = self.model.probabilistic_circuit.log_likelihood(
            self.data.reshape(-1, 1)
        )
        self.assertEqual(likelihood.shape, (1000,))
        self.assertGreater(likelihood.min(), -np.inf)


class SupportedUniformDistributionTestCase(unittest.TestCase):
    """
    Tests for how `InductionStep.create_uniform_distribution_from_indices` uses
    `NygaInduction.support`: the piece touching the globally first or last datapoint is
    intersected with it, which is what lets a leaf's tolerance widening be capped
    without a separate adjustment step.
    """

    variable: Continuous = Continuous("x")
    sorted_data: np.array = np.array([1, 2, 3, 4, 7, 9])

    def induction_step_with_support(self, support: SimpleInterval) -> InductionStep:
        nyga_distribution = NygaInduction(
            self.variable, min_samples_per_quantile=1, support=support
        )
        weights = np.ones((len(self.sorted_data),))
        cumulative_weights = np.append(0, np.cumsum(weights))
        cumulative_log_weights = np.append(0, np.cumsum(np.log(weights)))
        return InductionStep(
            self.sorted_data,
            cumulative_weights,
            cumulative_log_weights,
            0,
            len(self.sorted_data),
            nyga_distribution,
        )

    def test_edge_piece_is_unconstrained_by_the_default_support(self):
        distribution = self.induction_step_with_support(
            reals().simple_sets[0]
        ).create_uniform_distribution()
        self.assertEqual(
            distribution,
            UniformDistribution(
                variable=self.variable, interval=reals().simple_sets[0]
            ),
        )

    def test_edge_piece_is_clipped_to_a_given_support(self):
        support = closed(0.0, 5.0).simple_sets[0]
        distribution = self.induction_step_with_support(
            support
        ).create_uniform_distribution()
        self.assertEqual(distribution.interval, support)

    def test_internal_piece_is_unaffected_by_a_wider_support(self):
        induction_step = self.induction_step_with_support(
            closed(-10.0, 20.0).simple_sets[0]
        )
        distribution = induction_step.create_uniform_distribution_from_indices(3, 5)
        self.assertEqual(
            distribution,
            UniformDistribution(
                variable=self.variable, interval=closed_open(3.5, 8.0).simple_sets[0]
            ),
        )


class NygaInductionSupportTestCase(unittest.TestCase):
    """
    Tests for how `NygaInduction.fit` narrows `support` - the mechanism that lets a
    sibling leaf's data (e.g. another leaf of a :class:`JointProbabilityTree`) keep
    this distribution's support from overlapping it.
    """

    variable: Continuous = Continuous("x")

    def test_fit_narrows_the_default_support_to_data_range_widened_by_tolerance(self):
        model = NygaInduction(
            self.variable, min_samples_per_quantile=1, min_likelihood_improvement=0
        )
        model.fit([1.0, 2.0, 3.0])
        # SimpleInterval stores its bounds as single-precision floats, so the exact
        # value picks up rounding noise around the scale of tolerance_at_extremes itself
        self.assertAlmostEqual(
            model.support.lower, 1.0 - model.tolerance_at_extremes, delta=1e-5
        )
        self.assertAlmostEqual(
            model.support.upper, 3.0 + model.tolerance_at_extremes, delta=1e-5
        )

    def test_fit_narrows_a_given_support_no_further_than_the_tolerance_window(self):
        # wide enough that the tolerance window around the data sits inside it
        given_support = closed(0.0, 10.0).simple_sets[0]
        model = NygaInduction(
            self.variable,
            support=given_support,
            min_samples_per_quantile=1,
            min_likelihood_improvement=0,
        )
        model.fit([1.0, 2.0, 3.0])

        self.assertAlmostEqual(
            model.support.lower, 1.0 - model.tolerance_at_extremes, delta=1e-5
        )
        self.assertAlmostEqual(
            model.support.upper, 3.0 + model.tolerance_at_extremes, delta=1e-5
        )

    def test_fit_narrows_a_given_support_to_stay_inside_it(self):
        # narrower than the tolerance window around the data on the upper side
        given_support = closed(0.0, 3.0000005).simple_sets[0]
        model = NygaInduction(
            self.variable,
            support=given_support,
            min_samples_per_quantile=1,
            min_likelihood_improvement=0,
        )
        model.fit([1.0, 2.0, 3.0])

        widest_leaf = max(
            model.probabilistic_circuit.leaves,
            key=lambda leaf: leaf.distribution.interval.upper,
        )
        self.assertAlmostEqual(
            widest_leaf.distribution.interval.upper, 3.0000005, delta=1e-5
        )


if __name__ == "__main__":
    unittest.main()
