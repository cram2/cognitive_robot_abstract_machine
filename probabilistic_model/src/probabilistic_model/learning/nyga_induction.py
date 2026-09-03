from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Tuple, Dict, Any

import numpy as np
import numpy.typing as npt
import random_events
from random_events.interval import closed, closed_open, reals, SimpleInterval, Bound
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous
from typing_extensions import Self

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProbabilisticCircuit,
    UnivariateContinuousLeaf,
)


@dataclass
class InductionStep:
    """
    Class for performing induction in the NygaDistributions.
    """

    data: npt.NDArray
    """
    The entire sorted and unique data points
    """

    cumulative_weights: npt.NDArray
    """
    The cumulative log_weights of the samples in the dataset.
    """

    cumulative_log_weights: npt.NDArray
    """
    The cumulative logarithmic log_weights of the samples in the dataset.
    """

    begin_index: int
    """
    Included index of the first sample.
    """

    end_index: int
    """
    Excluded index of the last sample.
    """

    nyga_induction: NygaInduction
    """
    The Nyga Distribution to mount the quantile distributions into and read the parameters from.
    """

    @property
    def variable(self):
        """
        The variable of the distribution.
        """
        return self.nyga_induction.variable

    @property
    def min_samples_per_quantile(self):
        """
        The minimal number of samples per quantile.
        """
        return self.nyga_induction.min_samples_per_quantile

    @property
    def min_likelihood_improvement(self):
        """
        The relative, minimal likelihood improvement needed to create a new quantile.
        """
        return self.nyga_induction.min_likelihood_improvement

    def left_connecting_point(self) -> float:
        """
        Calculate the left connecting point.
        """
        return self.left_connecting_point_from_index(self.begin_index)

    @property
    def number_of_samples(self):
        """
        The number of samples in the induction step.
        """
        return self.end_index - self.begin_index

    @property
    def total_weights(self):
        """
        The total sum of log_weights of the samples in the induction step.
        """
        return (
            self.cumulative_weights[self.end_index]
            - self.cumulative_weights[self.begin_index]
        )

    @property
    def total_log_weights(self):
        """
        The total sum of logarithmic log_weights of the samples in the induction step.
        """
        return (
            self.cumulative_log_weights[self.end_index]
            - self.cumulative_log_weights[self.begin_index]
        )

    def left_connecting_point_from_index(self, index) -> float:
        """
        Calculate the left connecting point given some beginning index.

        :param index: The index of the left datapoint.
        """
        if index > 0:
            left_connecting_point = (self.data[index - 1] + self.data[index]) / 2
        else:
            left_connecting_point = self.data[index]
        return left_connecting_point

    def right_connecting_point(self) -> float:
        """
        Calculate the right connecting point.
        """
        return self.right_connecting_point_from_index(self.end_index)

    def right_connecting_point_from_index(self, index) -> float:
        """
        Calculate the right connecting point given some ending index.

        :param index: The index of the right datapoint.
        """
        if index < len(self.data):
            right_connecting_point = (self.data[index] + self.data[index - 1]) / 2
        else:
            right_connecting_point = self.data[index - 1]
        return right_connecting_point

    def create_uniform_distribution(self) -> UniformDistribution:
        """
        Create a uniform distribution from this induction step.
        """
        return self.create_uniform_distribution_from_indices(
            self.begin_index, self.end_index
        )

    def create_uniform_distribution_from_indices(
        self, begin_index: int, end_index: int
    ) -> UniformDistribution:
        """
        Create a uniform distribution from the datapoint at `begin_index` to the datapoint at `end_index`.

        The piece touching the globally first or last datapoint of the induction is
        built unbounded on that side and then intersected with
        :attr:`NygaInduction.support`, so that widening the support to absorb float
        instabilities - or narrowing it to stay clear of a sibling distribution
        elsewhere in a larger circuit - is expressed entirely through that one
        interval rather than through a separate adjustment step.

        :param begin_index: The index of the first datapoint.
        :param end_index: The index of the last datapoint.
        """
        is_last_piece = end_index == len(self.data)

        lower = (
            -float("inf")
            if begin_index == 0
            else self.left_connecting_point_from_index(begin_index)
        )
        upper = (
            float("inf")
            if is_last_piece
            else self.right_connecting_point_from_index(end_index)
        )

        interval = SimpleInterval.from_data(
            lower, upper, Bound.CLOSED, Bound.CLOSED if is_last_piece else Bound.OPEN
        )
        interval = interval.intersection_with(self.nyga_induction.support)
        return UniformDistribution(variable=self.variable, interval=interval)

    def sum_weights_from_indices(self, begin_index: int, end_index: int) -> float:
        """
        Sum the log_weights from `begin_index` to `end_index`.
        """
        return self.cumulative_weights[end_index] - self.cumulative_weights[begin_index]

    def sum_weights(self):
        """
        Sum the log_weights of this induction step.
        """
        return self.sum_weights_from_indices(self.begin_index, self.end_index)

    def sum_log_weights_from_indices(self, begin_index: int, end_index: int) -> float:
        """
        Sum the logarithmic log_weights from `begin_index` to `end_index`.
        """
        return (
            self.cumulative_log_weights[end_index]
            - self.cumulative_log_weights[begin_index]
        )

    def sum_log_weights(self):
        """
        Sum the logarithmic log_weights of this induction step.
        """
        return self.sum_log_weights_from_indices(self.begin_index, self.end_index)

    def _vectorized_log_likelihood_of_split_side(
        self,
        split_indices: npt.NDArray,
        split_values: npt.NDArray,
        connecting_point: float,
        is_left: bool,
    ) -> npt.NDArray:
        """
        Log likelihood of one side of a candidate split, evaluated for every candidate
        split index of :meth:`compute_best_split` at once. Which side is being evaluated
        is passed in directly (`is_left`) rather than inferred from the data, since
        :meth:`compute_best_split` already knows it statically: every `split_value` lies
        to the right of the left connecting point and to the left of the right
        connecting point.
        """
        begin, end = (
            (self.begin_index, split_indices)
            if is_left
            else (split_indices, self.end_index)
        )
        number_of_samples = end - begin
        log_density = np.log(np.abs(split_values - connecting_point))
        log_weight_sum_of_split = np.log(self.sum_weights_from_indices(begin, end))
        sum_of_log_weights = self.sum_log_weights_from_indices(begin, end)
        log_weight_sum = np.log(self.total_weights)

        return (
            number_of_samples * (log_weight_sum_of_split - log_weight_sum - log_density)
            + sum_of_log_weights
        )

    def compute_best_split(self) -> Tuple[float, Optional[int]]:
        """
        Compute the best split of the data.

        The best split of the data is computed by evaluating the log likelihood of every possible split and memorizing
        the best one.

        This evaluates every candidate split at once via
        :meth:`_vectorized_log_likelihood_of_split_side` instead of a per-candidate
        Python loop, since that loop is the dominant cost of :meth:`NygaInduction.fit`
        on large datasets.

        :return: The maximum log likelihood and the best split index.
        """

        first_split_index = self.begin_index + self.min_samples_per_quantile
        last_split_index = self.end_index - self.min_samples_per_quantile + 1

        # empty candidate range: too few samples to leave min_samples_per_quantile on
        # both sides of any split (e.g. this step's own segment is already that small)
        if first_split_index >= last_split_index:
            return -float("inf"), None

        split_indices = np.arange(first_split_index, last_split_index)
        split_values = (
            self.data[split_indices - 1] + self.data[split_indices]
        ) / 2

        left_hand_side = self._vectorized_log_likelihood_of_split_side(
            split_indices, split_values, self.left_connecting_point(), is_left=True
        )
        right_hand_side = self._vectorized_log_likelihood_of_split_side(
            split_indices, split_values, self.right_connecting_point(), is_left=False
        )

        log_likelihoods = left_hand_side + right_hand_side
        best = int(np.argmax(log_likelihoods))
        return float(log_likelihoods[best]), int(split_indices[best])

    def log_likelihood_without_split(self) -> float:
        """
        Calculate the log likelihood without splitting.

        :return: The log likelihood without splitting.
        """
        log_density = -np.log(
            self.right_connecting_point() - self.left_connecting_point()
        )
        return self.sum_log_weights() + (self.number_of_samples * log_density)

    def construct_left_induction_step(self, split_index: int) -> Self:
        """
        Construct the left induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(
            self.data,
            self.cumulative_weights,
            self.cumulative_log_weights,
            self.begin_index,
            split_index,
            self.nyga_induction,
        )

    def construct_right_induction_step(self, split_index: int) -> Self:
        """
        Construct the right induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(
            self.data,
            self.cumulative_weights,
            self.cumulative_log_weights,
            split_index,
            self.end_index,
            self.nyga_induction,
        )

    def improvement_is_good_enough(self, maximum_log_likelihood: float) -> bool:
        """
        Check if the improvement is good enough.
        :param maximum_log_likelihood: The improved maximum log likelihood.
        :return: Rather the improvement is good enough
        """
        log_likelihood_without_split = self.log_likelihood_without_split()
        return (
            np.exp(maximum_log_likelihood - log_likelihood_without_split)
            > self.min_likelihood_improvement
        )

    def induce(self) -> List[Self]:
        """
        Perform one induction step.

        :return: The (possibly empty) list of new induction steps.
        """

        # calculate the best likelihood with splitting
        maximum_log_likelihood, best_split_index = self.compute_best_split()

        # if the improvement is good enough
        if self.improvement_is_good_enough(maximum_log_likelihood):

            # create the left and right induction steps
            left_induction_step = self.construct_left_induction_step(best_split_index)
            right_induction_step = self.construct_right_induction_step(best_split_index)
            return [left_induction_step, right_induction_step]

        # if the improvement is not good enough
        else:
            # calculate the weight of the uniform distribution
            weight = self.total_weights / self.cumulative_weights[-1]

            # mount a uniform distribution
            distribution = self.create_uniform_distribution()

            # root_unit instead of probabilistic_circuit.root, which is O(graph size)
            self.nyga_induction.root_unit.add_subcircuit(
                UnivariateContinuousLeaf(
                    distribution,
                    probabilistic_circuit=self.nyga_induction.probabilistic_circuit,
                ),
                np.log(weight),
            )

            return []


@dataclass
class NygaInduction:
    """
    A Nyga distribution is a way to learn a deterministic mixture of uniform distributions.
    """

    variable: Continuous
    """
    The variable of the distribution.
    """

    min_likelihood_improvement: float = 0.01
    """
    The relative, minimal likelihood improvement needed to create a new quantile.
    """

    min_samples_per_quantile: int = 2
    """
    The minimal number of samples per quantile.
    """

    tolerance_at_extremes: float = 1e-6
    """
    The tolerance at the extremes of the entire distribution.
    This makes the support of the distribution bigger by this value to ensure that there are no likelihood errors due to
    float precision.
    """

    support: SimpleInterval = field(default_factory=lambda: reals().simple_sets[0])
    """
    The allowed span of the distribution's support. Every uniform piece the induction
    creates is intersected with this interval, so a sibling distribution for the same
    variable elsewhere in a larger circuit (e.g. another leaf of a
    :class:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree`) can be kept
    from being overlapped by simply narrowing this interval instead of it.

    Defaults to the entire real line; :meth:`fit` always narrows it further to at
    most the data's own range widened by `tolerance_at_extremes` on both sides.
    """

    probabilistic_circuit: ProbabilisticCircuit = field(
        init=False, default_factory=ProbabilisticCircuit, compare=False
    )
    """
    The probabilistic circuit to mount the distribution into.
    """

    root_unit: Optional[SumUnit] = field(init=False, default=None, compare=False)
    """
    Direct reference to the circuit's root SumUnit, set once in :meth:`fit`. Leaves are
    attached through this instead of ``probabilistic_circuit.root`` to avoid that
    property's O(graph size) rescan on every attach (see the note in
    :meth:`InductionStep.induce`).
    """

    def fit(
        self, data: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> ProbabilisticCircuit:
        """
        Fit the distribution to the data.

        :param data: The data to fit the distribution to.
        :param weights: The optional log_weights of the data points.

        :return: The fitted distribution.
        """

        # make the data unique and sort it
        sorted_unique_data, counts = np.unique(data, return_counts=True)

        # if the data contains only one value
        if len(sorted_unique_data) == 1:
            # mount a dirac delta distribution and return
            distribution = DiracDeltaDistribution(
                variable=self.variable,
                location=sorted_unique_data[0],
                tolerance=self.tolerance_at_extremes,
            )
            UnivariateContinuousLeaf(
                distribution, probabilistic_circuit=self.probabilistic_circuit
            )

            return self.probabilistic_circuit

        # narrow the support to at most the data's own range widened by tolerance_at_extremes
        self.support = self.support.intersection_with(
            SimpleInterval.from_data(
                sorted_unique_data[0] - self.tolerance_at_extremes,
                sorted_unique_data[-1] + self.tolerance_at_extremes,
                Bound.CLOSED,
                Bound.CLOSED,
            )
        )

        # if the log_weights are not given
        if weights is None:
            weights = counts

        log_weights = np.log(weights)
        cumulative_log_weights = np.cumsum(log_weights)
        cumulative_log_weights = np.insert(cumulative_log_weights, 0, 0)

        cumulative_weights = np.cumsum(weights)
        cumulative_weights = np.insert(cumulative_weights, 0, 0)

        # create the root
        self.root_unit = SumUnit(probabilistic_circuit=self.probabilistic_circuit)

        # construct the initial induction step
        initial_induction_step = InductionStep(
            data=sorted_unique_data,
            cumulative_weights=cumulative_weights,
            cumulative_log_weights=cumulative_log_weights,
            begin_index=0,
            end_index=len(sorted_unique_data),
            nyga_induction=self,
        )

        # initialize the queue
        induction_steps: Deque[InductionStep] = collections.deque(
            [initial_induction_step]
        )

        # induce the distribution
        while len(induction_steps) > 0:
            induction_step = induction_steps.popleft()
            new_induction_steps = induction_step.induce()
            induction_steps.extend(new_induction_steps)

        self.probabilistic_circuit.normalize()
        return self.probabilistic_circuit

    def empty_copy(self) -> Self:
        return self.__class__(
            variable=self.variable,
            min_samples_per_quantile=self.min_samples_per_quantile,
            min_likelihood_improvement=self.min_likelihood_improvement,
        )

    @staticmethod
    def from_uniform_mixture(mixture: ProbabilisticCircuit) -> ProbabilisticCircuit:
        """
        Construct a Nyga Distribution from a mixture of uniform distributions.
        The mixture does not have to be deterministic.

        :param mixture: An arbitrary, univariate mixture of uniform distributions
        :return: A Nyga Distribution describing the same function.
        """

        assert (
            len(mixture.variables) == 1
        ), "Can only convert univariate circuits to nyga distributions."
        assert all(
            [
                isinstance(leaf.distribution, UniformDistribution)
                for leaf in mixture.leaves
            ]
        ), "Can only convert mixtures of uniform distributions to nyga distributions."

        variable: Continuous = mixture.variables[0]
        result = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=result)

        all_mixture_points = []
        for leaf in mixture.leaves:
            leaf: UnivariateContinuousLeaf
            all_mixture_points += [
                leaf.distribution.interval.lower,
                leaf.distribution.interval.upper,
            ]

        all_mixture_points = list(sorted(set(all_mixture_points)))

        for index, (lower, upper) in enumerate(
            zip(all_mixture_points[:-1], all_mixture_points[1:])
        ):
            if index == len(all_mixture_points) - 2:
                interval = closed(lower, upper)
            else:
                interval = closed_open(lower, upper)
            distribution = UniformDistribution(
                variable=variable, interval=interval.simple_sets[0]
            )
            leaf = UnivariateContinuousLeaf(distribution, probabilistic_circuit=result)
            weight = mixture.probability_of_simple_event(
                SimpleEvent.from_data({variable: interval})
            )
            root.add_subcircuit(leaf, np.log(weight))

        return result

    def all_union_of_mixture_points_with(self, other: Self):
        """
        Computes all possible union intervals of mixture points when combining two intervals.

        Returns: list of closed intervals representing all mixture points between distributions
        """
        all_mixture_points = set()
        for leaf in self.leaves:
            leaf: UniformDistribution
            all_mixture_points.add(leaf.interval.lower)
            all_mixture_points.add(leaf.interval.upper)

        for leaf in other.leaves:
            leaf: UniformDistribution
            all_mixture_points.add(leaf.interval.lower)
            all_mixture_points.add(leaf.interval.upper)

        all_mixture_points = list(all_mixture_points)
        all_mixture_points.sort()
        portion_list = []
        for i in range(1, len(all_mixture_points) - 1):
            portion_list += random_events.product_algebra.SimpleInterval.from_data(
                all_mixture_points[i - 1], all_mixture_points[i]
            )
        return all_mixture_points
