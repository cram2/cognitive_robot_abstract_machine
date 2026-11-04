from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
from typing_extensions import Self
from random_events.interval import closed, closed_open, reals, SimpleInterval, Bound
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProbabilisticCircuit,
    UnivariateContinuousLeaf,
)


# %% induction steps


@dataclass
class InductionStep:
    """
    A segment of the sorted, unique data that the induction either splits in two or
    models with one uniform distribution.
    """

    data: npt.NDArray
    """
    The entire sorted and unique data points.
    """

    cumulative_weights: npt.NDArray
    """
    The cumulative weights of the unique data points, starting at 0.
    """

    begin_index: int
    """
    Included index of the first data point of the segment.
    """

    end_index: int
    """
    Excluded index of the last data point of the segment.
    """

    nyga_induction: NygaInduction
    """
    The Nyga Distribution to mount the quantile distributions into and read the parameters from.
    """

    minimum_split_gain: float
    """
    The minimal gain in weighted log likelihood a split of this segment needs to be
    made.
    """

    @property
    def variable(self) -> Continuous:
        """
        The variable of the distribution.
        """
        return self.nyga_induction.variable

    @property
    def minimum_samples_per_quantile(self) -> int:
        """
        The minimal number of unique values on each side of a split.
        """
        return self.nyga_induction.minimum_samples_per_quantile

    @property
    def total_weights(self) -> float:
        """
        The total weight of the data points in the segment.
        """
        return self.sum_weights_from_indices(self.begin_index, self.end_index)

    def left_connecting_point(self) -> float:
        """
        Calculate the left connecting point of the segment.
        """
        return float(self.left_connecting_point_from_index(self.begin_index))

    def right_connecting_point(self) -> float:
        """
        Calculate the right connecting point of the segment.
        """
        return float(self.right_connecting_point_from_index(self.end_index))

    def left_connecting_point_from_index(self, index: int | npt.NDArray) -> npt.NDArray:
        """
        Calculate the left connecting point of segments beginning at some indices.

        :param index: The indices of the first data point of each segment.
        :return: The connecting points, in the shape of the indices.
        """
        index = np.asarray(index)
        previous_value = self.data[np.maximum(index - 1, 0)]
        return np.where(
            index > 0, (previous_value + self.data[index]) / 2, self.data[index]
        )

    def right_connecting_point_from_index(
        self, index: int | npt.NDArray
    ) -> npt.NDArray:
        """
        Calculate the right connecting point of segments ending at some indices.

        :param index: The excluded indices of the last data point of each segment.
        :return: The connecting points, in the shape of the indices.
        """
        index = np.asarray(index)
        next_value = self.data[np.minimum(index, len(self.data) - 1)]
        return np.where(
            index < len(self.data),
            (self.data[index - 1] + next_value) / 2,
            self.data[index - 1],
        )

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
        Create a uniform distribution from the datapoint at `begin_index` to the
        datapoint at `end_index`.

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
            else float(self.left_connecting_point_from_index(begin_index))
        )
        upper = (
            float("inf")
            if is_last_piece
            else float(self.right_connecting_point_from_index(end_index))
        )

        interval = SimpleInterval.from_data(
            lower, upper, Bound.CLOSED, Bound.CLOSED if is_last_piece else Bound.OPEN
        )
        interval = interval.intersection_with(self.nyga_induction.support)
        return UniformDistribution(variable=self.variable, interval=interval)

    def sum_weights_from_indices(
        self, begin_index: int | npt.NDArray, end_index: int | npt.NDArray
    ) -> float | npt.NDArray:
        """
        Sum the weights from `begin_index` (included) to `end_index` (excluded).
        """
        return self.cumulative_weights[end_index] - self.cumulative_weights[begin_index]

    def log_likelihood_of_segments(
        self, begin_indices: npt.NDArray, end_indices: npt.NDArray
    ) -> npt.NDArray:
        """
        The weighted log likelihood of the data in each segment from a begin index
        (included) to an end index (excluded), under a uniform distribution over the
        segment between its connecting points, weighted by the segment's share of this
        step's total weight.

        :param begin_indices: The index of the first data point of each segment.
        :param end_indices: The excluded index of the last data point of each segment.
        :return: The log likelihood of each segment.
        """
        weights = self.sum_weights_from_indices(begin_indices, end_indices)
        widths = self.right_connecting_point_from_index(
            end_indices
        ) - self.left_connecting_point_from_index(begin_indices)
        return weights * (np.log(weights / self.total_weights) - np.log(widths))

    def compute_best_split(self) -> tuple[float, int | None]:
        """
        Compute the best split of the data.

        The best split of the data is computed by evaluating the log likelihood of every
        possible split at once and memorizing the best one.

        :return: The maximum log likelihood and the best split index.
        """
        first_split_index = self.begin_index + self.minimum_samples_per_quantile
        last_split_index = self.end_index - self.minimum_samples_per_quantile + 1

        # empty candidate range: too few samples to leave minimum_samples_per_quantile on
        # both sides of any split (e.g. this step's own segment is already that small)
        if first_split_index >= last_split_index:
            return -float("inf"), None

        split_indices = np.arange(first_split_index, last_split_index)
        log_likelihoods = self.log_likelihood_of_segments(
            np.full_like(split_indices, self.begin_index), split_indices
        ) + self.log_likelihood_of_segments(
            split_indices, np.full_like(split_indices, self.end_index)
        )
        best = int(np.argmax(log_likelihoods))
        return float(log_likelihoods[best]), int(split_indices[best])

    def log_likelihood_without_split(self) -> float:
        """
        :return: The weighted log likelihood of the segment under one uniform
            distribution.
        """
        return float(
            self.log_likelihood_of_segments(
                np.array([self.begin_index]), np.array([self.end_index])
            )[0]
        )

    def construct_left_induction_step(self, split_index: int) -> Self:
        """
        Construct the left induction step.

        :param split_index: The index of the split.
        """
        return replace(self, end_index=split_index)

    def construct_right_induction_step(self, split_index: int) -> Self:
        """
        Construct the right induction step.

        :param split_index: The index of the split.
        """
        return replace(self, begin_index=split_index)

    def improvement_is_good_enough(self, maximum_log_likelihood: float) -> bool:
        """
        Check if a split gains more than :attr:`minimum_split_gain`.

        :param maximum_log_likelihood: The log likelihood of the best split.
        :return: Whether the split gains enough.
        """
        return (
            maximum_log_likelihood - self.log_likelihood_without_split()
            > self.minimum_split_gain
        )

    def mount_uniform_distribution(self) -> None:
        """
        Mount a uniform distribution over the segment into the root of the induced
        circuit, weighted by the segment's share of the weight of the entire data.
        """
        weight = self.total_weights / self.cumulative_weights[-1]

        # root_unit instead of probabilistic_circuit.root, which is O(graph size)
        self.nyga_induction.root_unit.add_subcircuit(
            UnivariateContinuousLeaf(
                self.create_uniform_distribution(),
                probabilistic_circuit=self.nyga_induction.probabilistic_circuit,
            ),
            np.log(weight),
        )

    def induce(self) -> list[Self]:
        """
        Perform one induction step: split the segment if that gains enough, otherwise
        mount a uniform distribution over it.

        :return: The (possibly empty) list of new induction steps.
        """
        maximum_log_likelihood, best_split_index = self.compute_best_split()

        if not self.improvement_is_good_enough(maximum_log_likelihood):
            self.mount_uniform_distribution()
            return []

        return [
            self.construct_left_induction_step(best_split_index),
            self.construct_right_induction_step(best_split_index),
        ]

    def mount_greedy_partition(self) -> None:
        """
        Split the segment recursively at the best split until no split gains enough, and
        mount a uniform distribution over each resulting segment.
        """
        induction_steps: collections.deque[InductionStep] = collections.deque([self])
        while len(induction_steps) > 0:
            induction_steps.extend(induction_steps.popleft().induce())

    def optimal_segment_boundaries(self) -> list[int]:
        """
        Find the partition of the segment that maximizes the log likelihood minus
        :attr:`minimum_split_gain` for every split, among all partitions whose parts
        each hold at least :attr:`minimum_samples_per_quantile` unique values.

        .. note::
            This is optimal partitioning :cite:p:`jackson2005algorithm` with PELT
            pruning :cite:p:`killick2012optimal`. Pruning is exact here, since a split
            never lowers the log likelihood.

        :return: The indices that bound the parts, :attr:`begin_index` and
            :attr:`end_index` included.
        """
        if self.end_index - self.begin_index < 2 * self.minimum_samples_per_quantile:
            return [self.begin_index, self.end_index]

        # best penalized log likelihood of the data before each index, where the first
        # part is not penalized
        best_log_likelihood = np.full(self.end_index + 1, -np.inf)
        best_log_likelihood[self.begin_index] = self.minimum_split_gain
        previous_boundary = np.zeros(self.end_index + 1, dtype=int)

        candidates = np.array([self.begin_index])
        # a candidate dominated at some index is still needed for the indices that are
        # too close to that index to start a part there
        dominated_from = np.array([np.inf])

        for end_index in range(self.begin_index + 1, self.end_index + 1):
            log_likelihoods = best_log_likelihood[
                candidates
            ] + self.log_likelihood_of_segments(
                candidates, np.full_like(candidates, end_index)
            )
            admissible = end_index - candidates >= self.minimum_samples_per_quantile

            if np.any(admissible):
                best = int(np.argmax(np.where(admissible, log_likelihoods, -np.inf)))
                best_log_likelihood[end_index] = (
                    log_likelihoods[best] - self.minimum_split_gain
                )
                previous_boundary[end_index] = candidates[best]

            dominated = log_likelihoods <= best_log_likelihood[end_index]
            dominated_from[dominated] = np.minimum(
                dominated_from[dominated], end_index + self.minimum_samples_per_quantile
            )
            kept = dominated_from > end_index + 1
            candidates, dominated_from = candidates[kept], dominated_from[kept]

            if np.isfinite(best_log_likelihood[end_index]):
                candidates = np.append(candidates, end_index)
                dominated_from = np.append(dominated_from, np.inf)

        boundaries = [self.end_index]
        while boundaries[-1] != self.begin_index:
            boundaries.append(int(previous_boundary[boundaries[-1]]))
        return boundaries[::-1]

    def mount_optimal_partition(self) -> None:
        """
        Mount a uniform distribution over each part of
        :meth:`optimal_segment_boundaries`.
        """
        boundaries = self.optimal_segment_boundaries()
        for begin_index, end_index in itertools.pairwise(boundaries):
            replace(
                self, begin_index=begin_index, end_index=end_index
            ).mount_uniform_distribution()


# %% induction


@dataclass
class NygaInduction:
    """
    A Nyga distribution is a way to learn a deterministic mixture of uniform
    distributions.
    """

    variable: Continuous
    """
    The variable of the distribution.
    """

    split_penalty_factor: float = 1.0
    """
    The factor on the penalty a split has to outweigh in log likelihood.

    At 1 the penalty is the Bayesian information criterion
    :cite:p:`schwarz1978estimating` of the two parameters a split adds.
    """

    minimum_samples_per_quantile: int = 2
    """
    The minimal number of unique values on each side of a split.
    """

    tolerance_at_extremes: float = 1e-6
    """
    The tolerance at the extremes of the entire distribution.

    This makes the support of the distribution bigger by this value to ensure that there
    are no likelihood errors due to float precision.
    """

    support: SimpleInterval = field(default_factory=lambda: reals().simple_sets[0])
    """
    The allowed span of the distribution's support.

    Every uniform piece the induction creates is intersected with this interval, so a sibling distribution for the same
    variable elsewhere in a larger circuit (e.g. another leaf of a
    :class:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree`) can be kept
    from being overlapped by simply narrowing this interval instead of it.

    Defaults to the entire real line; :meth:`fit` always narrows it further to at
    most the data's own range widened by `tolerance_at_extremes` on both sides.
    """

    maximum_unique_values_for_optimal_partition: int = 5000
    """
    The largest number of unique values :meth:`fit` finds the partition with the highest
    penalized log likelihood for.

    Data with more unique values is split greedily.
    """

    probabilistic_circuit: ProbabilisticCircuit = field(
        init=False, default_factory=ProbabilisticCircuit, compare=False
    )
    """
    The probabilistic circuit to mount the distribution into.
    """

    root_unit: SumUnit | None = field(init=False, default=None, compare=False)
    """
    Direct reference to the circuit's root SumUnit, set once in :meth:`fit`.

    Leaves are attached through this instead of ``probabilistic_circuit.root`` to avoid
    that property's O(graph size) rescan on every attach (see the note in
    :meth:`InductionStep.mount_uniform_distribution`).
    """

    def minimum_split_gain(self, weights: npt.NDArray) -> float:
        """
        The gain in weighted log likelihood a split of data with these weights has to
        exceed: :attr:`split_penalty_factor` times the logarithm of the effective sample
        size, in the scale of the weights.

        :param weights: The weight of every data point.
        :return: The minimal gain.
        """
        total_weight = weights.sum()
        effective_sample_size = total_weight**2 / np.square(weights).sum()
        return float(
            self.split_penalty_factor
            * np.log(effective_sample_size)
            * total_weight
            / effective_sample_size
        )

    def initial_induction_step(
        self, data: npt.ArrayLike, weights: npt.ArrayLike | None = None
    ) -> InductionStep:
        """
        :param data: The data to fit the distribution to.
        :param weights: The weight of every data point; every point weighs 1 if not
            given.
        :return: The induction step over all the data, where every unique value weighs
            the sum of its points' weights.
        """
        data = np.asarray(data).reshape(-1)
        weights = (
            np.ones(len(data))
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        )
        sorted_unique_data, inverse = np.unique(data, return_inverse=True)
        unique_weights = np.bincount(inverse.reshape(-1), weights=weights)
        return InductionStep(
            data=sorted_unique_data,
            cumulative_weights=np.insert(np.cumsum(unique_weights), 0, 0),
            begin_index=0,
            end_index=len(sorted_unique_data),
            nyga_induction=self,
            minimum_split_gain=self.minimum_split_gain(weights),
        )

    def fit(
        self, data: npt.ArrayLike, weights: npt.ArrayLike | None = None
    ) -> ProbabilisticCircuit:
        """
        Fit the distribution to the data.

        :param data: The data to fit the distribution to.
        :param weights: The weight of every data point; every point weighs 1 if not
            given.
        :return: The fitted distribution.
        """
        initial_induction_step = self.initial_induction_step(data, weights)
        sorted_unique_data = initial_induction_step.data

        if len(sorted_unique_data) == 1:
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

        self.root_unit = SumUnit(probabilistic_circuit=self.probabilistic_circuit)

        if len(sorted_unique_data) <= self.maximum_unique_values_for_optimal_partition:
            initial_induction_step.mount_optimal_partition()
        else:
            initial_induction_step.mount_greedy_partition()

        self.probabilistic_circuit.normalize()
        return self.probabilistic_circuit

    def empty_copy(self) -> Self:
        """
        :return: An induction with the same parameters and an empty circuit.
        """
        return replace(self)

    @staticmethod
    def from_uniform_mixture(mixture: ProbabilisticCircuit) -> ProbabilisticCircuit:
        """
        Construct a Nyga Distribution from a mixture of uniform distributions.

        The mixture does not have to be deterministic.

        :param mixture: An arbitrary, univariate mixture of uniform distributions
        :return: A Nyga Distribution describing the same function.
        """
        assert len(mixture.variables) == 1, (
            "Can only convert univariate circuits to nyga distributions."
        )
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
