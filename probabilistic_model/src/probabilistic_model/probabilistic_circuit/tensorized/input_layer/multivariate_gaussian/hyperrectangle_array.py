from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from random_events.interval import Bound, SimpleInterval
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable
from typing_extensions import Any, List, Self, Sequence

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeScopeIntervalBounds,
    NodeScopeIntervals,
    SampleNodeMask,
    SampleScopeValues,
)


@dataclass
class HyperrectangleArray:
    """
    Hyperrectangles, which are axis-aligned boxes, over the variables in the scope of a
    layer: one simple interval per variable and hyperrectangle.

    The leading axes enumerate the hyperrectangles: a single one has the shape
    (#variables of the layer, 2), one per node (#nodes, #variables of the layer, 2).
    """

    interval: NodeScopeIntervals
    """
    The lower and upper bound of every interval.
    """

    bounds: NodeScopeIntervalBounds
    """
    Whether the lower and upper bound of every interval are open or closed.
    """

    @classmethod
    def of_simple_intervals(cls, intervals: Sequence[SimpleInterval]) -> Self:
        """
        :param intervals: One simple interval per variable in the scope of a layer.
        :return: The hyperrectangle they span.
        """
        return cls(
            np.array(
                [[interval.lower, interval.upper] for interval in intervals],
                dtype=float,
            ),
            np.array(
                [[int(interval.left), int(interval.right)] for interval in intervals],
                dtype=np.int64,
            ),
        )

    @classmethod
    def stack(cls, hyperrectangles: List[Self]) -> Self:
        """
        :param hyperrectangles: Single hyperrectangles over the same variables.
        :return: All of them along a new first axis, one per node.
        """
        return cls(
            np.array([hyperrectangle.interval for hyperrectangle in hyperrectangles]),
            np.array([hyperrectangle.bounds for hyperrectangle in hyperrectangles]),
        )

    @classmethod
    def concatenate(cls, arrays: List[Self]) -> Self:
        """
        :param arrays: Hyperrectangles over the same variables, one per node each.
        :return: All of their hyperrectangles, in order.
        """
        return cls(
            np.concatenate([array.interval for array in arrays]),
            np.concatenate([array.bounds for array in arrays]),
        )

    @property
    def lower(self) -> np.ndarray:
        """
        :return: The lower bound of every interval.
        """
        return self.interval[..., 0]

    @property
    def upper(self) -> np.ndarray:
        """
        :return: The upper bound of every interval.
        """
        return self.interval[..., 1]

    @property
    def is_whole_space(self) -> bool:
        """
        :return: Whether every hyperrectangle leaves every variable unbounded.
        """
        return bool(np.all(self.lower == -np.inf) and np.all(self.upper == np.inf))

    def intersection_with(self, other: HyperrectangleArray) -> HyperrectangleArray:
        """
        :param other: Hyperrectangles whose shape broadcasts against these.
        :return: The intersection of every hyperrectangle with the matching one of
            ``other``. Where two bounds coincide the result is open if either of them
            is.
            :attr:`Bound.OPEN` is the larger value, so that is a maximum.
        """
        own_lower, other_lower = np.broadcast_arrays(self.lower, other.lower)
        own_upper, other_upper = np.broadcast_arrays(self.upper, other.upper)
        own_left, other_left = np.broadcast_arrays(
            self.bounds[..., 0], other.bounds[..., 0]
        )
        own_right, other_right = np.broadcast_arrays(
            self.bounds[..., 1], other.bounds[..., 1]
        )
        left = np.where(
            own_lower > other_lower,
            own_left,
            np.where(
                own_lower < other_lower, other_left, np.maximum(own_left, other_left)
            ),
        )
        right = np.where(
            own_upper < other_upper,
            own_right,
            np.where(
                own_upper > other_upper,
                other_right,
                np.maximum(own_right, other_right),
            ),
        )
        return HyperrectangleArray(
            np.stack(
                [
                    np.maximum(own_lower, other_lower),
                    np.minimum(own_upper, other_upper),
                ],
                axis=-1,
            ),
            np.stack([left, right], axis=-1),
        )

    def contains(self, values: SampleScopeValues) -> SampleNodeMask:
        """
        :param values: Points over the variables of the hyperrectangles, shape
            (#samples, #variables of the layer).
        :return: Whether every hyperrectangle, one per node, contains every point, shape
            (#samples, #nodes).
        """
        points = values[:, None, :]
        left_closed = self.bounds[..., 0] == int(Bound.CLOSED)
        right_closed = self.bounds[..., 1] == int(Bound.CLOSED)
        left = np.where(left_closed, self.lower <= points, self.lower < points)
        right = np.where(right_closed, points <= self.upper, points < self.upper)
        return (left & right).all(axis=-1)

    def simple_event_of(self, index: int, variables: Sequence[Variable]) -> SimpleEvent:
        """
        :param index: The index of a hyperrectangle along the first axis.
        :param variables: The variables of the hyperrectangle.
        :return: That hyperrectangle as a simple event.
        """
        return SimpleEvent.from_data(
            {
                variable: SimpleInterval.from_data(
                    float(self.interval[index, position, 0]),
                    float(self.interval[index, position, 1]),
                    Bound(int(self.bounds[index, position, 0])),
                    Bound(int(self.bounds[index, position, 1])),
                ).as_composite_set()
                for position, variable in enumerate(variables)
            }
        )

    def select(self, indices: Any) -> HyperrectangleArray:
        """
        :param indices: A mask or index array over the first axis.
        :return: The selected hyperrectangles.
        """
        return HyperrectangleArray(self.interval[indices], self.bounds[indices])

    def over(self, positions: NodeIndices) -> HyperrectangleArray:
        """
        :param positions: Positions of variables of the hyperrectangles.
        :return: The hyperrectangles over only those variables, in that order.
        """
        return HyperrectangleArray(
            self.interval[..., positions, :], self.bounds[..., positions, :]
        )

    def translated(self, offsets: np.ndarray) -> HyperrectangleArray:
        """
        :param offsets: What to move every variable by.
        :return: The moved hyperrectangles.
        """
        return HyperrectangleArray(self.interval + offsets[:, None], self.bounds)

    def scaled(self, factors: np.ndarray) -> HyperrectangleArray:
        """
        :param factors: What to multiply every variable by.
        :return: The scaled hyperrectangles.
        """
        return HyperrectangleArray(self.interval * factors[:, None], self.bounds)

    def broadcast_to(self, number_of_hyperrectangles: int) -> HyperrectangleArray:
        """
        :param number_of_hyperrectangles: How many copies to make of this single
            hyperrectangle.
        :return: The copies, one per entry of the first axis.
        """
        return HyperrectangleArray(
            np.tile(self.interval, (number_of_hyperrectangles, 1, 1)),
            np.tile(self.bounds, (number_of_hyperrectangles, 1, 1)),
        )
