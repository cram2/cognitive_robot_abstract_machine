from __future__ import annotations

from _operator import or_
from dataclasses import dataclass

from enum import Enum, IntEnum
from functools import reduce
from typing import List, Tuple

from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_simple_event,
)
from random_events.interval import Bound, SimpleInterval
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous

from semantic_digital_twin.datastructures.variables import SpatialVariables


class IntervalConstants:
    """
    Predefined intervals for semantic directions.
    """

    ZERO = (0, 0, Bound.CLOSED, Bound.CLOSED)
    ZERO_TO_ONE_THIRD = (0, 1 / 3, Bound.CLOSED, Bound.CLOSED)
    ONE_THIRD_TO_TWO_THIRD = (1 / 3, 2 / 3, Bound.OPEN, Bound.OPEN)
    HALF = (0.5, 0.5, Bound.CLOSED, Bound.CLOSED)
    TWO_THIRD_TO_ONE = (2 / 3, 1, Bound.CLOSED, Bound.CLOSED)
    ONE = (1, 1, Bound.CLOSED, Bound.CLOSED)


class SemanticDirection(Enum): ...


class HorizontalSemanticDirection(SimpleInterval, SemanticDirection):
    """
    Semantic directions for horizontal positioning.
    """

    FULLY_LEFT = IntervalConstants.ZERO
    LEFT = IntervalConstants.ZERO_TO_ONE_THIRD
    CENTER = IntervalConstants.ONE_THIRD_TO_TWO_THIRD
    FULLY_CENTER = IntervalConstants.HALF
    RIGHT = IntervalConstants.TWO_THIRD_TO_ONE
    FULLY_RIGHT = IntervalConstants.ONE


class VerticalSemanticDirection(SimpleInterval, SemanticDirection):
    """
    Semantic directions for vertical positioning.
    """

    FULLY_BOTTOM = IntervalConstants.ZERO
    BOTTOM = IntervalConstants.ZERO_TO_ONE_THIRD
    CENTER = IntervalConstants.ONE_THIRD_TO_TWO_THIRD
    FULLY_CENTER = IntervalConstants.HALF
    TOP = IntervalConstants.TWO_THIRD_TO_ONE
    FULLY_TOP = IntervalConstants.ONE


@dataclass
class SemanticPositionDescription:
    """
    Describes a position by mapping semantic concepts (RIGHT, CENTER, LEFT, TOP, BOTTOM) to instances of
    random_events.intervals.SimpleInterval, which are then used to "zoom" into specific regions of an event.
    Each DirectionInterval divides the original event into three parts, either vertically or horizontally, and zooms into
    one of them depending on which specific direction was chosen.
    The sequence of zooms is defined by the order of directions in the horizontal_direction_chain and
    vertical_direction_chain lists.
    Finally, we can sample aa 2d pose from the resulting event
    """

    horizontal_direction_chain: List[HorizontalSemanticDirection]
    """
    Describes the sequence of zooms in the horizontal direction (Y axis).
    """

    vertical_direction_chain: List[VerticalSemanticDirection]
    """
    Describes the sequence of zooms in the vertical direction (Z axis).
    """

    @staticmethod
    def _zoom_interval(base: SimpleInterval, target: SimpleInterval) -> SimpleInterval:
        """
        Zoom 'base' interval by the percentage interval 'target' (0..1),
        preserving the base's boundary styles.

        :param base: The base interval to be zoomed in.
        :param target: The target interval defining the zoom percentage (0..1).
        :return: A new SimpleInterval representing the zoomed-in interval.
        """
        span = base.upper - base.lower
        new_lower = base.lower + span * target.lower
        new_upper = base.lower + span * target.upper
        return SimpleInterval(new_lower, new_upper, base.left, base.right)

    def _apply_zoom(self, simple_event: SimpleEvent) -> SimpleEvent:
        """
        Apply zooms in order and return the resulting intervals.

        :param simple_event: The event to zoom in.
        :return: A SimpleEvent containing the resulting intervals after applying all zooms.
        """
        simple_events = [
            self._apply_zoom_in_one_direction(
                axis,
                assignment.simple_sets[0],
            )
            for axis, assignment in simple_event.items()
        ]

        if not simple_events:
            return SimpleEvent()

        return reduce(or_, simple_events)

    def _apply_zoom_in_one_direction(
        self, axis: Continuous, current_interval: SimpleInterval
    ) -> SimpleEvent:
        """
        Apply zooms in one direction (Y, horizontal or Z, vertical) in order and return the resulting interval.

        :param axis: The axis to zoom in (SpatialVariables.y or SpatialVariables.z).
        :param current_interval: The current interval to zoom in.
        :return: A SimpleEvent containing the resulting interval after applying all zooms in the specified direction.
        """
        if axis == SpatialVariables.y.value:
            directions = self.horizontal_direction_chain
        elif axis == SpatialVariables.z.value:
            directions = self.vertical_direction_chain
        else:
            raise NotImplementedError

        for step in directions:
            current_interval = self._zoom_interval(current_interval, step)

        return SimpleEvent({axis: current_interval})

    def sample_point_from_event(self, event: Event) -> Tuple[float, float]:
        """
        Sample a 2D point from the given event by applying the zooms defined in the semantic position description.

        :param event: The event to sample from.
        :return: A sampled 2D point as a tuple (y, z).
        """
        simple_event = self._apply_zoom(event.bounding_box())
        event_circuit = uniform_measure_of_simple_event(simple_event)
        return event_circuit.sample(amount=1)[0]
