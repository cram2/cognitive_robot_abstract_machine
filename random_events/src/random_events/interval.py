from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, InitVar
from typing import Dict, Any, Tuple, Union, Optional

import random_events_lib as rl
from typing_extensions import Self, Iterable

from random_events import sigma_algebra


class Bound(enum.IntEnum):
    """
    Enumerates the possible bounds for an interval.
    """

    CLOSED = 0
    """
    Represents a closed bound, i. e. the element is included from the interval.
    """

    OPEN = 1
    """
    Represents an open bound, i. e. the element is excluded in the interval.
    """


@dataclass(eq=False)
class SimpleInterval(sigma_algebra.AbstractSimpleSet):
    """
    A simple interval.
    A simple interval is the convex hull of two points.

    .. attention::
        Use :py:func:`from_data` class method to create a simple interval from a dictionary, do not use the constructor directly.
    """

    cpp_object: rl.SimpleInterval = field(
        default_factory=lambda: rl.SimpleInterval(
            0.0, 0.0, Bound.OPEN.value, Bound.OPEN.value
        )
    )

    @classmethod
    def from_data(
        cls,
        lower: float = 0,
        upper: float = 0,
        left: Bound = Bound.OPEN,
        right: Bound = Bound.OPEN,
    ) -> Self:
        instance = cls.__new__(cls)
        instance.cpp_object = rl.SimpleInterval(lower, upper, left.value, right.value)
        return instance

    @property
    def lower(self) -> float:
        """
        :return: The lower bound of the interval.
        """
        return self.cpp_object.lower

    @lower.setter
    def lower(self, value: float):
        self.cpp_object.lower = value

    @property
    def upper(self) -> float:
        """
        :return: The upper bound of the interval.
        """
        return self.cpp_object.upper

    @upper.setter
    def upper(self, value: float):
        self.cpp_object.upper = value

    @property
    def left(self) -> Bound:
        """
        :return: The bound type of the lower bound.
        """
        return Bound(self.cpp_object.left.value)

    @left.setter
    def left(self, value: Bound):
        self.cpp_object.left = value.value

    @property
    def right(self) -> Bound:
        """
        :return: The bound type of the upper bound.
        """
        return Bound(self.cpp_object.right.value)

    @right.setter
    def right(self, value: Bound):
        self.cpp_object.right = value.value

    @classmethod
    def _from_cpp(cls, cpp_object: rl.SimpleInterval) -> Self:
        return cls.from_data(
            cpp_object.lower, cpp_object.upper, cpp_object.left, cpp_object.right
        )

    def as_composite_set(self) -> Interval:
        return Interval.from_simple_sets(self)

    def is_singleton(self) -> bool:
        """
        # TODO: fix this when random_events_lib is fixed
        :return: True if the interval is a singleton (contains only one value), False otherwise.
        """
        return (
            self.lower == self.upper
            and self.left == Bound.CLOSED
            and self.right == Bound.CLOSED
        )

    def contains(self, item: float) -> bool:
        return (
            self.lower < item < self.upper
            or (self.lower == item and self.left == Bound.CLOSED)
            or (self.upper == item and self.right == Bound.CLOSED)
        )

    def non_empty_to_string(self) -> str:
        left_bracket = "[" if self.left == Bound.CLOSED else "("
        right_bracket = "]" if self.right == Bound.CLOSED else ")"
        return f"{left_bracket}{self.lower}, {self.upper}{right_bracket}"

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "lower": self.lower,
            "upper": self.upper,
            "left": self.left.name,
            "right": self.right.name,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls.from_data(
            data["lower"], data["upper"], Bound[data["left"]], Bound[data["right"]]
        )

    def center(self) -> float:
        """
        :return: The center point of the interval
        """
        return (self.lower + self.upper) / 2

    def contained_integers(self) -> Iterable[int]:
        """
        :return: Yield integers contained in the interval
        """
        rounded_lower = math.ceil(self.lower)
        if rounded_lower == self.lower and self.left == Bound.OPEN:
            rounded_lower += 1

        rounded_upper = math.floor(self.upper)
        if rounded_upper == self.upper and self.right == Bound.OPEN:
            rounded_upper -= 1

        yield from range(rounded_lower, rounded_upper + 1)

    def __deepcopy__(self):
        return SimpleInterval.from_data(self.lower, self.upper, self.left, self.right)


@dataclass(eq=False)
class Interval(sigma_algebra.AbstractCompositeSet):
    """
    Represents an interval.

    An interval is a union of simple intervals.

    A simplified interval is an interval where adjacent simple intervals are merged.

    .. attention::
        Use :py:func:`from_simple_sets` class method to create an interval from a list of simple intervals, do not use the constructor directly.
    """

    cpp_object: rl.Interval = field(default_factory=lambda: rl.Interval())
    simple_set_example: SimpleInterval = field(init=False)

    @classmethod
    def from_simple_sets(
        cls, *simple_sets: Union[Tuple[SimpleInterval, ...], SimpleInterval]
    ) -> Self:
        """
        Create an interval from a list of simple intervals.
        :param simple_sets: The simple intervals that make up the interval.
        :return: The interval.
        """
        instance = cls.__new__(cls)
        instance.simple_set_example = SimpleInterval.from_data()
        if not isinstance(simple_sets, tuple):
            simple_sets = (simple_sets,)
        instance.cpp_object = rl.Interval(
            {simple_set.cpp_object for simple_set in simple_sets}
        )
        return instance

    @classmethod
    def _from_cpp(cls, cpp_object: rl.Interval) -> Self:
        return cls.from_simple_sets(
            *[
                SimpleInterval._from_cpp(cpp_simple_interval)
                for cpp_simple_interval in cpp_object.simple_sets
            ]
        )

    def is_singleton(self):
        """
        :return: True if the interval is a singleton (contains only one value), False otherwise.
        """
        return len(self.simple_sets) == 1 and self.simple_sets[0].is_singleton()

    def contained_integers(self) -> Iterable[int]:
        """
        :return: Yield integers contained in the interval
        """
        for simple_set in sorted(self.simple_sets):
            yield from simple_set.contained_integers()


def open(left: float, right: float) -> Interval:
    """
    Creates an open interval.

    :param left: The left bound of the interval.
    :param right: The right bound of the interval.
    :return: The open interval.
    """
    return Interval._from_cpp(rl.open(left, right))


def closed(left: float, right: float) -> Interval:
    """
    Creates a closed interval.

    :param left: The left bound of the interval.
    :param right: The right bound of the interval.
    :return: The closed interval.
    """
    return Interval._from_cpp(rl.closed(left, right))


def open_closed(left: float, right: float) -> Interval:
    """
    Creates an open-closed interval.

    :param left: The left bound of the interval.
    :param right: The right bound of the interval.
    :return: The open-closed interval.
    """
    return Interval._from_cpp(rl.open_closed(left, right))


def closed_open(left: float, right: float) -> Interval:
    """
    Creates a closed-open interval.

    :param left: The left bound of the interval.
    :param right: The right bound of the interval.
    :return: The closed-open interval.
    """
    return Interval._from_cpp(rl.closed_open(left, right))


def singleton(value: float) -> Interval:
    """
    Creates a singleton interval.

    :param value: The value of the interval.
    :return: The singleton interval.
    """
    return Interval._from_cpp(rl.singleton(value))


def reals() -> Interval:
    """
    Creates the set of real numbers.

    :return: The set of real numbers.
    """
    return Interval._from_cpp(rl.reals())
