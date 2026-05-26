from __future__ import annotations

import functools
import inspect
import time
import types
from collections import defaultdict
from functools import wraps

import numpy as np
from random_events.interval import SimpleInterval, Interval
from typing_extensions import Type
import datetime

from random_events.product_algebra import Event
from random_events.variable import Variable, Continuous


def simple_interval_as_array(interval: SimpleInterval) -> np.ndarray:
    """
    Convert a simple interval to a numpy array.
    :param interval:  The interval
    :return:  [lower, upper] as numpy array
    """
    return np.array([interval.lower, interval.upper])


def interval_as_array(interval: Interval) -> np.ndarray:
    """
    Convert an interval to a numpy array.
    The resulting array has shape (n, 2) where n is the number of simple intervals in the interval.
    The first column contains the lower bounds and the second column the upper bounds of the simple intervals.
    :param interval: The interval
    :return:  as numpy array
    """
    return np.array(
        [
            simple_interval_as_array(simple_interval)
            for simple_interval in interval.simple_sets
        ]
    )


class MissingDict(defaultdict):
    """
    A defaultdict that returns the default value when the key is missing and does **not** add the key to the dict.
    """

    def __missing__(self, key):
        return self.default_factory()


def timeit(func):
    """
    Decorator to measure the time a function takes to execute.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()

        total_time = end_time - start_time
        total_time = datetime.timedelta(microseconds=total_time / 1000)
        return result, total_time

    return timeit_wrapper


def timeit_print(func):

    @wraps(func)
    def timeit_print_wrapper(*args, **kwargs):
        self = args[0]
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()

        total_time = end_time - start_time
        total_time = datetime.timedelta(microseconds=total_time / 1000)
        print(f"{func.__qualname__} took : {total_time}")
        return result

    return timeit_print_wrapper


def neighbouring_points(point: float) -> np.array:
    """
    Embed the point in an array with the next left and next right point.

    :param point: The point.
    :return: The point and its two neighbours
    """
    return np.array([np.nextafter(point, -np.inf), point, np.nextafter(point, np.inf)])


def event_compatible_for_truncation_with_singletons(event: Event):
    """
    Check if the event is compatible for truncation with singletons.
    It is compatible if for each variable, either all intervals are singletons or all intervals are not singletons.
    :param event: The event to check.
    :return: True if the event is compatible, False otherwise.
    """
    intervals_per_dimensions = defaultdict(list)

    for variable in event.variables:
        variable: Variable
        if not isinstance(variable, Continuous):
            continue

        for simple_event in event.simple_sets:
            # collect all simple intervals for this variable across the composite event
            intervals_per_dimensions[variable].extend(
                simple_event[variable].simple_sets
            )

    for variable, intervals in intervals_per_dimensions.items():
        if all(interval.is_singleton() for interval in intervals):
            continue
        elif all(not interval.is_singleton() for interval in intervals):
            continue
        else:
            return False
    return True
