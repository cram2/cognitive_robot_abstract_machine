"""
Decorator utilities for RoboKudo annotators.
"""

import logging
import timeit
from collections import defaultdict
from functools import wraps

from typing_extensions import Callable, Any
from robokudo.annotators.core import BaseAnnotator


def timer_decorator(func: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Log execution time of decorated function.

    :param func: Function to be timed
    :return: Wrapped function that logs execution time

    :Example:

    .. code-block:: python

        @timer_decorator
        def my_function():
            # Function code here
            pass

    .. note::
        Logs at info level to the logger of the module that defines the decorated function.
    """
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Call the decorated function and log how long it took.
        """
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        processing_time = timeit.default_timer() - start_time
        logger.info(
            f"Function '{func.__qualname__}' took {processing_time:.3f} seconds to execute."
        )
        return result

    return wrapper


def record_time(func: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Record execution time of annotator method.

    :param func: Annotator method to time
    :return: Wrapped method that records execution time

    :Example:

    .. code-block:: python

        @record_time
        def compute(self):
            # Method code here
            pass

    .. note::
        Only works on annotator instance methods
    """
    function_name = func.__name__

    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, "_times"):
            setattr(self, "_times", defaultdict(list))
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        processing_time = end_time - start_time
        # self._times[function_name].append(processing_time)
        self._times[function_name] = processing_time
        return result

    return wrapper


def publish_variables(func: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Publish annotator variables after method execution.

    :param func: Annotator method to wrap
    :return: Wrapped method that publishes variables
    :raises AssertionError: If not used on BaseAnnotator instance method

    :Example:

    .. code-block:: python

        @publish_variables
        def update(self):
            # Method code here
            pass

    .. note::
        Only works on BaseAnnotator instance methods
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert isinstance(args[0], BaseAnnotator)
        args[0].publish_variables()
        return result

    return wrapper
