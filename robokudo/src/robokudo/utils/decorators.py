"""Decorator utilities for RoboKudo annotators."""

import timeit
from collections import defaultdict

from typing_extensions import Callable, Any
from robokudo.annotators.core import BaseAnnotator


def timer_decorator(func: Callable[[...], Any]) -> Callable[[...], Any]:
    """Log execution time of decorated function.

    :param func: Function to be timed
    :return: Wrapped function that logs execution time

    :Example:

    .. code-block:: python

        @timer_decorator
        def my_function():
            # Function code here
            pass

    .. note::
        Uses rk_logger if available, otherwise prints to stdout
    """

    def wrapper(*args, **kwargs):
        logger = None
        class_name = None

        if args and hasattr(args[0], "rk_logger"):
            logger = args[0].rk_logger
            class_name = args[0].__class__
        elif "self" in kwargs and hasattr(kwargs["self"], "rk_logger"):
            logger = kwargs["self"].rk_logger
            class_name = kwargs["self"].__class__

        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        processing_time = end_time - start_time

        if class_name:
            log_msg = f"Function '{class_name}.{func.__name__}' took {processing_time:.3f} seconds to execute."
        else:
            log_msg = f"Function '{func.__name__}' took {processing_time:.3f} seconds to execute."

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        return result

    return wrapper


def record_time(func: Callable[[...], Any]) -> Callable[[...], Any]:
    """Record execution time of annotator method.

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
    """Publish annotator variables after method execution.

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
