"""Error handling utilities for RoboKudo.

This module provides error handling utilities for behavior trees.

:module: error_handling
:synopsis: Error handling utilities for behavior trees
:moduleauthor: RoboKudo Team
"""

# import logging
import traceback
from functools import wraps

from py_trees.blackboard import Blackboard
from py_trees.common import Status
from typing_extensions import Callable, Union, Any, Optional

from robokudo.identifier import BBIdentifier

# rk_logger = logging.getLogger(robokudo.defs.PACKAGE_NAME)


def raise_to_blackboard(exception: Optional[Exception]) -> None:
    """Store exception in blackboard.

    :param exception: Exception to store or None to clear

    :Example:

    .. code-block:: python

        try:
            # Some code that might raise
            pass
        except Exception as e:
            raise_to_blackboard(e)
    """
    Blackboard().set(BBIdentifier.BLACKBOARD_EXCEPTION_NAME, exception)


def has_blackboard_exception() -> bool:
    """Check if blackboard contains an exception.

    :return: True if exception exists and is not None

    :Example:

    .. code-block:: python

        if has_blackboard_exception():
            # Handle exception
            pass
    """
    return (
        Blackboard().exists(BBIdentifier.BLACKBOARD_EXCEPTION_NAME)
        and get_blackboard_exception() is not None
    )


def get_blackboard_exception() -> Optional[Exception]:
    """Retrieve exception from blackboard.

    :return: Stored exception or None if not found
    :rtype: Exception or None

    :Example:

    .. code-block:: python

        exc = get_blackboard_exception()
        if exc is not None:
            # Handle exception
            pass
    """
    return Blackboard().get(BBIdentifier.BLACKBOARD_EXCEPTION_NAME)


def clear_blackboard_exception() -> None:
    """Clear any stored exception from blackboard.

    :Example:

    .. code-block:: python

        clear_blackboard_exception()
        assert not has_blackboard_exception()
    """
    raise_to_blackboard(None)


def catch_and_raise_to_blackboard(
    function: Callable[[...], Any],
) -> Union[Status, Callable]:
    """Catch and store exceptions in blackboard.

    This decorator is used to catch exceptions in Annotators to place them into the blackboard. Mostly useful in
    analysis-engines that have a query-interface which should return a failure back to the action-server caller
    if one of the Annotators yields an exception.

    :param function: The update or compute method of the desired Annotator
    :return: Exception that has been catched or found in the Blackboard or wrapped function that handles exceptions

    :Example:

    .. code-block:: python

        @catch_and_raise_to_blackboard
        def update(self) -> Status:
            # Method code here
            pass

    .. warning::
        Should not be used on ThreadedAnnotators
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        # This wrapper should NOT be used on ThreadedAnnotators!  See docstring.
        # assert (not (isinstance(args[0], ThreadedAnnotator)))
        if has_blackboard_exception():
            return Status.FAILURE
        try:
            r = function(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise_to_blackboard(e)
            return Status.FAILURE
        return r

    return wrapper
