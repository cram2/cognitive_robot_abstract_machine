"""
The marker for methods that a caller outside the repository's control calls with
positional arguments.
"""

from __future__ import annotations

from typing_extensions import Callable, TypeVar

MarkedMethod = TypeVar("MarkedMethod", bound=Callable)
"""
The type of the marked method, kept as it is.
"""


def called_positionally(method: MarkedMethod) -> MarkedMethod:
    """
    Marks *method* as taking its parameters positionally, such as a method applied as a
    decorator, which Python calls with the decorated function. The keyword-only parameter
    check of ``krrood check`` skips marked methods.

    :param method: The method to mark.
    :return: *method* unchanged.
    """
    return method
