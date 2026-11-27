from __future__ import annotations


def within_limit(first: int, second: int) -> int:
    """
    Add two numbers.

    >>> within_limit(1, 2)
    3
    >>> within_limit(
    ...     3, 4
    ... )
    7

    :param first: The first number.
    :param second: The second number.
    :return: Their sum.
    """
    return first + second


def over_limit(first: int, second: int) -> int:
    """
    Add two numbers.

    The sum is an integer.
    It never overflows.

    >>> over_limit(1, 2)
    3

    The order does not matter.

    Neither does the sign.
    """
    return first + second
