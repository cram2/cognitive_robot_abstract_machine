from __future__ import annotations


def documented(first: int, second: int) -> int:
    """
    Add two numbers.

    Usage::

        documented(1, 2)
        documented(3, 4)
        documented(5, 6)

    :param first: The first number.
    :param second: The second number.
    :return: Their sum.
    :raises ValueError: Never, in fact.
    """
    return first + second
