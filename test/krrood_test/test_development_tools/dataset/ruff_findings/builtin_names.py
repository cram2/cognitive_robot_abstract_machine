"""
A module naming a variable, a parameter and a method after builtins.
"""

from __future__ import annotations


def describe(type: str) -> str:  # reported
    """
    A parameter named after a builtin.
    """
    id = len(type)  # reported
    return f"{type}{id}"


class Registry:
    """
    A class whose method takes a builtin's name, then calls the builtin in its body.
    """

    def list(self) -> None:
        """
        A method named after a builtin.
        """

    empty = list()  # reported
    """
    Meant to be an empty list, but calls the method above.
    """
