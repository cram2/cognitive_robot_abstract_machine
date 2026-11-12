"""
A module that cannot be imported, with a dataclass and methods taking their fields and
parameters positionally.
"""

from __future__ import annotations

from dataclasses import dataclass

raise RuntimeError("This module cannot be imported.")


@dataclass
class PositionalFields:
    """
    A dataclass whose fields can be passed positionally.
    """

    value: int
    """
    A field.
    """


class WithoutBases:
    """
    A class without bases, whose methods override nothing.
    """

    def positional(self, value: int) -> int:
        """
        A method taking a parameter positionally.
        """
        return value


class WithBases(WithoutBases):
    """
    A class with a base that cannot be looked up.
    """

    def positional(self, value: int) -> int:
        """
        A method that might override one an installed base class calls positionally.
        """
        return value
