"""
A module that cannot be imported, whose class and function are undocumented.
"""

from __future__ import annotations

raise RuntimeError("This module cannot be imported.")


class Undocumented:
    """
    A documented class.
    """

    def undocumented_method(self) -> None:
        pass


def undocumented_function() -> None:
    pass
