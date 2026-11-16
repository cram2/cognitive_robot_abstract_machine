"""
A module whose function and methods begin their names with prepositions.
"""

from __future__ import annotations


class Pipe:
    """
    A class whose methods begin their names with prepositions.
    """

    @classmethod
    def for_hot_water(cls) -> Pipe:
        """
        A factory whose name the class has to finish.
        """
        return cls()

    def as_hose(self) -> Pipe:
        """
        A conversion whose name the class has to finish.
        """
        return self

    def _with_insulation(self) -> Pipe:
        """
        A private method whose name begins with a preposition after its underscore.
        """
        return self


def of_steel() -> Pipe:
    """
    A function whose name begins with a preposition.
    """
    return Pipe()
