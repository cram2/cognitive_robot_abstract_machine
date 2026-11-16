"""
A module whose names are verb phrases or alternative constructors and conversions.
"""

from __future__ import annotations


class Pipe:
    """
    A class whose method names need no class name in front of them.
    """

    @classmethod
    def from_json(cls, data: dict) -> Pipe:
        """
        An alternative constructor named for what it reads.
        """
        return cls()

    def to_json(self) -> dict:
        """
        A conversion named for what it produces.
        """
        return {}

    def assemble(self) -> None:
        """
        A verb whose letters begin like a preposition.
        """

    def offer_discount(self) -> None:
        """
        A verb phrase whose first letters spell a preposition.
        """
