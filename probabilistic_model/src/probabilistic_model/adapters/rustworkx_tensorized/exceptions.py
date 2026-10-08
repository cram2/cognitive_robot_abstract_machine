from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException


@dataclass
class NotExactlyOneRootError(DataclassException):
    """
    Raised when the converted layers of a circuit do not contain exactly one layer that
    holds the root of the circuit.
    """

    number_of_roots: int = field(kw_only=True)
    """
    The number of converted layers whose first unit is the root.
    """

    def error_message(self) -> str:
        return f"Expected exactly one layer holding the root, found {self.number_of_roots}."

    def suggest_correction(self) -> str:
        return "Convert a circuit that has exactly one root."
