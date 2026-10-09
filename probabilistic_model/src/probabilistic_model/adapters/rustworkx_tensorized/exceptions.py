from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
        Unit,
    )


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


@dataclass
class UnitNotConvertedError(DataclassException):
    """
    Raised when a unit is looked up that no layer of a converted circuit was created
    from.
    """

    unit: Unit
    """
    The unit.
    """

    def error_message(self) -> str:
        return f"No layer of the converted circuit was created from {self.unit}."

    def suggest_correction(self) -> str:
        return "Look up a unit of the rustworkx circuit that was converted."
