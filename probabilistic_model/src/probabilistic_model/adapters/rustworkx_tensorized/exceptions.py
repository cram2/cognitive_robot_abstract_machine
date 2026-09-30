from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException
from typing_extensions import Type


@dataclass
class CannotConvertError(DataclassException):
    """
    Raised when no converter handles an object.
    """

    data_type: Type = field(kw_only=True)
    """
    The type of the object that could not be converted.
    """

    def error_message(self) -> str:
        return f"No converter handles {self.data_type.__name__}."

    def suggest_correction(self) -> str:
        return (
            "Subclass the converter base class for the type, binding it as the input "
            "type."
        )


@dataclass
class NotExactlyOneRootError(DataclassException):
    """
    Raised when the converted layers of a circuit do not contain exactly one layer
    that holds the root of the circuit.
    """

    number_of_roots: int = field(kw_only=True)
    """
    The number of converted layers whose first unit is the root.
    """

    def error_message(self) -> str:
        return f"Expected exactly one layer holding the root, found {self.number_of_roots}."

    def suggest_correction(self) -> str:
        return "Convert a circuit that has exactly one root."
