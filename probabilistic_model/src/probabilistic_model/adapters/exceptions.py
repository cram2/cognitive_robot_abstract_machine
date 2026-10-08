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
class NoConversionPathError(DataclassException):
    """
    Raised when no chain of converters leads from the representation of a circuit to
    the requested representation.
    """

    circuit_type: Type = field(kw_only=True)
    """
    The type of the circuit that was to be converted.
    """

    representation: Type = field(kw_only=True)
    """
    The requested representation.
    """

    def error_message(self) -> str:
        return (
            f"No chain of converters converts a {self.circuit_type.__name__} into a "
            f"{self.representation.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "Add a converter of whole circuits to CircuitRepresentations that connects "
            "the two representations."
        )
