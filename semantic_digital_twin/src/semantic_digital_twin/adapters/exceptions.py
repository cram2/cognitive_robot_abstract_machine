from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.exceptions import ParsingError


@dataclass
class UnsupportedFBXAxis(ParsingError):
    """
    Raised when the global settings of an FBX file name an axis that does not exist.
    """

    axis_index: int = field(kw_only=True)
    """
    The index of the axis.
    """

    sign: int = field(kw_only=True)
    """
    The direction of the axis.
    """

    def error_message(self) -> str:
        return f"FBX axis index {self.axis_index} with sign {self.sign} names no coordinate axis."

    def suggest_correction(self) -> str:
        return "Use an axis index of 0, 1 or 2 and a sign of 1 or -1."


@dataclass
class UnexpectedArrayShape(ParsingError):
    """
    Raised when an array does not have the shape its use requires.
    """

    array_name: str = field(kw_only=True)
    """
    The name of the array.
    """

    expected_shape: str = field(kw_only=True)
    """
    The shape the array must have, with letters standing for any size.
    """

    actual_shape: list[int] = field(kw_only=True)
    """
    The shape the array has.
    """

    def error_message(self) -> str:
        return f"{self.array_name} has shape {tuple(self.actual_shape)}, but must have shape {self.expected_shape}."

    def suggest_correction(self) -> str:
        return f"Pass {self.array_name} with shape {self.expected_shape}."
