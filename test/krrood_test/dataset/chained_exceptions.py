"""
Exceptions that are persisted together with the exceptions they were chained to.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


# %% chained exceptions


@dataclass
class ExceptionWithOwnField(DataclassException):
    """
    An exception with a field of its own next to the inherited chaining fields.
    """

    offending_value: int
    """
    The value that caused the error.
    """

    def error_message(self) -> str:
        return f"The value {self.offending_value} is not allowed."

    def suggest_correction(self) -> str:
        return ""
