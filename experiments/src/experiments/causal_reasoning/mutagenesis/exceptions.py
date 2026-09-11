"""
Exceptions for the Mutagenesis causal-reasoning experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class MutagenesisDatasetUnavailableError(DataclassException):
    """
    Raised when the CTU relational-dataset repository cannot be reached.
    """

    reason: str
    """
    The underlying database error's message.
    """

    def error_message(self) -> str:
        return f"Could not reach the CTU Mutagenesis database: {self.reason}"

    def suggest_correction(self) -> str:
        return (
            "Check network access to relational.fel.cvut.cz, or skip tests that "
            "require it."
        )
