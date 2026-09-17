"""
Exceptions for the Mutagenesis causal-reasoning experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import List


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


@dataclass
class FlatTableSchemaMismatchError(DataclassException):
    """
    Raised when a query constrains a flat-table model on variables the table it was
    fitted on never had, such as one atom's element: the table carries a molecule's own
    scalars and its aggregation counts, not its atoms or bonds.
    """

    missing_variable_names: List[str]
    """
    The constrained variable names the fitted table does not carry.
    """

    def error_message(self) -> str:
        return (
            "The flat-table model has no column for the queried variables "
            f"{self.missing_variable_names}."
        )

    def suggest_correction(self) -> str:
        return (
            "Ask about the molecule's own scalars and aggregation counts, or use the "
            "relational pipeline, which grounds a circuit over the queried atoms and "
            "bonds."
        )


@dataclass
class OneCausePerQueryError(DataclassException):
    """
    Raised when a query marks more than one variable as its cause: each pipeline fits
    one support-deterministic model per cause variable, which cannot serve two at once.
    """

    cause_names: List[str]
    """
    The names of the variables the query marked.
    """

    def error_message(self) -> str:
        return f"The query marks {self.cause_names} as causes; only one is supported."

    def suggest_correction(self) -> str:
        return "Ask one query per candidate cause."


@dataclass
class PipelineNotFittedError(DataclassException):
    """
    Raised when a pipeline is asked for a model before it was fitted.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    def error_message(self) -> str:
        return f"The {self.pipeline_name} pipeline has not been fitted yet."

    def suggest_correction(self) -> str:
        return "Call fit(molecules) first."
