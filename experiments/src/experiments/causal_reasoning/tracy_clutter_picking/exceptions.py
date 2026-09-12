"""
Exceptions of the Tracy clutter-picking experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import List


@dataclass
class FlatTableSchemaMismatchError(DataclassException):
    """
    Raised when a query asks a flat-table model about variables the table it was fitted
    on never had -- for instance a scene with more, or fewer, neighbours than the table
    was flattened with.
    """

    missing_variable_names: List[str]
    """
    The queried variable names the fitted table does not carry.
    """

    def error_message(self) -> str:
        return (
            "The flat-table model has no column for the queried variables "
            f"{self.missing_variable_names}."
        )

    def suggest_correction(self) -> str:
        return (
            "Query the model with exactly the neighbour count it was flattened with, or "
            "use the relational pipeline, which grounds a circuit for any count."
        )


@dataclass
class EpisodePlanningFailedError(DataclassException):
    """
    Raised when the motion planner could not produce a reach for one of a pick attempt's
    own motions, so the attempt never got to test the grasp itself.
    """

    reason: str
    """
    The planner's own failure message.
    """

    def error_message(self) -> str:
        return f"A pick attempt's motion could not be planned: {self.reason}"

    def suggest_correction(self) -> str:
        return "Sample the attempt again; its layout is dropped, not recorded."


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
        return "Call fit(scenes) first."
