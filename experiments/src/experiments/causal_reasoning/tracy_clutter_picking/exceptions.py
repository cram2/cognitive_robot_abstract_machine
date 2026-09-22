"""
Exceptions of the Tracy clutter-picking experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import List


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
class UnevenClutterError(DataclassException):
    """
    Raised when a set of recorded attempts does not hold the same number of neighbours
    in every attempt, so no one question about the recorded clutter size fits them all.
    """

    neighbour_counts: List[int]
    """
    The numbers of neighbours found, ascending.
    """

    def error_message(self) -> str:
        return (
            "The recorded attempts hold different numbers of neighbours: "
            f"{self.neighbour_counts}"
        )

    def suggest_correction(self) -> str:
        return "Record every attempt on a clutter of the same size."


@dataclass
class UnreadableQuestionError(DataclassException):
    """
    Raised when a question is scored against the closed-form mechanism that the
    mechanism has no reading of, so no true interventional probability exists for it.
    """

    question: str
    """
    The question's class.
    """

    def error_message(self) -> str:
        return f"The mechanism has no reading of {self.question}"

    def suggest_correction(self) -> str:
        return "Score only the questions whose cause the mechanism forces."
