"""
Exceptions raised by RoboKudo annotators.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from robokudo.exceptions import RoboKudoError


# %% Collection reader placement
@dataclass
class InvalidCollectionReaderParent(RoboKudoError):
    """
    A collection reader has no parent sequence to initialize.
    """

    def error_message(self) -> str:
        """
        Identify the invalid reader placement.
        """
        return "The collection reader's parent is not a behavior-tree sequence."

    def suggest_correction(self) -> str:
        """
        Describe valid placement of the reader.
        """
        return "Add the collection reader to a pipeline or sequence before ticking it."


# %% Analysis contracts
@dataclass
class InvalidAnalysisData(RoboKudoError):
    """
    Analysis data does not satisfy its required type.
    """

    expected_type: type
    """
    Type required by the analysis operation.
    """
    actual_type: type
    """
    Type of the supplied data.
    """

    def error_message(self) -> str:
        return f"Expected {self.expected_type.__name__}, received {self.actual_type.__name__}."

    def suggest_correction(self) -> str:
        return "Provide analysis data of the required type."


@dataclass
class EmptyAnalysisScope(RoboKudoError, ValueError):
    """
    An analysis scope selects no data.
    """

    analysis_scope: object
    """
    Scope that selected no analysis data.
    """

    def error_message(self) -> str:
        return f"Analysis scope selected no data: {self.analysis_scope!r}."

    def suggest_correction(self) -> str:
        return "Select an available CAS view or annotation type."


# %% model configuration


@dataclass
class NotAPoseModelException(RoboKudoError, ValueError):
    """
    Raised when an annotator that reads body keypoints loaded a model reporting none.
    """

    annotator_name: str
    """
    Name of the annotator whose model reports no keypoints.
    """

    model: str
    """
    The model the annotator loaded.
    """

    def error_message(self) -> str:
        return f"{self.annotator_name} loaded {self.model}, which reports no body keypoints."

    def suggest_correction(self) -> str:
        return "set 'pretrained_model' to a YOLO pose model, e.g. 'yolov8n-pose.pt'."
