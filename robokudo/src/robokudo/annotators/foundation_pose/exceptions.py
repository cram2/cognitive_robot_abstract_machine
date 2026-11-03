"""
Exceptions raised by the FoundationPose annotator.
"""

from __future__ import annotations

from dataclasses import dataclass

from robokudo.exceptions import RoboKudoError


# %% CAD model configuration


@dataclass
class DuplicateCadModelClass(RoboKudoError, ValueError):
    """
    Raised when more than one CAD model is configured for an object class.
    """

    class_name: str
    """
    The object class with more than one CAD model.
    """

    def error_message(self) -> str:
        return (
            f"More than one CAD model is configured for the class '{self.class_name}'."
        )

    def suggest_correction(self) -> str:
        return "Configure exactly one CAD model per class name."
