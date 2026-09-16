"""
What segmind raises when it is asked for something it cannot give.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from krrood.exceptions import DataclassException


class OptionalDependency(StrEnum):
    """
    A package segmind uses only for a feature installed as one of its extras.
    """

    FLASK = "flask"
    """
    Serves the live event dashboard (segmind's ``dashboard`` extra).
    """


@dataclass
class DashboardNeedsFlask(DataclassException):
    """
    Raised when the live event dashboard is loaded without flask installed.
    """

    def error_message(self) -> str:
        return (
            f"The live event dashboard needs {OptionalDependency.FLASK}, which is not "
            f"installed."
        )

    def suggest_correction(self) -> str:
        return "Install segmind with its dashboard extra: pip install 'segmind[dashboard]'."
