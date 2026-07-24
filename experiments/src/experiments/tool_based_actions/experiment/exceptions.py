"""
Custom exceptions of the tool-based action experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import List, Tuple


@dataclass
class InvalidSpawnRegion(DataclassException):
    """
    Raised when a spawn region is constructed with bounds that contain no points.
    """

    minimum_x: float
    """
    Lower X bound of the invalid region.
    """

    maximum_x: float
    """
    Upper X bound of the invalid region.
    """

    minimum_y: float
    """
    Lower Y bound of the invalid region.
    """

    maximum_y: float
    """
    Upper Y bound of the invalid region.
    """

    def error_message(self) -> str:
        return (
            f"Spawn region bounds x=[{self.minimum_x}, {self.maximum_x}], "
            f"y=[{self.minimum_y}, {self.maximum_y}] contain no points."
        )

    def suggest_correction(self) -> str:
        return "Ensure every maximum bound is greater than its minimum bound."


@dataclass
class MissingSpawnSurfaces(DataclassException):
    """
    Raised when none of the configured spawn surfaces exist in the world.
    """

    surface_names: Tuple[str, ...]
    """
    The configured surface names none of which were found.
    """

    def error_message(self) -> str:
        return (
            f"None of the configured spawn surfaces {self.surface_names} exist in the "
            "world."
        )

    def suggest_correction(self) -> str:
        return "Check the surface names against the environment."


@dataclass
class SpawnRegionExhausted(DataclassException):
    """
    Raised when the spawn surfaces cannot hold the requested targets.
    """

    surface_names: List[str]
    """
    Names of the surfaces placement was attempted on.
    """

    count: int
    """
    The number of targets that could not be placed.
    """

    def error_message(self) -> str:
        return (
            f"Could not place {self.count} targets on the surfaces "
            f"{self.surface_names}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class IncompatibleResultRecord(DataclassException):
    """
    Raised when a stored result line does not match the current result schema, typically
    because the results file was written by an older version of the experiment.
    """

    missing_fields: List[str]
    """
    Fields the current schema requires that the stored record does not carry.
    """

    unexpected_fields: List[str]
    """
    Fields the stored record carries that the current schema does not know.
    """

    def error_message(self) -> str:
        return (
            f"Result record does not match the current TargetResult schema: missing "
            f"fields {self.missing_fields}, unexpected fields {self.unexpected_fields}."
        )

    def suggest_correction(self) -> str:
        return "Archive or delete the results file to start a fresh campaign."
