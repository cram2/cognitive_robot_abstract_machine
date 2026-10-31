"""
Failures at the external grasp-planning boundary.
"""

from dataclasses import dataclass
from krrood.exceptions import DataclassException

# %% Simox failures


@dataclass
class SimoxPlanningFailed(DataclassException):
    """
    The external planner did not supply candidates.
    """

    reason: str
    """Reported failure."""

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return "Check the service, shared model paths and planning parameters."


@dataclass
class InvalidSimoxResponse(DataclassException):
    """
    The service returned inconsistent candidate data.
    """

    reason: str
    """Violated response contract."""

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return "Check the service response lengths, finite values and pose units."
