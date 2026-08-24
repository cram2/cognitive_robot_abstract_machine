"""
What pytest tells the processes of a run about themselves.
"""

from __future__ import annotations

from enum import StrEnum


class PytestEnvironmentVariable(StrEnum):
    """
    Environment variables pytest sets for the processes of a run.
    """

    XDIST_WORKER = "PYTEST_XDIST_WORKER"
    """
    Names the xdist worker a process is; absent in the controller.
    """
