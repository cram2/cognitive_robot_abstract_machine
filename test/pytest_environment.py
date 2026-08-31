"""
What the processes of a test run read about the run they belong to.
"""

from __future__ import annotations

from enum import StrEnum


class PytestEnvironmentVariable(StrEnum):
    """
    Environment variables the processes of a run read.
    """

    XDIST_WORKER = "PYTEST_XDIST_WORKER"
    """
    Names the xdist worker a process is; absent in the controller.
    """

    ORM_BUILD = "CRAM_ORM_BUILD"
    """
    Names when the run builds the ORM interfaces, for runs that state no ``--orm-build``
    on their command line; absent, a run builds what is outdated.
    """
