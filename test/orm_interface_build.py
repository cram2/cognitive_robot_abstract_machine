"""
Building the ORM interfaces a test run needs.

Every package's ``ormatic_interface.py`` is generated rather than tracked, so a checkout
holds none until something builds them. Only the packages whose tests read a mapped
datastructure call this; the rest never pay for a build they would not read.
"""

from __future__ import annotations

import os
from functools import cache

from cognitive_robot_abstract_machine.orm_interfaces import WORKSPACE_ORM_INTERFACES

from .pytest_environment import PytestEnvironmentVariable


@cache
def regenerate_orm_interfaces() -> bool:
    """
    Build the ORM interfaces of this checkout, once per process and never on a worker.

    The controller imports the conftests before it starts any worker, so building there
    leaves the interfaces on disk by the time a worker imports a mapped datastructure.
    Letting the workers build too would set several processes writing the same files at
    once, each paying for the whole build again. A run covering several packages asks
    once per package, and every generator runs in the first of those calls.

    :return: Whether this call built them.
    """
    if os.environ.get(PytestEnvironmentVariable.XDIST_WORKER):
        return False
    WORKSPACE_ORM_INTERFACES.regenerate()
    return True
