"""
The ORM build and the ORM mappers every experiment test may need.

Each experiment keeps its own tests, and the fixtures they need, in a package of its own
under this one, so that tests needing ROS and tests needing nothing but the domain live
side by side.
"""

from __future__ import annotations

import pytest

# Built before anything reads a mapped datastructure: pytest imports every conftest of a
# run before calling any hook, so a hook would fire too late. The build runs once per
# process and never on an xdist worker.
from cognitive_robot_abstract_machine.orm_interfaces import OrmPackage

from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces(OrmPackage.EXPERIMENTS)


# %% the ORM mappers the tests use


@pytest.fixture(scope="session", autouse=True)
def register_orm_mappers() -> None:
    """
    Load experiments's ORM interface, whose import registers the mappers of it and the
    interfaces it builds on for every test.
    """
    import experiments.orm.ormatic_interface  # noqa: F401
