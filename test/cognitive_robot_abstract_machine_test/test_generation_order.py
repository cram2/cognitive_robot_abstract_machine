"""
Tests the constraints that the order of this repository's ORM interfaces has to satisfy.

A generator imports the generated interfaces of the packages it builds on, so those are
built before it.
"""

from __future__ import annotations

from cognitive_robot_abstract_machine.orm_interfaces import WORKSPACE_ORM_INTERFACES

# %% the order of this repository


def test_every_interface_follows_the_ones_it_builds_on():
    """
    A generator reads the ORM model of the packages it depends on, so those are built
    first.
    """
    built: set[str] = set()
    for interface in WORKSPACE_ORM_INTERFACES.interfaces:
        assert set(interface.dependencies) <= built
        built.add(interface.package_name)
