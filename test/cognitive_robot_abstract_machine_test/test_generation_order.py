"""
Tests the order this repository's ORM interfaces are built in.

A generator imports the generated interfaces of the packages it builds on, so those are
built before it.
"""

from __future__ import annotations

from cognitive_robot_abstract_machine.orm_interfaces import WORKSPACE_ORM_INTERFACES

# %% the order of this repository


def test_every_interface_follows_the_ones_it_builds_on():
    """
    Every interface of this repository has a place in the build order after the
    interfaces it builds on.
    """
    built: set[str] = set()
    for interface in WORKSPACE_ORM_INTERFACES.interfaces_in_build_order:
        assert set(interface.dependencies) <= built
        built.add(interface.package_name)
    assert built == {
        interface.package_name for interface in WORKSPACE_ORM_INTERFACES.interfaces
    }
