"""
Tests the constraints that the order of this repository's ORM interfaces has to satisfy.

A generator imports the generated interfaces of the packages it builds on, so those are
declared as its dependencies and built before it.
"""

from __future__ import annotations

import ast


from cognitive_robot_abstract_machine.orm_interfaces import (
    OrmInterface,
    WORKSPACE_ORM_INTERFACES,
)

ORM_FOLDER = "orm"
"""
Folder holding a package's hand written ORM model and its generated interface.
"""

# %% reading the sources


def imported_modules(node: ast.AST) -> list[str]:
    """
    The modules an import statement reads from.

    :param node: The node to inspect; anything but an import yields nothing.
    :return: The dotted names of the modules it imports.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        return [node.module]
    return []


def imported_orm_packages(interface: OrmInterface) -> set[str]:
    """
    Names of the other packages whose ORM modules a generator imports.

    :param interface: The interface whose generator is inspected.
    :return: The names of the packages it reads an ORM module of.
    """
    tree = ast.parse(interface.generator.read_text(encoding="utf-8"))
    imported = {
        module.split(".")[0]
        for node in ast.walk(tree)
        for module in imported_modules(node)
        if module.split(".")[1:2] == [ORM_FOLDER]
    }
    return imported - {interface.package_name}


# %% the order of this repository


def test_declared_dependencies_are_the_ones_the_generator_imports():
    """
    A generator reads another package's ORM model by importing it, so every such import
    is a declared dependency.
    """
    for interface in WORKSPACE_ORM_INTERFACES.interfaces:
        assert imported_orm_packages(interface) == set(interface.dependencies)


def test_every_interface_follows_the_ones_it_builds_on():
    """
    A generator reads the ORM model of the packages it depends on, so those are built
    first.
    """
    built: set[str] = set()
    for interface in WORKSPACE_ORM_INTERFACES.interfaces:
        assert set(interface.dependencies) <= built
        built.add(interface.package_name)
