"""
Tests the constraints that the order of this repository's ORM interfaces has to satisfy.

Every generator runs in the same interpreter, so what one of them imports is already
imported for the ones after it. That is what makes a build cheap, and it is also what
makes the order load bearing.
"""

from __future__ import annotations

import ast

from typing_extensions import Dict, List, Set

from cognitive_robot_abstract_machine.orm_interfaces import (
    OrmInterface,
    WORKSPACE_ORM_INTERFACES,
)

ORM_FOLDER = "orm"
"""
Folder holding a package's hand written ORM model and its generated interface.
"""

ALTERNATIVE_MAPPING = "AlternativeMapping"
"""
Base class through which a package tells ORMatic how to map a class it does not own.
"""

# %% reading the sources


def imported_modules(node: ast.AST) -> List[str]:
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


def imported_orm_packages(interface: OrmInterface) -> Set[str]:
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


def base_class_name(base: ast.expr) -> str:
    """
    The name a class states a base class under.

    :param base: The base class as it is written.
    :return: Its name, without the module it is read from or what it is subscripted
        with.
    """
    if isinstance(base, ast.Subscript):
        return base_class_name(base.value)
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Name):
        return base.id
    return ast.unparse(base)


def base_class_names_by_class_name(interface: OrmInterface) -> Dict[str, Set[str]]:
    """
    The base classes of every class a package's own sources define.

    :param interface: The interface whose package is inspected.
    :return: The names of each class's base classes, by class name.
    """
    source_root = interface.repository_root / interface.package_name / "src"
    bases_by_class_name = {}
    for source in source_root.rglob("*.py"):
        if source == interface.path:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases_by_class_name[node.name] = {
                    base_class_name(base) for base in node.bases
                }
    return bases_by_class_name


def alternative_mapping_class_names(interface: OrmInterface) -> Set[str]:
    """
    Names of the classes a package defines as ORMatic alternative mappings.

    Counts a class that reaches :data:`ALTERNATIVE_MAPPING` through another class of the
    same package, which is how a mapping that specializes another one is written.

    ..note:: Read from the source rather than from
        :func:`krrood.utils.recursive_subclasses`, which would need every workspace
        package imported here, and every ORM interface built first, since some of their
        modules import a generated interface at module level.

    :param interface: The interface whose package is inspected.
    :return: The names of its alternative mapping classes.
    """
    bases_by_class_name = base_class_names_by_class_name(interface)
    known_mapping_names = {ALTERNATIVE_MAPPING}
    newly_found_mapping_names = set(known_mapping_names)

    while newly_found_mapping_names:
        newly_found_mapping_names = set()
        for class_name, base_class_names in bases_by_class_name.items():
            is_known_mapping = class_name in known_mapping_names
            inherits_a_mapping = bool(base_class_names & known_mapping_names)
            if inherits_a_mapping and not is_known_mapping:
                newly_found_mapping_names.add(class_name)
        known_mapping_names |= newly_found_mapping_names

    return known_mapping_names - {ALTERNATIVE_MAPPING}


def dependency_closure(interface: OrmInterface) -> Set[str]:
    """
    Names of the packages an interface builds on, directly or through another one.

    :param interface: The interface whose dependencies are collected.
    :return: The names of the packages it builds on.
    """
    interfaces_by_name = {
        member.package_name: member for member in WORKSPACE_ORM_INTERFACES.interfaces
    }
    closure = set()
    pending = list(interface.dependencies)
    while pending:
        package_name = pending.pop()
        if package_name in closure:
            continue
        closure.add(package_name)
        pending.extend(interfaces_by_name[package_name].dependencies)
    return closure


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
    built: Set[str] = set()
    for interface in WORKSPACE_ORM_INTERFACES.interfaces:
        assert set(interface.dependencies) <= built
        built.add(interface.package_name)


def test_alternative_mappings_only_reach_the_interfaces_that_build_on_them():
    """
    ORMatic collects alternative mappings from every imported subclass, so a package
    defining them is built after every package that does not build on it.
    """
    interfaces = WORKSPACE_ORM_INTERFACES.interfaces
    for position, interface in enumerate(interfaces):
        allowed = dependency_closure(interface)
        leaking = {
            earlier.package_name
            for earlier in interfaces[:position]
            if earlier.package_name not in allowed
            and alternative_mapping_class_names(earlier)
        }
        assert not leaking
