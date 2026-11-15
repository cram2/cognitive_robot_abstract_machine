"""
Tests pinning how a string annotation of a field is resolved to a class.

A name bound in the module that declares the field resolves to that module's binding,
even when another loaded module defines a class of the same name; for an inherited field
the declaring base class's module is the one consulted. The class diagram, which knows
every class by its bare name, is only a fallback for names the declaring module does not
bind, such as ``TYPE_CHECKING``-only imports.

The test module defines its own ``Person`` on purpose, while the university dataset
module and a shadowing helper module, both imported here, each define another one.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType

import pytest

from typing_extensions import Optional

from krrood.symbol_graph.helpers import get_field_type_endpoint
from krrood.symbol_graph.symbol_graph import Symbol

from ..dataset.role_and_ontology import university_ontology_like_classes
from . import annotation_resolution_helper_module as helper_module
from .annotation_resolution_fallback_target import FallbackTarget
from .annotation_resolution_helper_module import FallbackHolder, OccupiedBase

# %% Domain


@dataclass(kw_only=True, eq=False)
class ResolutionOrganization(Symbol):
    name: str


@dataclass(kw_only=True, eq=False)
class Person(Symbol):
    """
    Deliberately named like the university dataset's ``Person``.
    """

    name: str
    is_dean_of: set[ResolutionOrganization] = field(default_factory=set)


@dataclass(kw_only=True, eq=False)
class ResolutionCourse(Symbol):
    name: str
    is_taught_by: set[Person] = field(default_factory=set)
    coordinator: Optional[Person] = None


@dataclass(kw_only=True, eq=False)
class Occupant(Symbol):
    """
    Deliberately named like the helper module's ``Occupant``.
    """

    name: str = ""


@dataclass(kw_only=True, eq=False)
class OccupiedSubclass(OccupiedBase):
    """
    Inherits a field typed by the helper module's ``Occupant`` and declares one typed by
    this module's.
    """

    visitors: set[Occupant] = field(default_factory=set)


@pytest.fixture(scope="module", autouse=True)
def shadowing_module() -> ModuleType:
    """
    :return: The module defining another ``Person``, loaded only once the classes above
        are defined, so the class diagram lists its ``Person`` after the one declared
        here.
    """
    return importlib.import_module(
        f"{__package__}.annotation_resolution_shadowing_module"
    )


# %% A same-named class in another loaded module does not shadow


def test_the_dataset_person_is_a_distinct_loaded_class():
    """
    The precondition of the shadowing tests: the dataset module defines a ``Person``
    that is not this module's.
    """
    assert university_ontology_like_classes.Person is not Person


def test_the_shadowing_person_is_a_distinct_loaded_class(shadowing_module: ModuleType):
    """
    The precondition of the shadowing tests: the shadowing module defines a ``Person``
    that is not this module's.
    """
    assert shadowing_module.Person is not Person


def test_collection_field_resolves_to_the_declaring_modules_class():
    """
    ``is_taught_by: set[Person]`` resolves to this module's ``Person`` although the
    university dataset module, loaded in the same process, defines another ``Person``.
    """
    assert get_field_type_endpoint(ResolutionCourse, "is_taught_by") is Person


def test_single_valued_field_resolves_to_the_declaring_modules_class():
    """
    ``coordinator: Optional[Person]`` resolves to this module's ``Person`` as well.
    """
    assert get_field_type_endpoint(ResolutionCourse, "coordinator") is Person


# %% Inherited fields resolve through the module that declares them


def test_inherited_field_resolves_through_the_module_of_the_base_class():
    """
    ``occupants`` is declared in the helper module, so its ``Occupant`` is meant, not
    the subclass module's ``Occupant``.
    """
    assert (
        get_field_type_endpoint(OccupiedSubclass, "occupants") is helper_module.Occupant
    )


def test_own_field_resolves_through_the_module_of_the_subclass():
    """
    ``visitors`` is declared in this module, so this module's ``Occupant`` is meant, not
    the helper module's.
    """
    assert get_field_type_endpoint(OccupiedSubclass, "visitors") is Occupant


def test_inherited_field_resolves_the_same_on_the_base_class():
    """
    The base class itself resolves ``occupants`` to the helper module's ``Occupant``.
    """
    assert get_field_type_endpoint(OccupiedBase, "occupants") is helper_module.Occupant


# %% The class diagram is a fallback for names the module does not bind


def test_type_checking_only_name_resolves_through_the_class_diagram():
    """
    ``targets: set[FallbackTarget]`` names a class the declaring module imports only
    under ``TYPE_CHECKING``, so it is found through the class diagram.
    """
    assert get_field_type_endpoint(FallbackHolder, "targets") is FallbackTarget
