"""
Golden-surface snapshot over every symbolic callable's verbalization.

Every concrete :class:`~krrood.entity_query_language.predicate.Predicate` /
:class:`~krrood.entity_query_language.predicate.SymbolicFunction` that krrood's source package
defines is discovered automatically — by walking the package, not from a hardcoded module list — and
its rendered surface is checked against the committed snapshot in :mod:`verbalization_surfaces`, one
:class:`~verbalization_surfaces.VerbalizationSurface` per class. Every covered class must implement a
fragment; a fragment-less class fails.

The point is review visibility: adding or renaming a symbolic callable, or changing the shared
surface builders, shows up as a diff in ``verbalization_surfaces.py`` (a new entry, a broken import,
or a changed sentence), so the author and every reviewer see — and can object to — the exact wording
a class produces, without anyone writing a per-class test.

When an intentional change alters a surface, the match test prints each class and its new sentence;
update that class's :class:`~verbalization_surfaces.VerbalizationSurface` in
``verbalization_surfaces.py`` and commit it (the diff is the review record). A newly added symbolic
callable fails the coverage test until it gets an entry.
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

import krrood
from typing_extensions import Any, Dict, Tuple, Type

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.predicate import SymbolicCallable, Verbalizable
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.ormatic.utils import classes_of_package

from .verbalization_surfaces import SURFACES

OPERAND_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "HasType": {"types_": int},
    "HasTypes": {"types_": (int, str)},
}
"""
Concrete operands for classes whose fragments read a field's raw VALUE, keyed by class
name; every other field defaults to a fresh placeholder variable of its annotated type.

``HasType`` gets the SINGLE type its contract declares (``types_: Type``) — the tuple
form belongs to ``HasTypes``. The annotation alone cannot drive this: a ``Type`` field
is a value here but a symbolic operand in ``InheritancePathLength``, so the choice is
made per class.
"""


def _qualified_name(cls: Type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@lru_cache(maxsize=None)
def _source_symbolic_callables() -> Tuple[Type[SymbolicCallable], ...]:
    """:return: every concrete-enough symbolic callable defined anywhere in krrood's source package,
    discovered by walking the package so no module list is hardcoded, sorted by qualified name. A
    class whose only missing piece is the verbalization fragment is still included, so the fragment
    guard can flag it instead of it silently escaping coverage.
    """
    discovered = {
        cls
        for cls in classes_of_package(krrood, recursive=True)
        if isinstance(cls, type)
        and issubclass(cls, SymbolicCallable)
        and set(cls.__abstractmethods__) <= {"_verbalization_fragment_"}
    }
    return tuple(sorted(discovered, key=_qualified_name))


def _has_fragment(cls: Type[SymbolicCallable]) -> bool:
    """:return: whether *cls* decided its surface by implementing its own fragment, mirroring the
    check the grammar call-site uses."""
    return (
        cls._verbalization_fragment_.__func__
        is not Verbalizable._verbalization_fragment_.__func__
    )


def _placeholder_operands(cls: Type[SymbolicCallable]) -> Dict[str, Any]:
    """:return: one placeholder operand per init dataclass field — an override where registered, else
    a fresh variable of the field's type endpoint as the class diagram resolves it (``object`` when
    the endpoint is not a plain class — ``Any``, a union, a parametrized generic), so the surface
    reads the operand as *"a <TypeName>"*."""
    overrides = OPERAND_OVERRIDES.get(cls.__name__, {})
    wrapped_class = WrappedClass(clazz=cls)
    operands: Dict[str, Any] = {}
    for field in fields(cls):
        if not field.init:
            continue
        if field.name in overrides:
            operands[field.name] = overrides[field.name]
            continue
        endpoint = WrappedField(wrapped_class, field).type_endpoint
        placeholder_type = (
            endpoint if isinstance(endpoint, type) and endpoint is not Any else object
        )
        operands[field.name] = variable(placeholder_type, [])
    return operands


def _surface_of(cls: Type[SymbolicCallable]) -> str:
    """:return: the sentence *cls* renders with placeholder operands."""
    return verbalize_expression(cls(**_placeholder_operands(cls)))


def test_every_covered_symbolic_callable_has_a_verbalization_fragment():
    """
    Every discovered symbolic callable must implement its own verbalization fragment —
    there is no undecided surface.

    A new predicate or function that ships without a fragment is a red test until it
    gets one.
    """
    fragment_less = sorted(
        _qualified_name(cls)
        for cls in _source_symbolic_callables()
        if not _has_fragment(cls)
    )
    assert not fragment_less, (
        "These symbolic callables have no verbalization fragment and must implement "
        f"_verbalization_fragment_: {fragment_less}."
    )


def test_covered_symbolic_callables_match_the_declared_surfaces():
    """
    The discovered symbolic callables are exactly the classes declared in
    :data:`verbalization_surfaces.SURFACES`.

    A newly added class with no entry, or an entry for a class that no longer exists, is a
    red test — so the snapshot cannot silently drift out of coverage.
    """
    discovered = {
        _qualified_name(cls)
        for cls in _source_symbolic_callables()
        if _has_fragment(cls)
    }
    declared = {_qualified_name(surface.callable_class) for surface in SURFACES}
    missing = sorted(discovered - declared)
    stale = sorted(declared - discovered)
    assert discovered == declared, (
        f"verbalization_surfaces.SURFACES is out of sync. Discovered classes with no entry "
        f"(add one): {missing}. Entries whose class is no longer discovered (remove them): {stale}."
    )


def test_every_declared_surface_matches_what_its_class_renders():
    """
    Each declared sentence matches what its class renders with placeholder operands, so
    any wording change is re-approved by updating the entry and reviewing the diff.
    """
    mismatches = {
        _qualified_name(surface.callable_class): _surface_of(surface.callable_class)
        for surface in SURFACES
        if _has_fragment(surface.callable_class)
        and _surface_of(surface.callable_class) != surface.sentence
    }
    assert not mismatches, (
        "Verbalization surfaces changed. Update the sentence for each of these in "
        f"verbalization_surfaces.py: {mismatches}."
    )
