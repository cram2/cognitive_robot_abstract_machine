"""
Golden-surface snapshot over every symbolic callable's verbalization.

Every concrete :class:`~krrood.entity_query_language.predicate.Predicate` /
:class:`~krrood.entity_query_language.predicate.SymbolicFunction` defined in krrood's source is
rendered with placeholder operands and compared against the committed snapshot
(``verbalization_surfaces.yaml``, one ``qualified class name: sentence`` entry per class). A class without a fragment (and without the ``NameVerbalized``
opt-in) is recorded as fragment-less rather than rendered.

The point is review visibility: adding or renaming a symbolic callable, opting into
``NameVerbalized``, or changing the shared surface builders makes the affected sentences appear as
diff lines in the snapshot file, so the author and every reviewer see — and can object to — the
exact wording a class produces, without anyone writing a per-class test.

To update the snapshot after an intentional change, run::

    UPDATE_VERBALIZATION_SURFACES=1 pytest test/krrood_test/test_eql/test_verbalization/test_verbalization_surfaces.py

and commit the regenerated YAML with the change that caused it.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import fields
from pathlib import Path

import yaml
from typing_extensions import Any, Dict, List, Type

import krrood.entity_query_language.factories
import krrood.entity_query_language.verbalization.example_domain
import krrood.inheritance_path_length
import krrood.patterns.role_predicates
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.predicate import (
    SymbolicCallable,
    Verbalizable,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.utils import recursive_subclasses

SNAPSHOT_PATH = Path(__file__).parent / "verbalization_surfaces.yaml"
"""The committed snapshot mapping each symbolic callable to its rendered surface."""

COVERED_MODULES = frozenset(
    {
        "krrood.entity_query_language.predicate",
        "krrood.entity_query_language.factories",
        "krrood.entity_query_language.verbalization.example_domain",
        "krrood.inheritance_path_length",
        "krrood.patterns.role_predicates",
    }
)
"""The source modules the snapshot covers — an explicit list (matching the imports above) so the
population is deterministic regardless of what other tests import. A new module that defines
symbolic callables is added here (and imported above) to join the snapshot."""

UPDATE_ENVIRONMENT_VARIABLE = "UPDATE_VERBALIZATION_SURFACES"
"""Set to regenerate the snapshot instead of asserting against it."""

FRAGMENT_LESS_MARKER = (
    "(no fragment -- verbalizing raises PredicateFragmentRequiredError)"
)
"""Snapshot entry for a class that made no surface decision yet."""

OPERAND_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "HasType": {"types_": int},
    "HasTypes": {"types_": (int, str)},
}
"""Concrete operands for classes whose fragments read a field's raw VALUE, keyed by class name;
every other field defaults to a fresh placeholder variable of its annotated type. ``HasType`` gets
the SINGLE type its contract declares (``types_: Type``) — the tuple form belongs to ``HasTypes``."""


def _source_symbolic_callables() -> List[Type[SymbolicCallable]]:
    """:return: the symbolic callables defined in the covered source modules, sorted by qualified
    name — the deterministic population the snapshot covers. A class whose only missing piece is
    the verbalization fragment is included (recorded as fragment-less): it is still symbolically
    constructible, so its undecided surface is exactly what the snapshot should make visible.
    """
    return sorted(
        (
            cls
            for cls in recursive_subclasses(SymbolicCallable)
            if cls.__module__ in COVERED_MODULES
            and set(getattr(cls, "__abstractmethods__", ()))
            <= {"_verbalization_fragment_"}
        ),
        key=_qualified_name,
    )


def _qualified_name(cls: Type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _has_fragment(cls: Type[SymbolicCallable]) -> bool:
    """:return: whether *cls* decided its surface (own fragment or the ``NameVerbalized`` opt-in),
    mirroring the check the grammar call-site uses."""
    return (
        cls._verbalization_fragment_.__func__
        is not Verbalizable._verbalization_fragment_.__func__
    )


def _placeholder_operands(cls: Type[SymbolicCallable]) -> Dict[str, Any]:
    """:return: one placeholder operand per init dataclass field — an override where registered,
    else a fresh variable of the field's type endpoint as the class diagram resolves it (``object``
    when the endpoint is not a plain class — ``Any``, a union, a parametrized generic), so the
    surface reads the operand as *"a <TypeName>"*."""
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
    """:return: the sentence *cls* renders with placeholder operands, or the fragment-less marker."""
    if not _has_fragment(cls):
        return FRAGMENT_LESS_MARKER
    return verbalize_expression(cls(**_placeholder_operands(cls)))


def _render_surfaces() -> Dict[str, str]:
    """:return: the snapshot mapping — ``qualified class name -> rendered surface``."""
    return {
        _qualified_name(cls): _surface_of(cls) for cls in _source_symbolic_callables()
    }


def test_every_symbolic_callable_surface_matches_the_snapshot():
    """The rendered surface of every source symbolic callable matches the committed snapshot, so
    any new class or changed wording must be re-approved by regenerating the file (see module
    docstring) and reviewing its diff."""
    actual = _render_surfaces()
    if os.environ.get(UPDATE_ENVIRONMENT_VARIABLE):
        SNAPSHOT_PATH.write_text(
            yaml.safe_dump(actual, sort_keys=True, width=1000, allow_unicode=True)
        )
    expected = yaml.safe_load(SNAPSHOT_PATH.read_text())
    assert actual == expected, (
        "Verbalization surfaces changed. Review the differences, and if the new wording is "
        f"intended, regenerate the snapshot with {UPDATE_ENVIRONMENT_VARIABLE}=1 and commit it."
    )
