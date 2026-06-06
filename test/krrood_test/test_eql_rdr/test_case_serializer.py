"""Unit tests for ``CaseSerializer`` ABC and ``AsdictCaseSerializer``."""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import asdict, dataclass, field
from typing_extensions import Any, Set, Tuple, Type

import pytest

from krrood.entity_query_language.rdr.corner_case import (
    AsdictCaseSerializer,
    CaseNotSerializableError,
    CaseSerializer,
    CornerCaseStore,
)
from krrood.entity_query_language.factories import variable

from .animal import Animal, Species

# ---------------------------------------------------------------------------
# Inline dataclass fixtures (pattern-named, never collected by pytest)
# ---------------------------------------------------------------------------


@dataclass
class FlatAnimal:
    """Pattern: FlatAnimal — flat dataclass with only scalar fields.

    Used to verify that ``AsdictCaseSerializer.to_source`` emits valid eval-able
    Python constructor source for the simplest possible case.
    """

    hair: bool
    legs: int
    name: str


@dataclass
class AnimalWithSpecies:
    """Pattern: AnimalWithSpecies — flat dataclass with one enum field.

    Used to verify that enum fields are serialized as ``EnumType.member`` notation
    and that the enum type appears in ``referenced_types``.
    """

    name: str
    species: Species


@dataclass
class Limb:
    """Pattern: Limb — inner dataclass for nesting tests.

    Used as the nested field type in ``LimbedAnimal``.
    """

    count: int
    has_claws: bool


@dataclass
class LimbedAnimal:
    """Pattern: LimbedAnimal — outer dataclass whose ``limb`` field is a nested dataclass.

    Used to verify recursive serialization and that both the outer and inner types
    appear in ``referenced_types``.
    """

    name: str
    limb: Limb


@dataclass
class BadFieldCase:
    """Pattern: BadFieldCase — dataclass with an unserializable field.

    The ``payload`` field holds an arbitrary Python object (a ``set`` of strings)
    that ``value_to_source`` cannot handle without falling back to ``repr()``.
    ``AsdictCaseSerializer`` must raise ``CaseNotSerializableError`` for this case.
    """

    name: str
    payload: set  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Shared serializer instance (stateless — safe to reuse across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ser() -> AsdictCaseSerializer:
    """Module-scoped ``AsdictCaseSerializer`` instance.

    ``AsdictCaseSerializer`` is stateless, so module scope avoids repeated
    construction without sacrificing test isolation.
    """
    return AsdictCaseSerializer()


# ---------------------------------------------------------------------------
# AsdictCaseSerializer.to_source — happy paths
# ---------------------------------------------------------------------------


def test_flat_dataclass_to_source_is_eval_able(ser):
    """``to_source`` for a flat dataclass emits source that ``eval`` reconstructs to an equal instance.

    Guarantees: the emitted string is valid Python that, when evaluated with the
    case type in scope, produces an object equal to the original.
    """
    case = FlatAnimal(hair=True, legs=4, name="cat")

    source, _ = ser.to_source(case)

    reconstructed = eval(source, {"FlatAnimal": FlatAnimal})  # noqa: S307
    assert reconstructed == case


def test_flat_dataclass_to_source_source_is_nonempty_string(ser):
    """``to_source`` returns a non-empty string as its first element."""
    case = FlatAnimal(hair=False, legs=6, name="ant")

    source, _ = ser.to_source(case)

    assert isinstance(source, str)
    assert source.strip() != ""


def test_flat_dataclass_referenced_types_includes_case_type(ser):
    """``to_source`` includes the case class itself in the returned ``referenced_types`` set."""
    case = FlatAnimal(hair=True, legs=2, name="bird")

    _, ref_types = ser.to_source(case)

    assert FlatAnimal in ref_types


def test_enum_field_to_source_is_eval_able(ser):
    """``to_source`` for a dataclass with an enum field emits eval-able source.

    Guarantees: ``eval(source)`` with both the case type and the enum type in
    scope reconstructs an instance equal to the original.
    """
    case = AnimalWithSpecies(name="wolf", species=Species.mammal)

    source, _ = ser.to_source(case)

    ns = {"AnimalWithSpecies": AnimalWithSpecies, "Species": Species}
    reconstructed = eval(source, ns)  # noqa: S307
    assert reconstructed == case


def test_enum_field_referenced_types_includes_enum_type(ser):
    """``referenced_types`` for a dataclass with an enum field includes the enum class."""
    case = AnimalWithSpecies(name="eagle", species=Species.bird)

    _, ref_types = ser.to_source(case)

    assert Species in ref_types


def test_enum_field_referenced_types_includes_case_type(ser):
    """``referenced_types`` for a dataclass with an enum field also includes the case type."""
    case = AnimalWithSpecies(name="frog", species=Species.amphibian)

    _, ref_types = ser.to_source(case)

    assert AnimalWithSpecies in ref_types


def test_nested_dataclass_to_source_is_eval_able(ser):
    """``to_source`` for a dataclass with a nested dataclass field emits eval-able source.

    Guarantees: ``eval(source)`` with both outer and inner types in scope
    reconstructs an instance equal to the original.
    """
    case = LimbedAnimal(name="cat", limb=Limb(count=4, has_claws=True))

    source, _ = ser.to_source(case)

    ns = {"LimbedAnimal": LimbedAnimal, "Limb": Limb}
    reconstructed = eval(source, ns)  # noqa: S307
    assert reconstructed == case


def test_nested_dataclass_referenced_types_includes_outer_type(ser):
    """``referenced_types`` for a case with a nested field includes the outer case type."""
    case = LimbedAnimal(name="dog", limb=Limb(count=4, has_claws=False))

    _, ref_types = ser.to_source(case)

    assert LimbedAnimal in ref_types


def test_nested_dataclass_referenced_types_includes_inner_type(ser):
    """``referenced_types`` for a case with a nested field includes the inner dataclass type."""
    case = LimbedAnimal(name="dog", limb=Limb(count=4, has_claws=False))

    _, ref_types = ser.to_source(case)

    assert Limb in ref_types


# ---------------------------------------------------------------------------
# AsdictCaseSerializer.to_source — error path
# ---------------------------------------------------------------------------


def test_non_serializable_field_raises_case_not_serializable_error(ser):
    """``to_source`` raises ``CaseNotSerializableError`` for a field holding an arbitrary object.

    A ``set`` of strings is not a scalar, not an enum, and not a nested dataclass.
    ``AsdictCaseSerializer`` must not silently fall through to ``repr()``; it must
    raise ``CaseNotSerializableError`` instead.
    """
    case = BadFieldCase(name="problem", payload={"a", "b", "c"})

    with pytest.raises(CaseNotSerializableError):
        ser.to_source(case)


# ---------------------------------------------------------------------------
# AsdictCaseSerializer.from_data — reconstruction
# ---------------------------------------------------------------------------


def test_from_data_reconstructs_flat_dataclass(ser):
    """``from_data(asdict(case), CaseType)`` reconstructs a flat dataclass equal to the original.

    Guarantees: round-trip ``asdict`` → ``from_data`` is lossless for flat dataclasses.
    """
    case = FlatAnimal(hair=True, legs=4, name="lynx")

    reconstructed = ser.from_data(asdict(case), FlatAnimal)

    assert reconstructed == case
    assert isinstance(reconstructed, FlatAnimal)


def test_from_data_reconstructs_nested_dataclass(ser):
    """``from_data(asdict(case), CaseType)`` reconstructs a case with a nested dataclass field.

    Guarantees: ``from_data`` recursively converts inner dicts back to their
    declared dataclass types using ``get_type_hints`` on the outer type.
    """
    case = LimbedAnimal(name="panther", limb=Limb(count=4, has_claws=True))

    reconstructed = ser.from_data(asdict(case), LimbedAnimal)

    assert reconstructed == case
    assert isinstance(reconstructed, LimbedAnimal)
    assert isinstance(reconstructed.limb, Limb)


# ---------------------------------------------------------------------------
# Pluggable serializer on CornerCaseStore
# ---------------------------------------------------------------------------


def _make_condition_node():
    """Return a single EQL comparator node with a stable ``_id_`` UUID."""
    av = variable(Animal, domain=[])
    return av.milk == True  # noqa: E712


@dataclass
class SentinelSerializer(CaseSerializer):
    """Pattern: SentinelSerializer — records calls and returns a sentinel source string.

    Used to verify that ``CornerCaseStore.to_ordered_sources`` delegates to
    ``self.serializer`` rather than any hardcoded serialization logic.
    """

    SENTINEL: str = "SENTINEL_SOURCE"
    called_with: list = field(default_factory=list)

    def to_source(self, case: Any) -> Tuple[str, Set[Type]]:
        """Record the case and return a sentinel string so callers can detect this was used."""
        self.called_with.append(case)
        return (self.SENTINEL, set())

    def from_data(self, data: Any, case_type: Type) -> Any:
        """Stub: not exercised in CornerCaseStore integration tests."""
        return data


def test_custom_serializer_is_used_by_to_ordered_sources():
    """``CornerCaseStore(serializer=custom)`` causes ``to_ordered_sources`` to use ``custom.to_source``.

    Guarantees: the store delegates serialization to ``self.serializer``; the default
    ``AsdictCaseSerializer`` is NOT used when a custom serializer is provided.
    """
    sentinel_ser = SentinelSerializer()
    store = CornerCaseStore(serializer=sentinel_ser)
    node = _make_condition_node()
    case = FlatAnimal(hair=True, legs=4, name="fox")
    store.record(node, case)

    result = store.to_ordered_sources([node])

    assert len(sentinel_ser.called_with) == 1
    assert sentinel_ser.called_with[0] is case
    assert result[0][0] == SentinelSerializer.SENTINEL


def test_default_serializer_is_asdict_serializer():
    """``CornerCaseStore()`` uses an ``AsdictCaseSerializer`` by default.

    Guarantees: the zero-argument constructor produces a store whose ``serializer``
    attribute is an instance of ``AsdictCaseSerializer``.
    """
    store = CornerCaseStore()

    assert isinstance(store.serializer, AsdictCaseSerializer)
