from __future__ import annotations

from dataclasses import dataclass, field, fields

from typing_extensions import List, Optional

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.field_metadata import FieldMetadata

# %% mimics


@dataclass
class MarksAStructuralPart(FieldMetadata):
    """
    Metadata mimicking a relation marker that carries a payload describing the relation.
    """

    consumes_the_part: bool = False
    """
    Payload distinguishing two instances of this metadata from each other.
    """


@dataclass
class MarksAnIdentifyingField(FieldMetadata):
    """
    A second metadata type, used to check that discovery discriminates by type.
    """


@dataclass
class OwnerWithMarkedAndUnmarkedFields:
    """
    Mimic declaring both marked and unmarked fields, the marked ones interleaved with
    the unmarked ones so declaration order is observable.
    """

    unmarked_before: int = 0
    consuming_part: Optional[str] = field(
        default=None, metadata=MarksAStructuralPart(consumes_the_part=True).as_dict()
    )
    unmarked_between: int = 0
    plain_part: Optional[str] = field(
        default=None, metadata=MarksAStructuralPart().as_dict()
    )
    identifier: str = field(default="", metadata=MarksAnIdentifyingField().as_dict())


@dataclass
class HeirDeclaringAnotherMarkedField(OwnerWithMarkedAndUnmarkedFields):
    """
    Mimic adding a marked field to an owner that already has some, to exercise
    discovery across the inheritance chain.
    """

    inherited_owner_parts: List[str] = field(
        default_factory=list, metadata=MarksAStructuralPart().as_dict()
    )


@dataclass
class OwnerWithoutAnyMarkedField:
    """
    Mimic marking no field at all.
    """

    unmarked: int = 0


class NotADataclass:
    """
    Mimic that is not a dataclass, so it has no fields to discover.
    """


# %% discovery


def test_only_marked_fields_are_discovered_in_declaration_order():
    wrapped_fields = WrappedClass(
        OwnerWithMarkedAndUnmarkedFields
    ).fields_with_metadata(MarksAStructuralPart)

    assert [wrapped_field.field.name for wrapped_field in wrapped_fields] == [
        "consuming_part",
        "plain_part",
    ]


def test_inherited_marked_fields_are_discovered():
    wrapped_fields = WrappedClass(HeirDeclaringAnotherMarkedField).fields_with_metadata(
        MarksAStructuralPart
    )

    assert [wrapped_field.field.name for wrapped_field in wrapped_fields] == [
        "consuming_part",
        "plain_part",
        "inherited_owner_parts",
    ]


def test_discovery_discriminates_between_metadata_types():
    wrapped_fields = WrappedClass(
        OwnerWithMarkedAndUnmarkedFields
    ).fields_with_metadata(MarksAnIdentifyingField)

    assert [wrapped_field.field.name for wrapped_field in wrapped_fields] == [
        "identifier"
    ]


def test_class_without_marked_fields_discovers_nothing():
    assert (
        WrappedClass(OwnerWithoutAnyMarkedField).fields_with_metadata(
            MarksAStructuralPart
        )
        == []
    )


def test_non_dataclass_discovers_nothing():
    assert WrappedClass(NotADataclass).fields_with_metadata(MarksAStructuralPart) == []


# %% reading the metadata back


def _dataclass_field(clazz: type, name: str):
    return next(
        dataclass_field
        for dataclass_field in fields(clazz)
        if dataclass_field.name == name
    )


def test_metadata_of_wrapped_field_carries_the_payload():
    [consuming_part, plain_part] = WrappedClass(
        OwnerWithMarkedAndUnmarkedFields
    ).fields_with_metadata(MarksAStructuralPart)

    assert MarksAStructuralPart.of_wrapped_field(consuming_part).consumes_the_part
    assert not MarksAStructuralPart.of_wrapped_field(plain_part).consumes_the_part


def test_metadata_of_unmarked_wrapped_field_is_none():
    wrapped_field = WrappedField(
        WrappedClass(OwnerWithMarkedAndUnmarkedFields),
        _dataclass_field(OwnerWithMarkedAndUnmarkedFields, "unmarked_before"),
    )

    assert MarksAStructuralPart.of_wrapped_field(wrapped_field) is None


def test_metadata_is_found_when_the_public_name_differs_from_the_field_name():
    # A field managed by a property descriptor is addressed under a public name that is
    # not its dataclass field name; the metadata still lives on the dataclass field.
    wrapped_field = WrappedField(
        WrappedClass(OwnerWithMarkedAndUnmarkedFields),
        _dataclass_field(OwnerWithMarkedAndUnmarkedFields, "consuming_part"),
        public_name="parts",
    )

    assert MarksAStructuralPart.of_wrapped_field(wrapped_field).consumes_the_part
