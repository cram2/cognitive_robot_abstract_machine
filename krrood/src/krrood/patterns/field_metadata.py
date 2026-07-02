from __future__ import annotations

from dataclasses import Field, dataclass, field

from typing_extensions import ClassVar, Dict, List, Optional, Type, TypeVar

MetadataType = TypeVar("MetadataType", bound="FieldMetadata")


@dataclass(frozen=True)
class FieldMetadata:
    """Krrood-specific metadata carried inside a dataclass field's ``metadata`` mapping.

    A field carries a single :class:`FieldMetadata` under :attr:`METADATA_KEY` (attach it with
    :meth:`as_dict`). Specific, typed hints are held as sub-metadatas in :attr:`other_metadata` and
    read back by type via :meth:`get_metadata_by_type`, so krrood reads per-field hints without the
    owning class implementing any protocol.
    """

    METADATA_KEY: ClassVar[str] = "krrood_field_metadata"
    """The key this metadata is stored under inside a field's ``metadata`` mapping."""

    other_metadata: List[FieldMetadata] = field(default_factory=list)
    """The typed sub-metadatas this field carries (e.g. :class:`GrammarMetadata`), retrieved by type
    via :meth:`get_metadata_by_type`."""

    def as_dict(self) -> Dict[str, FieldMetadata]:
        """:return: a dataclass-field ``metadata`` mapping carrying this metadata under
        :attr:`METADATA_KEY`, ready to pass to ``field(metadata=...)``.
        """
        return {self.METADATA_KEY: self}

    def get_metadata_by_type(
        self, metadata_type: Type[MetadataType]
    ) -> Optional[MetadataType]:
        """:return: the first sub-metadata in :attr:`other_metadata` that is an instance of
        *metadata_type*, or ``None`` when none is present.
        """
        for metadata in self.other_metadata:
            if isinstance(metadata, metadata_type):
                return metadata
        return None

    @classmethod
    def of_field(cls, dataclass_field: Field) -> Optional[FieldMetadata]:
        """:return: The :class:`FieldMetadata` attached to *dataclass_field*, or ``None`` when it
        carries none."""
        return dataclass_field.metadata.get(cls.METADATA_KEY)


@dataclass(frozen=True)
class GrammarMetadata(FieldMetadata):
    """Grammar / verbalization hints for a field."""

    is_identifying_field: bool = False
    """``True`` when this field identifies its instance for verbalization
    (*"a specific <Type> with <field> '<value>'"*)."""


@dataclass(frozen=True)
class IsPartWholeRelationship(FieldMetadata):
    """Marks a field as holding a structural *part* of its owner (the part-whole relation).

    The relation is signalled by the mere presence of an instance of this class in a field's
    :attr:`~FieldMetadata.other_metadata`; it carries no further data.
    """
