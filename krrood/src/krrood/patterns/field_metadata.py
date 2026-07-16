from __future__ import annotations

from dataclasses import field, fields, dataclass, is_dataclass

from typing_extensions import ClassVar, Dict, List, Optional, Type, TypeVar, Self


@dataclass
class FieldMetadata:
    """
    Krrood-specific metadata carried inside a dataclass field's ``metadata`` mapping.

    A field carries a single :class:`FieldMetadata` under :attr:`METADATA_KEY` (attach
    it with :meth:`as_dict`). Specific metadata classes are held as sub-metadatas in
    :attr:`other_metadata` and read back by type via :meth:`get_metadata_by_type`.
    """

    METADATA_KEY: ClassVar[str] = "krrood_field_metadata"
    """
    The key this metadata is stored under inside a field's ``metadata`` mapping.
    """

    other_metadata: List[FieldMetadata] = field(default_factory=list)
    """
    The typed sub-metadatas this field carries (e.g. :class:`GrammarMetadata`),
    retrieved by type via :meth:`get_metadata_by_type`.
    """

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
    def of_field(cls, clazz: Type, field_name: str) -> Optional[Self]:
        """:return: The :class:`FieldMetadata` attached to *field_name* of *clazz*, or ``None`` when
        *clazz* is not a dataclass, has no such field, or the field carries no metadata.
        """
        if not is_dataclass(clazz):
            return None
        field_ = next((f for f in fields(clazz) if f.name == field_name), None)
        if field_ is None:
            return None
        field_metadata: Optional[FieldMetadata] = field_.metadata.get(cls.METADATA_KEY)
        if field_metadata is None:
            return None
        if cls is FieldMetadata:
            return field_metadata
        return field_metadata.get_metadata_by_type(cls)


MetadataType = TypeVar("MetadataType", bound=FieldMetadata)
"""
A type that is a subclass of :class:`FieldMetadata`.
"""


@dataclass
class GrammarMetadata(FieldMetadata):
    """
    Grammar / verbalization hints for a field.
    """

    is_identifying_field: bool = False
    """
    ``True`` when this field identifies its instance for verbalization (*"a specific
    <Type> with <field> '<value>'"*).
    """

    display_name: Optional[str] = None
    """
    Surface word to use for this field when verbalized, in place of its attribute name
    (*"beginning"* for a field named ``begin``); ``None`` keeps the attribute name.
    """
