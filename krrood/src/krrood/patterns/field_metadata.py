from __future__ import annotations

from dataclasses import Field, dataclass

from typing_extensions import ClassVar, Dict, Optional


@dataclass(frozen=True)
class FieldMetadata:
    """Krrood-specific metadata carried inside a dataclass field's ``metadata`` mapping.

    Attach it to a field with :meth:`as_dict` so krrood can read per-field hints (such as
    which field identifies an instance) without the owning class implementing any protocol.
    """

    METADATA_KEY: ClassVar[str] = "krrood_field_metadata"
    """The key this metadata is stored under inside a field's ``metadata`` mapping."""

    is_identifying_attribute: bool = False
    """``True`` when this field identifies its instance for verbalization
    (*"a specific <Type> with <field> '<value>'"*)."""

    is_part_whole_relationship: bool = False
    """``True`` when this field holds a structural *part* of its owner (the part-whole relation)."""

    @classmethod
    def as_dict(
        cls,
        *,
        is_identifying_attribute: bool = False,
        is_part_whole_relationship: bool = False,
    ) -> Dict[str, FieldMetadata]:
        """:return: A dataclass-field ``metadata`` mapping carrying a :class:`FieldMetadata`
        with the given hints under :attr:`METADATA_KEY`, ready to pass to ``field(metadata=...)``.
        """
        return {
            cls.METADATA_KEY: cls(
                is_identifying_attribute=is_identifying_attribute,
                is_part_whole_relationship=is_part_whole_relationship,
            )
        }

    @classmethod
    def of_field(cls, field: Field) -> Optional[FieldMetadata]:
        """:return: The :class:`FieldMetadata` attached to *field*, or ``None`` when it carries
        none."""
        return field.metadata.get(cls.METADATA_KEY)
