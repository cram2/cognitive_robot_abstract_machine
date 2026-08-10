"""
Metadata for the part-whole relation between semantic annotations.

This module holds only the vocabulary of the relation, so both the annotation mixins
that declare part-whole fields and the specification API that fills them can depend on
it without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.field_metadata import FieldMetadata

# %% relation metadata


@dataclass
class IsPartWholeRelationship(FieldMetadata):
    """
    Marks a field as holding a structural *part* of its owner (the part-whole relation).

    The relation is signalled by the presence of an instance of this class in the
    field's ``metadata`` mapping (attach it with :meth:`~FieldMetadata.as_dict`), and
    the instance describes how mounting a part into that field affects the whole.
    """

    removes_part_geometry_from_whole: bool = False
    """
    Whether mounting a part into this field removes the part's volume from the whole's
    collision and visual geometry.

    This is a property of the relation rather than of the part: the same
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.EntryWay`
    cuts the wall it is an aperture of, but not the door whose passage it marks.
    """
