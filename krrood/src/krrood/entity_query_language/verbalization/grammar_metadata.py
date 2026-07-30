"""
Per-field grammar / verbalization hints.

:class:`GrammarMetadata` is the verbalization layer's field metadata: a domain class attaches it to a
field to steer how that field is verbalized — its surface word, whether it identifies its instance,
how a boolean field reads as a predicate. It subclasses
:class:`~krrood.patterns.field_metadata.FieldMetadata` but lives here in ``verbalization``, not in
``patterns``, because every hint it carries is a verbalization concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import Optional

from krrood.patterns.field_metadata import FieldMetadata

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.boolean_predicate import (
        BooleanPredicate,
    )


@dataclass
class GrammarMetadata(FieldMetadata):
    """
    Grammar / verbalization hints for a field.
    """

    is_identifying_field: bool = False
    """
    ``True`` when this field identifies its instance for verbalization (*"a specific <Type> with
    <field> '<value>'"*).
    """

    display_name: Optional[str] = None
    """
    Surface word to use for this field when verbalized, in place of its attribute name
    (*"beginning"* for a field named ``begin``); ``None`` keeps the attribute name.
    """

    boolean_predicate: Optional[BooleanPredicate] = None
    """
    How a boolean field reads as a predicate — *"has milk"* / *"produces milk"* / *"is operational"*
    — overriding the default heuristic.

    ``None`` lets the verbalizer infer the form from the attribute name's shape.
    """
