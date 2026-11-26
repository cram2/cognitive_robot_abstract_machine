"""
The concept class of annotations that have qualities.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.qualities.metadata import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import Quality
from semantic_digital_twin.semantic_annotations.semantic_relationship import (
    HasSemanticRelationships,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

# %% Quality relationship


@dataclass(eq=False, kw_only=True)
class HasQualities(HasSemanticRelationships, Mixin):
    """
    Base for annotations that have qualities (the quality relation).

    Each quality mixin (such as
    :class:`~semantic_digital_twin.semantic_annotations.qualities.mixins.HasRipeness`)
    declares a field holding a :class:`Quality`, marked with
    :class:`IsQualityRelationship`. :meth:`set_quality` routes a quality to the field
    whose type matches it.
    """

    @synchronized_attribute_modification
    def set_quality(self, *, quality: Quality, field_name: str = "") -> None:
        """
        Set ``quality`` as a quality of this annotation, routing it to the matching
        quality field by type, and record the change in the world. A single-valued field
        has its quality replaced; a list field gets it appended.

        :param quality: The quality to set.
        :param field_name: The name of the quality field to set the quality in, which
            resolves the ambiguity when several fields accept ``type(quality)``. When
            empty, the field is chosen by type alone.
        :raises UnknownSemanticRelationshipField: If ``field_name`` names no quality field
            of this annotation.
        :raises NoFieldAcceptsEntity: If no quality field accepts ``type(quality)``.
        :raises SeveralFieldsAcceptEntity: If several quality fields accept
            ``type(quality)``.
        """
        wrapped_field = self._field_accepting(
            entity=quality, relationship_type=IsQualityRelationship, field_name=field_name
        )
        self._store_in_field(entity=quality, wrapped_field=wrapped_field)

    @property
    def qualities(self) -> list[Quality]:
        """
        :return: The qualities of this annotation, from all of its quality fields.
        """
        return self.related_entities(relationship_type=IsQualityRelationship)
