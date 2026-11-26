"""
The concept class of annotations that have qualities.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.qualities.exceptions import (
    AmbiguousQuality,
    CannotHaveQuality,
)
from semantic_digital_twin.semantic_annotations.qualities.metadata import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import Quality
from semantic_digital_twin.semantic_annotations.semantic_relationship import (
    SemanticRelationship,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

# %% Quality relationship


@dataclass(eq=False, kw_only=True)
class QualityRelationship(SemanticRelationship, Mixin):
    """
    Base for annotations that have qualities (the quality relation).

    Each quality mixin (such as
    :class:`~semantic_digital_twin.semantic_annotations.qualities.mixins.HasRipeness`)
    declares one field holding a :class:`Quality`, marked with
    :class:`IsQualityRelationship`. :meth:`set_quality` routes a quality to the field
    whose type matches it.
    """

    @synchronized_attribute_modification
    def set_quality(self, *, quality: Quality) -> None:
        """
        Set ``quality`` as this annotation's quality of its type, replacing the one it
        had, and record the change in the world.

        :param quality: The quality to set.
        :raises CannotHaveQuality: If no quality field of this annotation accepts
            ``type(quality)``.
        :raises AmbiguousQuality: If several quality fields accept ``type(quality)``.
        """
        matches = self.fields_accepting(
            entity=quality, relationship_type=IsQualityRelationship
        )
        if not matches:
            raise CannotHaveQuality(annotation=self, quality=quality)
        if len(matches) > 1:
            raise AmbiguousQuality(
                annotation=self,
                quality=quality,
                field_names=[match.field.name for match in matches],
            )
        [match] = matches
        setattr(self, match.field.name, quality)

    @property
    def qualities(self) -> list[Quality]:
        """
        :return: The qualities of this annotation, from all of its quality fields that
            are set.
        """
        return self.related_entities(relationship_type=IsQualityRelationship)
