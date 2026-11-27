"""
The concept class of annotations that have qualities.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from krrood.patterns.classproperty import classproperty
from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.qualities.relationship_marker import (
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

    A new annotation starts with the :attr:`default_qualities` of its kind, which a
    fresher source, such as perception, replaces through :meth:`set_quality`.
    """

    def __post_init__(self) -> None:
        """
        Start with each of :attr:`default_qualities` whose type no quality given at
        construction has.
        """
        super().__post_init__()
        for quality in self.qualities_not_yet_set(among=self.default_qualities):
            self._store_in_field(
                entity=quality,
                wrapped_field=self._choose_field_for_entity(
                    entity=quality, relationship_type=IsQualityRelationship
                ),
            )

    @classproperty
    def default_qualities(cls) -> list[Quality]:
        """
        :return: The qualities an annotation of this kind has until a fresher source
            replaces them, new ones on each call so that no two annotations share one;
            none by default.
        """
        return []

    def qualities_not_yet_set(self, *, among: list[Quality]) -> list[Quality]:
        """
        :param among: Qualities, at most one of each type.
        :return: Those whose type no quality of this annotation has yet.
        """
        known_types = {type(quality) for quality in self.qualities}
        return [quality for quality in among if type(quality) not in known_types]

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
        :raises UnknownSemanticRelationshipField: If ``field_name`` names no quality
            field of this annotation.
        :raises NoFieldAcceptsEntity: If no quality field accepts ``type(quality)``.
        :raises SeveralFieldsAcceptEntity: If several quality fields accept
            ``type(quality)``.
        """
        wrapped_field = self._choose_field_for_entity(
            entity=quality,
            relationship_type=IsQualityRelationship,
            field_name=field_name,
        )
        self._store_in_field(entity=quality, wrapped_field=wrapped_field)

    @property
    def qualities(self) -> list[Quality]:
        """
        :return: The qualities of this annotation, from all of its quality fields.
        """
        return self.related_entities(relationship_type=IsQualityRelationship)
