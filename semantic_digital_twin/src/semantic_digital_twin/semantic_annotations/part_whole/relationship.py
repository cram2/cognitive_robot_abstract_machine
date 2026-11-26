"""
The concept class of wholes: annotations whose fields hold their structural parts.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.part_whole.relationship_marker import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.semantic_relationship import (
    HasSemanticRelationships,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

# %% Part-whole relationship


@dataclass(eq=False, kw_only=True)
class HasParts(HasRootKinematicStructureEntity, HasSemanticRelationships, Mixin):
    """
    Base for annotations that have structural *parts* (the part-whole relation).

    Each part mixin (such as
    :class:`~semantic_digital_twin.semantic_annotations.part_whole.furniture_parts.HasDoors`)
    declares a part-whole relationship field marked with :class:`IsPartWholeRelationship`.
    :meth:`add` routes a part to the field whose element type matches it and lets the
    part mount itself (:meth:`HasRootKinematicStructureEntity._mount_strategy`).
    """

    @synchronized_attribute_modification
    def add(
        self, *, part: HasRootKinematicStructureEntity, field_name: str = ""
    ) -> None:
        """
        Add ``part`` as a structural part, routing it to the matching part-whole
        relationship field by type, and record the change in the world.

        :param part: The part to add.
        :param field_name: The name of the part-whole relationship field to add the part
            to, which resolves the ambiguity when several fields accept ``type(part)``.
            When empty, the field is chosen by type alone.
        :raises UnknownSemanticRelationshipField: If ``field_name`` names no part-whole
            relationship field of this annotation.
        :raises NoFieldAcceptsEntity: If no part-whole relationship field accepts
            ``type(part)``.
        :raises SeveralFieldsAcceptEntity: If several part-whole relationship fields
            accept ``type(part)``.
        """
        wrapped_field = self._choose_field_for_entity(
            entity=part,
            relationship_type=IsPartWholeRelationship,
            field_name=field_name,
        )
        part._mount_strategy(
            self, IsPartWholeRelationship.of_wrapped_field(wrapped_field=wrapped_field)
        )
        self._store_in_field(entity=part, wrapped_field=wrapped_field)

    @property
    def parts(self) -> list[HasRootKinematicStructureEntity]:
        """
        :return: The structural parts of this annotation, from all of its part-whole
            relationship fields.
        """
        return self.related_entities(relationship_type=IsPartWholeRelationship)
