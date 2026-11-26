"""
The semantic relationships of annotations to the entities held in their fields, such as
the parts of a whole or the qualities of a fruit.

Each kind of relationship marks its fields with its own subclass of
:class:`IsSemanticRelationship`.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Any

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.patterns.field_metadata import FieldMetadata
from krrood.patterns.mixin import Mixin
from semantic_digital_twin.exceptions import (
    NoFieldAcceptsEntity,
    SeveralFieldsAcceptEntity,
    UnknownSemanticRelationshipField,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

if TYPE_CHECKING:
    from krrood.class_diagrams.wrapped_field import WrappedField

# %% Relationship metadata


@dataclass(kw_only=True)
class IsSemanticRelationship(FieldMetadata):
    """
    Marks a field as holding entities its owner is related to by one kind of semantic
    relationship.

    Each kind of relationship is a subclass.
    """


# %% Relationships


@dataclass(eq=False, kw_only=True)
class HasSemanticRelationships(SemanticAnnotation, Mixin):
    """
    Base for annotations related to entities through fields marked with a kind of
    :class:`IsSemanticRelationship`.

    An annotation can have several kinds of relationship at once, such as the parts and
    the qualities of a fruit, so each method takes the kind it is about.
    """

    @classmethod
    def semantic_relationship_fields(
        cls, *, relationship_type: type[IsSemanticRelationship]
    ) -> list[WrappedField]:
        """
        :param relationship_type: The kind of relationship.
        :return: The fields of this class marked with ``relationship_type``.
        """
        return WrappedClass(clazz=cls).fields_with_metadata(
            metadata_type=relationship_type
        )

    def related_entities(
        self, *, relationship_type: type[IsSemanticRelationship]
    ) -> list[Any]:
        """
        :param relationship_type: The kind of relationship.
        :return: The entities held in the fields of ``relationship_type``, the elements
            of list fields one by one, leaving out fields that hold nothing.
        """
        entities = []
        for wrapped_field in self.semantic_relationship_fields(
            relationship_type=relationship_type
        ):
            value = getattr(self, wrapped_field.field.name)
            if wrapped_field.is_many_to_many_relationship:
                entities.extend(value)
            elif value is not None:
                entities.append(value)
        return entities

    def _choose_field_for_entity(
        self,
        *,
        entity: Any,
        relationship_type: type[IsSemanticRelationship],
        field_name: str = "",
    ) -> WrappedField:
        """
        :param entity: The entity to relate this annotation to.
        :param relationship_type: The kind of relationship.
        :param field_name: The name of the field to consider; when empty, every field of
            ``relationship_type`` is considered.
        :return: The one considered field whose type accepts ``entity``.
        :raises UnknownSemanticRelationshipField: If ``field_name`` names no field of
            ``relationship_type``.
        :raises NoFieldAcceptsEntity: If no considered field accepts ``entity``.
        :raises SeveralFieldsAcceptEntity: If several considered fields accept it.
        """
        candidate_fields = self.semantic_relationship_fields(
            relationship_type=relationship_type
        )
        if field_name:
            named_fields = [
                wrapped_field
                for wrapped_field in candidate_fields
                if wrapped_field.field.name == field_name
            ]
            if not named_fields:
                raise UnknownSemanticRelationshipField(
                    annotation=type(self),
                    field_name=field_name,
                    relationship_type=relationship_type,
                    available_fields=[
                        wrapped_field.field.name for wrapped_field in candidate_fields
                    ],
                )
            candidate_fields = named_fields
        matches = [
            wrapped_field
            for wrapped_field in candidate_fields
            if isinstance(entity, wrapped_field.type_endpoint)
        ]
        if not matches:
            raise NoFieldAcceptsEntity(
                annotation=self, entity=entity, relationship_type=relationship_type
            )
        if len(matches) > 1:
            raise SeveralFieldsAcceptEntity(
                annotation=self,
                entity=entity,
                relationship_type=relationship_type,
                field_names=[match.field.name for match in matches],
            )
        [match] = matches
        return match

    def _store_in_field(self, *, entity: Any, wrapped_field: WrappedField) -> None:
        """
        Put ``entity`` into ``wrapped_field``: appended when the field holds a list,
        replacing its value otherwise.
        """
        if wrapped_field.is_many_to_many_relationship:
            getattr(self, wrapped_field.field.name).append(entity)
        else:
            setattr(self, wrapped_field.field.name, entity)
