"""
The part-whole relationships of annotation types whose parts can be split off the shape
of the annotation's root body.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.utils import camel_case_to_words

from semantic_digital_twin.exceptions import (
    NoSinglePartWholeRelationshipField,
    UnknownPartWholeRelationshipField,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasHandle,
    HasRootBody,
    PartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)


@dataclass
class PartRelationshipInspector:
    """
    Tells what a part-whole relationship field of an annotation type asks for: which
    type of part, how many, and whether they are optional.

    Only covers relationships whose parts have a body of their own, so that they can be
    split off the mesh of the annotation's root body.
    """

    relationship_field: WrappedField
    """
    The field of the annotation type that holds the parts.
    """
    name_override: str | None = None
    holds_many_override: bool | None = None
    may_be_empty_override: bool | None = None

    @classmethod
    def all_of_type(
        cls, whole_type: type[PartWholeRelationship]
    ) -> list[PartRelationshipInspector]:
        """
        :param whole_type: A type of annotation.
        :return: Its part-whole relationships whose parts have a body of their own, in
            declaration order.
        """
        relationships = []

        for wrapped_field in WrappedClass(whole_type).fields_with_metadata(
            IsPartWholeRelationship
        ):
            if not issubclass(wrapped_field.type_endpoint, HasRootBody):
                continue

            # HasHandle stores its handle in the inherited `handles` list,
            # but exposes a singular optional `handle` relationship.
            if (
                issubclass(whole_type, HasHandle)
                and wrapped_field.field.name == "handles"
            ):
                relationships.append(
                    cls(
                        relationship_field=wrapped_field,
                        name_override="handle",
                        holds_many_override=False,
                        may_be_empty_override=True,
                    )
                )
            else:
                relationships.append(
                    cls(relationship_field=wrapped_field)
                )

        return relationships

    @classmethod
    def named(
        cls, whole_type: type[PartWholeRelationship], name: str
    ) -> PartRelationshipInspector:
        """
        :param whole_type: A type of annotation.
        :param name: The name of one of its part-whole relationship fields.
        :return: That relationship.
        """
        relationships = cls.all_of_type(whole_type)
        for relationship in relationships:
            if relationship.name == name:
                return relationship
        raise UnknownPartWholeRelationshipField(
            whole_type, name, [relationship.name for relationship in relationships]
        )

    @classmethod
    def for_part(
        cls,
        whole_type: type[PartWholeRelationship],
        part_type: type[HasRootBody],
        name: str = "",
    ) -> PartRelationshipInspector:
        """
        :param whole_type: A type of annotation.
        :param part_type: A type of part.
        :param name: The name of the relationship field the part goes to; found by the
            part's type if not given.
        :return: The relationship a part of that type goes to.
        :raises NoSinglePartWholeRelationshipField: If no name is given and not exactly
            one relationship holds parts of that type.
        """
        if name:
            return cls.named(whole_type, name)
        matching = [
            relationship
            for relationship in cls.all_of_type(whole_type)
            if issubclass(part_type, relationship.part_type)
        ]
        if len(matching) != 1:
            raise NoSinglePartWholeRelationshipField(
                whole_type, part_type, [relationship.name for relationship in matching]
            )
        return matching[0]

    @property
    def name(self) -> str:
        """
        :return: The name of the field that holds the parts.
        """
        if self.name_override is not None:
            return self.name_override
        return self.relationship_field.field.name

    @property
    def part_type(self) -> type[HasRootBody]:
        """
        :return: The type of annotation each part is.
        """
        return self.relationship_field.type_endpoint

    @property
    def part_name(self) -> str:
        """
        :return: What a part is called in plain words, such as ``"handle"``.
        """
        return camel_case_to_words(self.part_type.__name__)

    @property
    def holds_many(self) -> bool:
        """
        :return: Whether the relationship holds any number of parts rather than one.
        """
        if self.holds_many_override is not None:
            return self.holds_many_override

        return self.relationship_field.is_many_to_many_relationship

    @property
    def may_be_empty(self) -> bool:
        """
        :return: Whether an annotation may lack the parts of this relationship, which is
            the case for an optional part and for any number of parts.
        """
        if self.may_be_empty_override is not None:
            return self.may_be_empty_override

        return self.relationship_field.is_optional or self.holds_many

    def parts_of(self, whole: PartWholeRelationship) -> list[HasRootBody]:
        """
        :param whole: An annotation of the type this relationship belongs to.
        :return: The parts the annotation has in this relationship.
        """
        value = getattr(whole, self.name)
        if self.holds_many:
            return list(value)
        return [] if value is None else [value]

    def is_filled_in(self, whole: PartWholeRelationship) -> bool:
        """
        :param whole: An annotation of the type this relationship belongs to.
        :return: Whether the annotation has a part in this relationship already.
        """
        return bool(self.parts_of(whole))
