"""
The relationships of a DAO class, as the conversion between DAOs and domain objects
reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import count

import sqlalchemy.inspection
from sqlalchemy.orm import MANYTOONE, ONETOMANY, RelationshipProperty
from typing_extensions import TYPE_CHECKING, Any

from krrood.ormatic.collection_shape import (
    AssociationLeafAttribute,
    CollectionLevel,
    CollectionShape,
    LevelColumnNames,
    MappingLevel,
    SequenceLevel,
)

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


# %% association rows


class AssociationDataAccessObject:
    """
    Base class for association objects in the Data Access Object layer.
    Association objects are used to map many-to-many relationships that
    require additional information or identity for each association,
    such as when duplicates are allowed in a collection.
    """

    @property
    def target(self) -> DataAccessObject:
        """
        :return: The target Data Access Object of this association.
        """
        raise NotImplementedError

    @target.setter
    def target(self, value: DataAccessObject) -> None:
        """
        :param value: The target Data Access Object of this association.
        """
        raise NotImplementedError


# %% single relationships


class RelationshipInformationKey(StrEnum):
    """
    Keys of the information a generated relationship carries for the conversion between
    DAOs and domain objects.
    """

    DECLARED_TYPE = "krrood_declared_type"
    """
    The type a field is declared with, for a relationship that targets the table of
    every stored object because objects of that type may lie outside the type's table
    hierarchy.
    """


@dataclass(kw_only=True, frozen=True)
class SingleRelationship:
    """
    Metadata for a single-valued (many-to-one or one-to-one) DAO relationship.
    """

    key: str
    """The attribute name of the relationship on the DAO."""

    domain_type: type
    """The expected domain type of the related object."""

    foreign_key_attribute: str | None = None
    """
    The attribute name of the local foreign key column for many-to-one
    relationships, or ``None`` if the foreign key is on the remote side.
    """

    target_dao_class: type | None = None
    """The DAO class this relationship points to."""

    @staticmethod
    def is_single_valued(*, relationship: RelationshipProperty) -> bool:
        """
        :param relationship: A relationship of a DAO class.
        :return: Whether the relationship holds at most one object.
        """
        return relationship.direction == MANYTOONE or (
            relationship.direction == ONETOMANY and not relationship.uselist
        )

    @staticmethod
    def declared_type(*, relationship: RelationshipProperty) -> type:
        """
        :param relationship: A relationship of a DAO class.
        :return: The domain type of the objects the relationship holds.
        """
        declared_type = relationship.info.get(RelationshipInformationKey.DECLARED_TYPE)
        if declared_type is not None:
            return declared_type
        return relationship.mapper.class_.original_class()

    @classmethod
    def from_relationship_property(
        cls, *, relationship: RelationshipProperty, domain_type: type
    ) -> SingleRelationship:
        """
        Describe a relationship, recording the local foreign key of a many-to-one
        relationship so that :func:`~krrood.ormatic.data_access_objects.relationship_loading.read_single_relationship`
        resolves the related object from the identity map instead of a lazy load.

        :param relationship: The SQLAlchemy relationship property to inspect.
        :param domain_type: The expected domain type of the related object.
        :return: The description of the relationship.
        """
        local_columns = list(relationship.local_columns)
        foreign_key_attribute = (
            local_columns[0].key
            if relationship.direction == MANYTOONE and len(local_columns) == 1
            else None
        )
        return cls(
            key=relationship.key,
            domain_type=domain_type,
            foreign_key_attribute=foreign_key_attribute,
            target_dao_class=relationship.mapper.class_,
        )


# %% collection relationships


@dataclass(kw_only=True, frozen=True)
class AssociationLevel:
    """
    A level of a collection an association table holds, and the relationship to the
    objects of its keys if they are mapped.
    """

    collection_level: CollectionLevel
    """
    The level, whose containers are read back as lists and dictionaries.
    """

    key_relationship: SingleRelationship | None = None
    """
    The relationship to the object of a row's key, for a level keyed by mapped objects.
    """

    @property
    def is_keyed(self) -> bool:
        """
        :return: Whether the containers of the level hold their elements by key.
        """
        return isinstance(self.collection_level, MappingLevel)

    @property
    def column_names(self) -> LevelColumnNames:
        """
        :return: The names of the columns of the association table for this level.
        """
        return self.collection_level.column_names


@dataclass(kw_only=True, frozen=True)
class AssociationRowPosition:
    """
    Where a row of an association table places its value in one level of a collection.
    """

    level: AssociationLevel
    """
    The level.
    """

    index: int
    """
    The position of the element in its container of the level.
    """

    key: Any = None
    """
    The key of the element, for a keyed level.
    """


@dataclass(kw_only=True, frozen=True)
class CollectionRelationship:
    """
    Metadata for a collection-valued (one-to-many or many-to-many) DAO relationship.
    """

    key: str
    """The attribute name of the relationship on the DAO."""

    association_class: type | None
    """
    The association DAO class used as an intermediary, or ``None`` for direct
    collection relationships.
    """

    domain_type: type
    """
    The expected domain type of the values in the collection; :class:`object` for
    values no table maps.
    """

    association_target_relationship: SingleRelationship | None = None
    """
    The ``target`` relationship of the association class, if the values are mapped
    objects. Used to resolve association targets via the identity map instead of lazy
    loads.
    """

    target_dao_class: type | None
    """The DAO class of the values in the collection, if they are mapped objects."""

    levels: tuple[AssociationLevel, ...] = ()
    """
    The levels of the collection an association table holds, the outermost first.
    """

    holds_values_in_column: bool = False
    """
    Whether the association table holds each value in a column of its own rather than as
    a reference to a mapped object.
    """

    @classmethod
    def from_association_class(
        cls, *, key: str, association_class: type
    ) -> CollectionRelationship:
        """
        :param key: The attribute name of the relationship on the DAO.
        :param association_class: The association DAO class the relationship holds rows of.
        :return: The description of the collection whose values the rows hold, read from the
            columns and relationships of the association class.
        """
        relationships = sqlalchemy.inspection.inspect(association_class).relationships
        levels = cls._association_levels(association_class=association_class)
        if AssociationLeafAttribute.TARGET not in relationships:
            return cls(
                key=key,
                association_class=association_class,
                domain_type=object,
                target_dao_class=None,
                levels=levels,
                holds_values_in_column=True,
            )
        target_relationship = relationships[AssociationLeafAttribute.TARGET]
        domain_type = SingleRelationship.declared_type(relationship=target_relationship)
        return cls(
            key=key,
            association_class=association_class,
            domain_type=domain_type,
            association_target_relationship=SingleRelationship.from_relationship_property(
                relationship=target_relationship, domain_type=domain_type
            ),
            target_dao_class=target_relationship.mapper.class_,
            levels=levels,
        )

    @staticmethod
    def _association_levels(*, association_class: type) -> tuple[AssociationLevel, ...]:
        """
        :param association_class: An association DAO class.
        :return: The levels of the collection its rows describe, read from its index and
            key columns, the outermost first.
        """
        association_mapper = sqlalchemy.inspection.inspect(association_class)
        relationships = association_mapper.relationships
        levels = []
        for depth in count(1):
            column_names = LevelColumnNames(depth=depth)
            if column_names.index not in association_mapper.attrs:
                break
            if column_names.key not in association_mapper.attrs:
                levels.append(
                    AssociationLevel(
                        collection_level=SequenceLevel(container_type=list, depth=depth)
                    )
                )
                continue
            key_relationship = None
            if column_names.key in relationships:
                key_relationship_property = relationships[column_names.key]
                key_relationship = SingleRelationship.from_relationship_property(
                    relationship=key_relationship_property,
                    domain_type=SingleRelationship.declared_type(
                        relationship=key_relationship_property
                    ),
                )
            levels.append(
                AssociationLevel(
                    collection_level=MappingLevel(
                        container_type=dict, depth=depth, key_type=object
                    ),
                    key_relationship=key_relationship,
                )
            )
        return tuple(levels)

    @property
    def loaded_shape(self) -> CollectionShape:
        """
        :return: The shape of the collection as an association table holds it.
        """
        return CollectionShape(
            levels=tuple(level.collection_level for level in self.levels),
            leaf_type=self.domain_type,
        )

    @property
    def has_keyed_level(self) -> bool:
        """
        :return: Whether a container of the collection holds its elements by key.
        """
        return any(level.is_keyed for level in self.levels)

    def is_innermost(self, *, level: AssociationLevel) -> bool:
        """
        :param level: One of :attr:`levels`.
        :return: Whether the containers of the level hold the values of the collection
            rather than other containers.
        """
        return level is self.levels[-1]
