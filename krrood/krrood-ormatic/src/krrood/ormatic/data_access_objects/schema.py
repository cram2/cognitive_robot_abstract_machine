"""
The class level metadata of a DAO class that the conversion between DAOs and domain
objects reads, computed once per class.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy.orm import MANYTOMANY, ONETOMANY
from typing_extensions import TYPE_CHECKING, get_origin

from krrood.ormatic.data_access_objects.helper import get_part_class
from krrood.ormatic.data_access_objects.relationships import (
    AssociationDataAccessObject,
    CollectionRelationship,
    SingleRelationship,
)
from krrood.ormatic.utils import is_data_column

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
        PartDataAccessObject,
    )


# %% schema


@dataclass(kw_only=True, frozen=True)
class DataAccessObjectSchema:
    """
    Precomputed, class-level metadata used by the to_dao/from_dao hot paths.

    All information in here is derived from the SQLAlchemy mapper and the generic
    arguments of a DAO class. Computing it once per class avoids repeated mapper
    introspection, relationship classification and MRO walks per converted instance.
    """

    data_column_names: tuple[str, ...]
    """
    Names of all data columns (no primary keys, foreign keys or polymorphic markers).
    """

    single_relationships: tuple[SingleRelationship, ...]
    """
    Single-valued relationships of this DAO class.
    """

    collection_relationships: tuple[CollectionRelationship, ...]
    """
    Collection-valued relationships of this DAO class.
    """

    relationship_keys: tuple[str, ...]
    """
    The keys of all relationships of the DAO class.
    """

    part_classes: tuple[type[PartDataAccessObject], ...]
    """
    The classes of the parts holding the rows of the DAO's objects in the tables of
    their mapped base classes outside the DAO class's own tables.
    """

    shares_id_with_stored_object: bool
    """
    Whether the DAO's object takes its id from its row in the table of every stored
    object.
    """

    alternative_base: type | None
    """
    The first base class of the DAO that maps to an AlternativeMapping, if any.
    """

    uses_alternative_mapping: bool
    """
    Whether the DAO class itself maps to an AlternativeMapping.
    """


@lru_cache(maxsize=None)
def get_dao_schema(dao_class: type[DataAccessObject]) -> DataAccessObjectSchema:
    """
    Build (and cache) the conversion plan for a DAO class.

    :param dao_class: The DAO class to build the plan for.
    :return: The conversion plan.
    """
    mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(dao_class)

    single_relationships = []
    collection_relationships = []
    for relationship in mapper.relationships:
        if SingleRelationship.is_single_valued(relationship=relationship):
            single_relationships.append(
                SingleRelationship.from_relationship_property(
                    relationship=relationship,
                    domain_type=SingleRelationship.declared_type(
                        relationship=relationship
                    ),
                )
            )
        elif relationship.direction in (ONETOMANY, MANYTOMANY):
            target_dao_clazz = relationship.mapper.class_
            if issubclass(target_dao_clazz, AssociationDataAccessObject):
                collection_relationships.append(
                    CollectionRelationship.from_association_class(
                        key=relationship.key, association_class=target_dao_clazz
                    )
                )
            else:
                collection_relationships.append(
                    CollectionRelationship(
                        key=relationship.key,
                        association_class=None,
                        domain_type=target_dao_clazz.original_class(),
                        target_dao_class=target_dao_clazz,
                    )
                )

    return DataAccessObjectSchema(
        data_column_names=tuple(
            column.name for column in mapper.columns if is_data_column(column)
        ),
        single_relationships=tuple(single_relationships),
        collection_relationships=tuple(collection_relationships),
        relationship_keys=tuple(
            relationship.key for relationship in mapper.relationships
        ),
        part_classes=_part_classes(dao_class=dao_class, mapper=mapper),
        shares_id_with_stored_object=dao_class._shares_id_with_stored_object(),
        alternative_base=dao_class._find_alternative_mapping_base(),
        uses_alternative_mapping=dao_class._uses_alternative_mapping(),
    )


def _part_classes(
    *, dao_class: type[DataAccessObject], mapper: sqlalchemy.orm.Mapper
) -> tuple[type[PartDataAccessObject], ...]:
    """
    :param dao_class: A DAO class.
    :param mapper: The mapper of the DAO class.
    :return: The part classes of the mapped base classes of the DAO class's domain class
        whose tables lie outside the mapper's tables.
    """
    if not dao_class._shares_id_with_stored_object():
        return ()
    domain_class = dao_class.original_class()
    if dao_class._uses_alternative_mapping():
        domain_class = domain_class.original_class()
    domain_class = get_origin(domain_class) or domain_class
    result = []
    for base in domain_class.__mro__[1:]:
        part_class = get_part_class(base)
        if (
            part_class is not None
            and part_class not in result
            and not any(part_class.__table__ is table for table in mapper.tables)
        ):
            result.append(part_class)
    return tuple(result)


# %% partition between a DAO class and its alternatively mapped base class


@dataclass(kw_only=True, frozen=True)
class AlternativePartitionPlan:
    """
    Precomputed partition of columns and relationships between a DAO class and its
    alternatively mapped base class.
    """

    parent_data_column_names: tuple[str, ...]
    """
    Data column names that belong to the alternatively mapped base.
    """

    own_data_column_names: tuple[str, ...]
    """
    Data column names that belong to this DAO's own tables.
    """

    intermediate_attribute_keys: tuple[str, ...]
    """
    Column attribute keys of intermediate ancestors not covered by the parent.
    """

    parent_single_relationships: tuple[SingleRelationship, ...]
    """
    Single-valued relationships that belong to the alternatively mapped base.
    """

    parent_collection_relationships: tuple[CollectionRelationship, ...]
    """
    Collection relationships that belong to the alternatively mapped base.
    """

    own_single_relationships: tuple[SingleRelationship, ...]
    """
    Single-valued relationships that belong to this DAO's own tables, i.e. those not
    covered by the alternatively mapped base.
    """

    own_collection_relationships: tuple[CollectionRelationship, ...]
    """
    Collection relationships that belong to this DAO's own tables, i.e. those not
    covered by the alternatively mapped base.
    """


@lru_cache(maxsize=None)
def get_alternative_partition_plan(
    dao_class: type, alternative_base: type
) -> AlternativePartitionPlan:
    """
    Build (and cache) the column/relationship partition between a DAO class and its
    alternatively mapped base class.

    :param dao_class: The DAO class.
    :param alternative_base: The alternatively mapped base class.
    :return: The partition plan.
    """
    mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(dao_class)
    parent_mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(
        alternative_base
    )
    plan = get_dao_schema(dao_class)
    parent_plan = get_dao_schema(alternative_base)

    parent_column_names = {column.name for column in parent_mapper.columns}
    parent_relationship_keys = set(parent_plan.relationship_keys)

    return AlternativePartitionPlan(
        parent_data_column_names=parent_plan.data_column_names,
        own_data_column_names=tuple(
            column.name
            for column in mapper.columns
            if column.name not in parent_column_names and is_data_column(column)
        ),
        intermediate_attribute_keys=tuple(
            prop.key
            for prop in mapper.column_attrs
            if prop.key not in parent_column_names and is_data_column(prop.columns[0])
        ),
        parent_single_relationships=parent_plan.single_relationships,
        parent_collection_relationships=parent_plan.collection_relationships,
        own_single_relationships=tuple(
            entry
            for entry in plan.single_relationships
            if entry.key not in parent_relationship_keys
        ),
        own_collection_relationships=tuple(
            entry
            for entry in plan.collection_relationships
            if entry.key not in parent_relationship_keys
        ),
    )
