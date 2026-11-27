"""
Writing a collection as the rows of an association table, and reading the rows back
into the collection they describe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Any, Iterable

from krrood.class_diagrams.collection_kind import AnyCollection
from krrood.entity_query_language.core.mapped_variable import Attribute, IndexByValue
from krrood.ormatic.collection_shape import AssociationLeafAttribute
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.base import DomainObject
from krrood.ormatic.data_access_objects.relationship_loading import (
    read_single_relationship,
)
from krrood.ormatic.data_access_objects.relationships import (
    AssociationDataAccessObject,
    AssociationLevel,
    AssociationRowPosition,
    CollectionRelationship,
    SingleRelationship,
)

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject
    from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
    from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState


# %% writing association rows


@dataclass(kw_only=True)
class AssociationRowWriter:
    """
    Writes a collection as the rows of an association table, a row without value marking
    an empty inner container.
    """

    relationship: CollectionRelationship
    """
    The collection relationship the rows are written for.
    """

    holder: DataAccessObject
    """
    The DAO the collection belongs to.
    """

    state: ToDataAccessObjectState
    """
    The conversion state.
    """

    def write_rows(
        self, *, container: AnyCollection
    ) -> list[AssociationDataAccessObject]:
        """
        :param container: The collection.
        :return: The rows of the association table describing the collection.
        """
        return self._rows_of_container(container=container, positions=())

    def _rows_of_container(
        self,
        *,
        container: AnyCollection,
        positions: tuple[AssociationRowPosition, ...],
    ) -> list[AssociationDataAccessObject]:
        """
        :param container: A container of the collection, at the level after the levels
            of *positions*.
        :param positions: The positions of the container in the levels outside it.
        :return: The rows for the values of the container and of the containers it holds.
        """
        level = self.relationship.levels[len(positions)]
        is_innermost = self.relationship.is_innermost(level=level)
        rows = []
        for index, entry in enumerate(
            level.collection_level.entries(container=container)
        ):
            element_positions = (
                *positions,
                AssociationRowPosition(level=level, index=index, key=entry.key),
            )
            if not is_innermost and entry.element:
                rows.extend(
                    self._rows_of_container(
                        container=entry.element, positions=element_positions
                    )
                )
                continue
            row = self._placed_row(positions=element_positions)
            if is_innermost:
                self._set_value(row=row, value=entry.element)
            rows.append(row)
        return rows

    def _placed_row(
        self, *, positions: tuple[AssociationRowPosition, ...]
    ) -> AssociationDataAccessObject:
        """
        :param positions: The positions of the row's value, the outermost level first.
        :return: A row of the association table placed at the positions, without value.
        """
        row = self.relationship.association_class()
        for position in positions:
            column_names = position.level.column_names
            setattr(row, column_names.index, position.index)
            if position.level.key_relationship is not None:
                setattr(
                    row,
                    column_names.key,
                    self._referenced_dao(
                        source_object=position.key,
                        relationship=position.level.key_relationship,
                    ),
                )
            elif position.level.is_keyed:
                setattr(row, column_names.key, position.key)
        return row

    def _set_value(self, *, row: AssociationDataAccessObject, value: Any) -> None:
        """
        Set the value a row of the association table holds.

        :param row: The row.
        :param value: A value of the collection.
        """
        if self.relationship.holds_values_in_column:
            setattr(row, AssociationLeafAttribute.VALUE, value)
            return
        row.target = self.holder.get_or_queue_referenced_dao(
            source_object=value,
            state=self.state,
            expected_type=self.relationship.domain_type,
            target_dao_class=self.relationship.target_dao_class,
        )

    def _referenced_dao(
        self, *, source_object: Any, relationship: SingleRelationship
    ) -> DataAccessObject:
        """
        :param source_object: A key of the collection that a table maps.
        :param relationship: The relationship of the association table to the key.
        :return: What the relationship holds for the key.
        """
        return self.holder.get_or_queue_referenced_dao(
            source_object=source_object,
            state=self.state,
            expected_type=relationship.domain_type,
            target_dao_class=relationship.target_dao_class,
        )


# %% reading association rows


@dataclass(kw_only=True, frozen=True)
class LoadedContainerPlace:
    """
    A container of a collection being read back from an association table, and where it
    lies within its holder.
    """

    container: list
    """
    The container, a list while it is read back.
    """

    reference: Attribute | IndexByValue
    """
    The access path from the holder of the collection to the container.
    """

    path: tuple[int, ...] = ()
    """
    The positions of the container in the levels outside it.
    """


@dataclass(kw_only=True)
class AssociationRowReader:
    """
    Reads the rows of an association table back into the collection they describe,
    whose containers stay lists until the objects they hold are complete.
    """

    relationship: CollectionRelationship
    """
    The collection relationship holding the rows.
    """

    holder: DomainObject
    """
    The domain object the collection belongs to.
    """

    state: FromDataAccessObjectState
    """
    The conversion state.
    """

    collection: list = field(default_factory=list)
    """
    The collection read back so far.
    """

    places_by_path: dict[tuple[int, ...], LoadedContainerPlace] = field(
        default_factory=dict
    )
    """
    The inner containers read back so far, by their positions.
    """

    def read_rows(self, *, rows: Iterable[AssociationDataAccessObject]) -> list:
        """
        :param rows: The rows of the association table, in the order of their positions.
        :return: The collection the rows describe.
        """
        for row in rows:
            self._read_row(row=row)
        return self.collection

    def _read_row(self, *, row: AssociationDataAccessObject) -> None:
        """
        Add the value a row holds, or the empty container it marks, to the collection.

        :param row: A row of the association table.
        """
        place = LoadedContainerPlace(
            container=self.collection,
            reference=Attribute(_attribute_name_=self.relationship.key, _child_=None),
        )
        for level in self.relationship.levels:
            index = getattr(row, level.column_names.index)
            if index is None:
                return
            if self.relationship.is_innermost(level=level):
                self._add_element(
                    row=row, level=level, place=place, element=self._value(row=row)
                )
                return
            place = self._inner_place(row=row, level=level, place=place, index=index)

    def _inner_place(
        self,
        *,
        row: AssociationDataAccessObject,
        level: AssociationLevel,
        place: LoadedContainerPlace,
        index: int,
    ) -> LoadedContainerPlace:
        """
        :param row: A row of the association table.
        :param level: The level of the container at *place*.
        :param place: A container the row's value lies in.
        :param index: The position the row gives within the container at *place*.
        :return: The inner container at the position, added if this is its first row.
        """
        path = (*place.path, index)
        if path not in self.places_by_path:
            inner_place = self._add_element(
                row=row, level=level, place=place, element=[]
            )
            self.places_by_path[path] = LoadedContainerPlace(
                container=inner_place.container,
                reference=inner_place.reference,
                path=path,
            )
        return self.places_by_path[path]

    def _add_element(
        self,
        *,
        row: AssociationDataAccessObject,
        level: AssociationLevel,
        place: LoadedContainerPlace,
        element: Any,
    ) -> LoadedContainerPlace:
        """
        Add an element, with the key the row gives it, to the container at *place*.

        :param row: A row of the association table.
        :param level: The level of the container at *place*.
        :param place: The container.
        :param element: The element.
        :return: Where the element lies.
        """
        collection_level = level.collection_level
        key = self._key(row=row, level=level)
        position = collection_level.add_loaded_element(
            container=place.container, key=key, element=element
        )
        if level.is_keyed:
            self._remember_alternative_mapping(
                value=key,
                reference=self._reference_through(
                    reference=place.reference,
                    path=collection_level.loaded_key_path(position=position),
                ),
            )
        element_reference = self._reference_through(
            reference=place.reference,
            path=collection_level.loaded_element_path(position=position),
        )
        self._remember_alternative_mapping(value=element, reference=element_reference)
        return LoadedContainerPlace(container=element, reference=element_reference)

    def _key(self, *, row: AssociationDataAccessObject, level: AssociationLevel) -> Any:
        """
        :param row: A row of the association table.
        :param level: A level of the collection.
        :return: The key the row gives its element in the level, the domain object of
            a mapped key, None for an unkeyed level.
        """
        if level.key_relationship is not None:
            return read_single_relationship(row, level.key_relationship).from_dao(
                state=self.state
            )
        if level.is_keyed:
            return getattr(row, level.column_names.key)
        return None

    def _value(self, *, row: AssociationDataAccessObject) -> Any:
        """
        :param row: A row of the association table.
        :return: The value the row holds, the domain object of a mapped value.
        """
        if self.relationship.holds_values_in_column:
            return getattr(row, AssociationLeafAttribute.VALUE)
        return read_single_relationship(
            row, self.relationship.association_target_relationship
        ).from_dao(state=self.state)

    @staticmethod
    def _reference_through(
        *, reference: Attribute | IndexByValue, path: tuple[int, ...]
    ) -> Attribute | IndexByValue:
        """
        :param reference: The access path to a container.
        :param path: The indices leading from the container to one of its elements.
        :return: The access path to the element.
        """
        for index in path:
            reference = IndexByValue(_key_=index, _child_=reference)
        return reference

    def _remember_alternative_mapping(
        self, *, value: Any, reference: Attribute | IndexByValue
    ) -> None:
        """
        Remember to replace the value at the reference by its domain object, if it is an
        alternative mapping.

        :param value: A key or element read back.
        :param reference: The access path from the holder to the value.
        """
        if isinstance(value, AlternativeMapping):
            self.state._alternative_mappings_being_referenced[value].append(
                (self.holder, reference)
            )
