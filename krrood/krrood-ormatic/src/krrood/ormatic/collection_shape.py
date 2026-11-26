"""
The nesting of the collection a field holds: its containers from the outermost inwards,
and the type of the values at the bottom.
"""

from __future__ import annotations

import collections.abc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from types import NoneType, UnionType

from typing_extensions import Any, Iterable, TypeVar, Union, get_args, get_origin

from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.class_diagrams.utils import common_base_class

# %% column names


@dataclass(kw_only=True, frozen=True)
class LevelColumnNames:
    """
    The names of the columns of an association table that describe one level of a
    collection.
    """

    depth: int
    """
    The depth of the level, 1 for the outermost container.
    """

    @property
    def index(self) -> str:
        """
        :return: The name of the column holding the position of an element within its
            container of the level.
        """
        return f"index_{self.depth}"

    @property
    def key(self) -> str:
        """
        :return: The name of the column, or relationship, holding the key of an element
            within its container of the level.
        """
        return f"key_{self.depth}"


class AssociationLeafAttribute(StrEnum):
    """
    The attributes of an association table that hold the value of a row.
    """

    TARGET = "target"
    """
    The relationship to the row of a mapped object.
    """

    VALUE = "value"
    """
    The column holding a value no table maps.
    """


# %% levels


@dataclass(kw_only=True, frozen=True)
class CollectionLevel(ABC):
    """
    One container in the nesting of a collection.

    While an association table is read back, a container of any level is a list, so
    that the objects it holds need not be complete to be added; see
    :meth:`finish_loaded_container`.
    """

    container_type: type
    """
    The class a container of this level is restored as.
    """

    depth: int
    """
    The position of the level in the nesting, 1 for the outermost container.
    """

    @property
    def column_names(self) -> LevelColumnNames:
        """
        :return: The names of the columns of an association table that describe this
            level.
        """
        return LevelColumnNames(depth=self.depth)

    @property
    @abstractmethod
    def finished_type(self) -> type:
        """
        :return: The class, list or dictionary, of a container of this level once read
            back and finished.
        """

    @abstractmethod
    def entries(self, *, container: Any) -> Iterable[tuple[Any, Any]]:
        """
        :param container: A container of this level.
        :return: The key of every element, None for an unkeyed container, with the
            element, in the container's order.
        """

    @abstractmethod
    def add_loaded_element(self, *, container: list, key: Any, element: Any) -> int:
        """
        Add an element to a container of this level being read back.

        :param container: The container being read back.
        :param key: The key of the element, None for an unkeyed container.
        :param element: The element.
        :return: The position of the element in the container.
        """

    @abstractmethod
    def loaded_element_path(self, *, position: int) -> tuple[int, ...]:
        """
        :param position: The position of an element in a container being read back.
        :return: The indices leading from the container to the element.
        """

    @abstractmethod
    def finish_loaded_container(self, *, container: list) -> list | dict:
        """
        :param container: A container of this level, read back, whose objects are
            complete.
        :return: The container as :attr:`finished_type`.
        """

    @abstractmethod
    def restore_container_type(self, *, elements: list | dict) -> Any:
        """
        :param elements: A finished container, its elements already restored.
        :return: The container as :attr:`container_type`.
        """


@dataclass(kw_only=True, frozen=True)
class SequenceLevel(CollectionLevel):
    """
    A container of elements without keys, such as a list or a set; a set's elements
    keep the order it iterates them in.
    """

    @property
    def finished_type(self) -> type:
        return list

    def entries(self, *, container: Any) -> Iterable[tuple[Any, Any]]:
        return ((None, element) for element in container)

    def add_loaded_element(self, *, container: list, key: Any, element: Any) -> int:
        container.append(element)
        return len(container) - 1

    def loaded_element_path(self, *, position: int) -> tuple[int, ...]:
        return (position,)

    def finish_loaded_container(self, *, container: list) -> list:
        return container

    def restore_container_type(self, *, elements: list) -> Any:
        if self.container_type is list:
            return elements
        return self.container_type(elements)


@dataclass(kw_only=True, frozen=True)
class MappingLevel(CollectionLevel):
    """
    A container of elements by key, such as a dictionary.

    Read back, it is a list of ``[key, element]`` entries until it is finished, since
    the key of an entry is hashed only once it is complete.
    """

    key_type: Any
    """
    The type of the keys.
    """

    @property
    def finished_type(self) -> type:
        return dict

    def entries(self, *, container: Any) -> Iterable[tuple[Any, Any]]:
        return container.items()

    def add_loaded_element(self, *, container: list, key: Any, element: Any) -> int:
        container.append([key, element])
        return len(container) - 1

    def loaded_element_path(self, *, position: int) -> tuple[int, ...]:
        return position, 1

    def loaded_key_path(self, *, position: int) -> tuple[int, ...]:
        """
        :param position: The position of an entry in a container being read back.
        :return: The indices leading from the container to the entry's key.
        """
        return position, 0

    def finish_loaded_container(self, *, container: list) -> dict:
        return {key: element for key, element in container}

    def restore_container_type(self, *, elements: dict) -> Any:
        if self.container_type is dict:
            return elements
        return self.container_type(elements)


# %% shape


@dataclass(kw_only=True, frozen=True)
class CollectionShape:
    """
    The containers of a collection, from the outermost inwards, and the type of the
    values the innermost containers hold.
    """

    levels: tuple[CollectionLevel, ...]
    """
    The containers, the outermost first.
    """

    leaf_type: Any
    """
    The type of the values the innermost containers hold.
    """

    @classmethod
    def from_annotation(cls, *, annotation: Any) -> CollectionShape | None:
        """
        :param annotation: The annotation of a field.
        :return: The shape of the collection the annotation declares, or None if it
            declares none; an unparameterized sequence holds values of any type, and a
            dictionary needs its key and value types.
        """
        return cls._from_annotation_at_depth(annotation=annotation, depth=1)

    @classmethod
    def _from_annotation_at_depth(
        cls, *, annotation: Any, depth: int
    ) -> CollectionShape | None:
        """
        :param annotation: The annotation of a container at *depth*, or of a leaf.
        :param depth: The depth a container declared by *annotation* has.
        :return: The shape from *depth* inwards, or None if *annotation* declares no
            parameterized collection.
        """
        level_and_element = cls._level_and_element_type(
            annotation=annotation, depth=depth
        )
        if level_and_element is None:
            return None
        level, element_type = level_and_element
        inner = cls._from_annotation_at_depth(annotation=element_type, depth=depth + 1)
        if inner is None:
            return cls(
                levels=(level,), leaf_type=cls._leaf_type(annotation=element_type)
            )
        return cls(levels=(level, *inner.levels), leaf_type=inner.leaf_type)

    @staticmethod
    def _level_and_element_type(
        *, annotation: Any, depth: int
    ) -> tuple[CollectionLevel, Any] | None:
        """
        :param annotation: An annotation.
        :param depth: The depth of the container the annotation declares.
        :return: The level the annotation declares with the type of its elements, or
            None if it declares no parameterized collection.
        """
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin is None and annotation in CollectionKind.ELEMENT_SEQUENCE.value:
            return SequenceLevel(container_type=annotation, depth=depth), Any
        if origin is None or not arguments:
            return None
        if origin in CollectionKind.MAPPING.value or origin in (
            collections.abc.Mapping,
            collections.abc.MutableMapping,
        ):
            if len(arguments) != 2:
                return None
            key_type, value_type = arguments
            return MappingLevel(
                container_type=dict, depth=depth, key_type=key_type
            ), value_type
        if origin in CollectionKind.ELEMENT_SEQUENCE.value:
            container_type = origin
        elif origin in (collections.abc.Sequence, collections.abc.MutableSequence):
            container_type = list
        elif origin in (collections.abc.Set, collections.abc.MutableSet):
            container_type = set
        else:
            return None
        element_types = [
            element_type
            for element_type in dict.fromkeys(arguments)
            if element_type is not Ellipsis
        ]
        if len(element_types) == 1:
            element_type = element_types[0]
        else:
            element_type = (
                common_base_class(element_types) or Union[tuple(element_types)]
            )
        return SequenceLevel(container_type=container_type, depth=depth), element_type

    @staticmethod
    def _leaf_type(*, annotation: Any) -> Any:
        """
        :param annotation: The annotation of the values the innermost containers hold.
        :return: The type of the values: a bounded type variable's bound, and a union's
            lowest common base class of its members other than None, or the union of
            those members if they have none.
        """
        if isinstance(annotation, TypeVar) and annotation.__bound__ is not None:
            return annotation.__bound__
        if get_origin(annotation) in (Union, UnionType):
            members = [
                member for member in get_args(annotation) if member is not NoneType
            ]
            if len(members) == 1:
                return members[0]
            return common_base_class(members) or Union[tuple(members)]
        return annotation

    @property
    def leaf_types(self) -> tuple[Any, ...]:
        """
        :return: The members of :attr:`leaf_type` if it is a union, else the leaf type.
        """
        if get_origin(self.leaf_type) in (Union, UnionType):
            return get_args(self.leaf_type)
        return (self.leaf_type,)

    @property
    def key_types(self) -> tuple[Any, ...]:
        """
        :return: The types of the keys of every keyed level, the outermost first.
        """
        return tuple(
            level.key_type for level in self.levels if isinstance(level, MappingLevel)
        )

    @property
    def has_containers_to_restore(self) -> bool:
        """
        :return: Whether a container of the collection is of another class than the
            lists and dictionaries it is read back as.
        """
        return any(level.container_type not in (list, dict) for level in self.levels)

    @property
    def is_nested(self) -> bool:
        """
        :return: Whether a container holds other containers, which may be empty.
        """
        return len(self.levels) > 1

    @property
    def has_keyed_level(self) -> bool:
        """
        :return: Whether a container of the collection holds its elements by key.
        """
        return any(isinstance(level, MappingLevel) for level in self.levels)

    def finish_loaded_collection(self, *, value: list) -> list | dict:
        """
        :param value: A collection of this shape read back from an association table,
            whose objects are complete.
        :return: The collection with every container finished as a list or dictionary.
        """
        return self._finish_from_level(value=value, level_index=0)

    def _finish_from_level(self, *, value: list, level_index: int) -> list | dict:
        """
        :param value: A container of the level at *level_index*, read back.
        :param level_index: The index of the level in :attr:`levels`.
        :return: The container and the containers it holds, finished.
        """
        level = self.levels[level_index]
        if level_index + 1 < len(self.levels):
            if isinstance(level, MappingLevel):
                value = [
                    [
                        key,
                        self._finish_from_level(
                            value=element, level_index=level_index + 1
                        ),
                    ]
                    for key, element in value
                ]
            else:
                value = [
                    self._finish_from_level(value=element, level_index=level_index + 1)
                    for element in value
                ]
        return level.finish_loaded_container(container=value)

    def restore_container_types(self, *, value: Any) -> Any:
        """
        :param value: A collection of this shape whose containers are lists and
            dictionaries, as read back from the database.
        :return: The collection with every container of its annotated class; a value
            that is not a list or dictionary where one is expected is returned as it is.
        """
        return self._restore_from_level(value=value, level_index=0)

    def _restore_from_level(self, *, value: Any, level_index: int) -> Any:
        """
        :param value: A container of the level at *level_index*.
        :param level_index: The index of the level in :attr:`levels`.
        :return: The container and the containers it holds as their annotated classes.
        """
        if level_index == len(self.levels):
            return value
        level = self.levels[level_index]
        if type(value) is not level.finished_type:
            return value
        if isinstance(level, MappingLevel):
            elements = {
                key: self._restore_from_level(
                    value=element, level_index=level_index + 1
                )
                for key, element in value.items()
            }
        else:
            elements = [
                self._restore_from_level(value=element, level_index=level_index + 1)
                for element in value
            ]
        return level.restore_container_type(elements=elements)
