"""
Finishing the domain objects read back from DAOs, once every object they refer to is
populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from typing_extensions import TYPE_CHECKING

from krrood.ormatic.collection_shape import CollectionShape
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.base import DomainObject
from krrood.ormatic.data_access_objects.schema import get_dao_schema

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject
    from krrood.ormatic.data_access_objects.from_dao import (
        FromDataAccessObjectState,
        FromDataAccessObjectWorkItem,
    )


# %% post initialization


@lru_cache(maxsize=None)
def _has_post_init(clazz: type) -> bool:
    """
    :param clazz: The class to check.
    :return: Whether the class defines a ``__post_init__``.
    """
    return hasattr(clazz, "__post_init__")


# %% finishing domain objects


@dataclass(kw_only=True)
class DomainObjectFinisher:
    """
    Finishes the domain objects one conversion from DAOs read back.
    """

    state: FromDataAccessObjectState
    """
    The conversion state.
    """

    discovery_order: list[FromDataAccessObjectWorkItem]
    """
    The work items of the conversion, in the order their DAOs were discovered.
    """

    def finish_containers(self) -> None:
        """
        Convert the lists every domain object holds while it is read back to the
        dictionaries and container classes its fields declare.
        """
        processed_ids = set()
        for work_item in self.discovery_order:
            domain_object = self.state.get(source=work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                self._finish_association_collections(
                    data_access_object=work_item.dao_instance,
                    domain_object=domain_object,
                )
                self._restore_container_types(domain_object=domain_object)
                processed_ids.add(id(domain_object))

    @staticmethod
    def _finish_association_collections(
        *, data_access_object: DataAccessObject, domain_object: DomainObject
    ) -> None:
        """
        Turn the entries of every keyed container read back from an association table
        into a dictionary, now that the objects they hold are complete.

        :param data_access_object: The DAO the domain object was read back from.
        :param domain_object: The domain object.
        """
        for relationship in get_dao_schema(
            type(data_access_object)
        ).collection_relationships:
            if not relationship.has_keyed_level:
                continue
            value = getattr(domain_object, relationship.key, None)
            if isinstance(value, list):
                # object.__setattr__ (not setattr) so this also works on frozen dataclasses.
                object.__setattr__(
                    domain_object,
                    relationship.key,
                    relationship.loaded_shape.finish_loaded_collection(value=value),
                )

    @staticmethod
    def _restore_container_types(*, domain_object: DomainObject) -> None:
        """
        Convert the lists of every collection field, nested ones included, to the sets,
        tuples and other classes its type hint declares.
        """
        for collection_field in CollectionShape.fields_with_containers_to_restore(
            clazz=type(domain_object)
        ):
            value = getattr(domain_object, collection_field.attribute_name, None)
            restored = collection_field.shape.restore_container_types(value=value)
            if restored is not value:
                # object.__setattr__ (not setattr) so this also works on frozen dataclasses.
                object.__setattr__(
                    domain_object, collection_field.attribute_name, restored
                )

    def call_post_inits(self) -> None:
        """
        Call ``__post_init__`` on every domain object not created by an alternative
        mapping, whose constructor already called it.
        """
        processed_ids = set()
        for work_item in self.discovery_order:
            # Skip post_init for objects that were created via AlternativeMapping
            # because they are created via their constructor, which already
            # calls __post_init__.
            if issubclass(
                work_item.dao_instance.constructable_original_class(),
                AlternativeMapping,
            ):
                continue

            domain_object = self.state.get(source=work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                if _has_post_init(type(domain_object)):
                    domain_object.__post_init__()
                processed_ids.add(id(domain_object))
