from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from inspect import isclass
from types import ModuleType

import sqlalchemy

from krrood.ormatic.base import Base
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.utils import classes_of_module

# %% what an interface holds


@dataclass(kw_only=True)
class OrmaticInterfaceInformation:
    """
    Classes, alternative mappings, type mappings, and externally-mapped classes read
    from an existing ormatic interface.
    """

    classes: list[type] = field(default_factory=list)
    """
    The domain classes mapped in the interface.
    """

    alternative_mappings: list[type[AlternativeMapping]] = field(default_factory=list)
    """
    The alternative mappings used in the interface.
    """

    type_mappings: dict = field(default_factory=dict)
    """
    The type mappings used in the interface.
    """

    externally_mapped_classes: dict[type, type] = field(default_factory=dict)
    """
    Maps every domain class already mapped anywhere in this interface's dependency chain
    to the :class:`DataAccessObject` subclass that maps it, so a dependent interface can
    import and reuse that DAO instead of regenerating it.
    """


# %% what an interface builds on


@dataclass(kw_only=True)
class OrmaticInterfaceChain:
    """
    Loaded ormatic interfaces together with every loaded interface they build on.

    An interface builds on another when one of its tables inherits from, or relates
    to, a table of the other.
    """

    interfaces: list[ModuleType]
    """
    The interfaces the chain starts from.
    """

    @cached_property
    def module_names(self) -> set[str]:
        """
        :return: The names of the modules of the interfaces and of every interface they
            build on, directly or through other interfaces.
        """
        mapped_classes_by_module: dict[str, list[type]] = {}
        for mapper in Base.registry.mappers:
            mapped_classes_by_module.setdefault(mapper.class_.__module__, []).append(
                mapper.class_
            )
        module_names = {interface.__name__ for interface in self.interfaces}
        pending = list(module_names)
        while pending:
            for mapped_class in mapped_classes_by_module.get(pending.pop(), []):
                mapper = sqlalchemy.inspect(mapped_class)
                referenced_classes = [
                    ancestor.class_ for ancestor in mapper.iterate_to_root()
                ] + [
                    relationship.mapper.class_ for relationship in mapper.relationships
                ]
                for referenced_class in referenced_classes:
                    if referenced_class.__module__ in module_names:
                        continue
                    module_names.add(referenced_class.__module__)
                    pending.append(referenced_class.__module__)
        return module_names

    @cached_property
    def data_access_objects(self) -> list[type[DataAccessObject]]:
        """
        :return: The data access objects of the interfaces in the chain.
        """
        return [
            mapper.class_
            for mapper in Base.registry.mappers
            if issubclass(mapper.class_, DataAccessObject)
            and mapper.class_.__module__ in self.module_names
        ]

    @cached_property
    def alternative_mappings(self) -> list[type[AlternativeMapping]]:
        """
        :return: The alternative mappings the interfaces in the chain map.
        """
        return [
            data_access_object.original_class()
            for data_access_object in self.data_access_objects
            if isclass(data_access_object.original_class())
            and issubclass(data_access_object.original_class(), AlternativeMapping)
        ]


# %% reading an interface


def get_classes_of_ormatic_interface(
    interface: ModuleType,
) -> OrmaticInterfaceInformation:
    """
    Get all classes and alternative mappings of an existing ormatic interface.

    :param interface: The ormatic interface to extract the information from.
    """
    from krrood.ormatic.ormatic import ORMatic

    information = OrmaticInterfaceInformation()
    classes_of_ormatic_interface = classes_of_module(interface)

    for cls in filter(
        lambda x: issubclass(x, DataAccessObject), classes_of_ormatic_interface
    ):
        original_class = cls.original_class()

        if not isclass(original_class):
            continue

        if issubclass(original_class, AlternativeMapping):
            information.alternative_mappings.append(original_class)
            information.classes.append(original_class.original_class())
        else:
            information.classes.append(original_class)

    for dao_class in OrmaticInterfaceChain(interfaces=[interface]).data_access_objects:
        ORMatic.register_externally_mapped_class(
            dao_class=dao_class,
            externally_mapped_classes=information.externally_mapped_classes,
        )

    # every interface shares one Base, so its type_mappings already include everything upstream
    information.type_mappings.update(interface.Base.type_mappings)

    return information
