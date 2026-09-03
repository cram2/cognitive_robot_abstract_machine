from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isclass
from types import ModuleType
from typing import List, Type, Dict

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.utils import classes_of_module


@dataclass
class OrmaticInterfaceInformation:
    """
    Classes, alternative mappings, type mappings, and externally-mapped classes read
    from an existing ormatic interface.
    """

    classes: List[Type] = field(default_factory=list)
    """
    The domain classes mapped in the interface.
    """

    alternative_mappings: List[Type[AlternativeMapping]] = field(default_factory=list)
    """
    The alternative mappings used in the interface.
    """

    type_mappings: Dict = field(default_factory=dict)
    """
    The type mappings used in the interface.
    """

    externally_mapped_classes: Dict[Type, Type] = field(default_factory=dict)
    """
    Maps every domain class already mapped anywhere in this interface's dependency
    chain to the :class:`DataAccessObject` subclass that maps it, so a dependent
    interface can import and reuse that DAO instead of regenerating it.
    """


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

    # every interface in the chain shares one Base, so its registry already
    # transitively includes every DAO mapped anywhere upstream
    for mapper in interface.Base.registry.mappers:
        dao_class = mapper.class_
        if issubclass(dao_class, DataAccessObject):
            ORMatic.register_externally_mapped_class(
                dao_class, information.externally_mapped_classes
            )

    # every interface shares one Base, so its type_mappings already include everything upstream
    information.type_mappings.update(interface.Base.type_mappings)

    return information
