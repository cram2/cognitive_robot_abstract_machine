from __future__ import annotations

from inspect import isclass
from types import ModuleType
from typing import Tuple, List, Type, Dict

from sqlalchemy.orm import DeclarativeBase

from .dao import AlternativeMapping, DataAccessObject
from .utils import classes_of_module, is_direct_subclass


def get_classes_of_ormatic_interface(
    interface: ModuleType,
) -> Tuple[List[Type], List[Type[AlternativeMapping]], Dict]:
    """
    Get all classes and alternative mappings of an existing ormatic interface.

    :param interface: The ormatic interface to extract the information from.
    :return: A list of classes and a list of alternative mappings used in the interface.
    """
    classes = []
    alternative_mappings = []
    classes_of_ormatic_interface = classes_of_module(interface)
    type_mappings = {}

    for cls in filter(
        lambda x: issubclass(x, DataAccessObject) and isclass(x.original_class()),
        classes_of_ormatic_interface,
    ):
        original_class = cls.original_class()

        if issubclass(original_class, AlternativeMapping):
            alternative_mappings.append(original_class)
            classes.append(original_class.original_class())
        else:
            classes.append(original_class)

    # get the type mappings from the direct subclass of declarative base
    for cls in filter(
        lambda x: is_direct_subclass(x, DeclarativeBase), classes_of_ormatic_interface
    ):
        type_mappings.update(cls.type_mappings)

    return classes, alternative_mappings, type_mappings
