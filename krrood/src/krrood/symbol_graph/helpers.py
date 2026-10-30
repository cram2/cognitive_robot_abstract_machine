import inspect
from dataclasses import is_dataclass, fields
from types import NoneType

from typing_extensions import Any, TypeVar

from krrood import logger

try:
    from krrood.symbolic_math.symbolic_math import Scalar as SymbolicScalar
except ImportError as e:
    # This was added because casadi (used by symbolic_math) has no distribution for Windows platform.
    SymbolicScalar = NoneType
    logger.debug(f"SymbolicScalar is not available, importing it raised an error : {e}")

from krrood.class_diagrams.class_diagram import WrappedClass, ParseError
from krrood.class_diagrams.exceptions import (
    ClassIsUnMappedInClassDiagram,
    CouldNotResolveType,
)
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.symbol_graph.symbol_graph import SymbolGraph


def get_method_return_type(owner_class: type, method_name: str) -> type | None:
    """
    :param owner_class: The class that owns the method.
    :param method_name: The name of the method.
    :return: The type the method is annotated to return, or ``None`` when *owner_class* has
        no such method, the method carries no return annotation, or that annotation cannot
        be resolved.
    """
    if owner_class is None:
        return None
    method = inspect.getattr_static(owner_class, method_name, None)
    if not inspect.isfunction(method):
        return None
    try:
        return get_type_hints_of_object(method).get("return")
    except CouldNotResolveType:
        return None


def get_field_type_endpoint(owner_class: type, field_name: str) -> type | None:
    """
    :param owner_class: The class of that owns the field.
    :param field_name: The name of the field.
    :return: The type of the field from the class diagram.
    """
    if owner_class is None:
        return None
    if isinstance(owner_class, TypeVar):
        owner_class = owner_class.__bound__
        if owner_class is None:
            return None
    wrapped_field = get_wrapped_field(owner_class, field_name)
    if wrapped_field is None:
        # Static (non-triggering) lookup across the MRO, so a property inherited from a
        # base class resolves the same as one declared directly on ``owner_class``.
        try:
            prop = inspect.getattr_static(owner_class, field_name)
        except AttributeError:
            prop = None
        if not isinstance(prop, property) or prop.fget is None:
            return None
        try:
            return_type = get_type_hints_of_object(prop.fget).get("return")
        except CouldNotResolveType:
            return None
        if return_type is None:
            return None

        # TODO: Why do we have a special handling for this?
        if (SymbolicScalar is not NoneType and return_type is SymbolicScalar) or (
            isinstance(return_type, type) and issubclass(return_type, SymbolicScalar)
        ):
            return float
        return return_type
    try:
        return wrapped_field.type_endpoint
    except (AttributeError, RuntimeError):
        return None


def get_wrapped_field(owner_class: type, field_name: str) -> WrappedField | None:
    """
    :param owner_class: The class of that owns the field.
    :param field_name: The name of the field.
    :return: The wrapped field of the class from the symbol graph if it exists, otherwise create a new
    wrapped field instance for it.
    """
    wrapped_class = get_wrapped_class(owner_class)
    if wrapped_class is None:
        return None
    try:
        _ = wrapped_class.fields
        return wrapped_class._wrapped_field_name_map_.get(field_name, None)
    except ParseError:
        data_class_field = get_dataclass_field(owner_class, field_name)
        return (
            WrappedField(wrapped_class, data_class_field) if data_class_field else None
        )


def get_dataclass_field(owner_class: type, field_name: str) -> Any | None:
    """
    :param owner_class: The class of that owns the field.
    :param field_name: The name of the field.
    :return: The dataclass field of the class if it exists, otherwise None.
    """
    if not is_dataclass(owner_class):
        return None
    return next(
        (f for f in fields(owner_class) if f.name == field_name),
        None,
    )


def get_wrapped_class(owner_class: type) -> WrappedClass | None:
    """
    :return: The wrapped owner class of the class from the symbol graph if it exists, otherwise create a new
    wrapped class instance for it.
    """
    if not is_dataclass(owner_class):
        return None
    try:
        return SymbolGraph().class_diagram.get_wrapped_class(owner_class)
    except ClassIsUnMappedInClassDiagram:
        wrapped_cls = WrappedClass(owner_class)
        wrapped_cls._class_diagram = SymbolGraph().class_diagram
        return wrapped_cls
