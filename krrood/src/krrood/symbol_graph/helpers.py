from dataclasses import is_dataclass, fields

from typing_extensions import Optional, Any, Type

from ..class_diagrams.class_diagram import WrappedClass, ParseError
from ..class_diagrams.failures import ClassIsUnMappedInClassDiagram
from ..class_diagrams.wrapped_field import WrappedField
from .symbol_graph import SymbolGraph


def get_field_type_endpoint(owner_class: Type, field_name: str) -> Optional[Type]:
    """
    :param owner_class: The class of that owns the field.
    :param field_name: The name of the field.
    :return: The type of the field from the class diagram.
    """
    wrapped_field = get_wrapped_field(owner_class, field_name)
    if wrapped_field is None:
        return None
    try:
        return wrapped_field.type_endpoint
    except (AttributeError, RuntimeError):
        return None


def get_wrapped_field(owner_class: Type, field_name: str) -> Optional[WrappedField]:
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


def get_dataclass_field(owner_class: Type, field_name: str) -> Optional[Any]:
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


def get_wrapped_class(owner_class: Type) -> Optional[WrappedClass]:
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
