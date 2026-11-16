"""
Tests of looking up the types of a class's fields and properties in the class diagram.
"""

from dataclasses import dataclass

from typing_extensions import TypeVar

from krrood.entity_query_language.class_diagram_lookup import get_field_type_endpoint


@dataclass(kw_only=True)
class _Tool:
    """
    A dataclass that a type variable is bound to.
    """

    name: str
    """
    The field looked up through the type variable.
    """


_TTool = TypeVar("_TTool", bound=_Tool)


def test_get_field_type_endpoint_resolves_typevar_owner():
    """
    A type variable given as the owner class is resolved to its bound before the field
    type is looked up, as a generic attribute typed by a bound type variable needs.
    """
    concrete_field_type = get_field_type_endpoint(_Tool, "name")

    assert get_field_type_endpoint(_TTool, "name") == concrete_field_type


class _Named:
    """
    A class that declares a property.
    """

    @property
    def label(self) -> str:
        """
        The property looked up on this class and on its subclass.
        """
        return "unnamed"


class _NamedTool(_Named):
    """
    A class that inherits the property without redeclaring it.
    """


def test_get_field_type_endpoint_resolves_inherited_property():
    """
    A property defined only on a base class of the owner class resolves the same as one
    the owner class declares itself.
    """
    assert get_field_type_endpoint(_Named, "label") == str
    assert get_field_type_endpoint(_NamedTool, "label") == str
