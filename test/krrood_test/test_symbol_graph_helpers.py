from dataclasses import dataclass

from typing_extensions import TypeVar

from krrood.symbol_graph.helpers import get_field_type_endpoint


@dataclass
class _Tool:
    name: str


_TTool = TypeVar("_TTool", bound=_Tool)


def test_get_field_type_endpoint_resolves_typevar_owner():
    """
    A TypeVar ``owner_class`` must be resolved to its bound before looking up the field
    type.

    Generic attributes (for example ``Connection.parent`` typed as a bound TypeVar)
    reach the symbol-graph helper as a bare TypeVar, which previously raised
    ``AttributeError`` because a TypeVar has no ``__dict__``.
    """
    concrete_field_type = get_field_type_endpoint(_Tool, "name")

    assert get_field_type_endpoint(_TTool, "name") == concrete_field_type


class _Named:
    @property
    def label(self) -> str:
        return "unnamed"


class _NamedTool(_Named):
    pass


def test_get_field_type_endpoint_resolves_inherited_property():
    """
    A property that is only defined on a base class -- not on ``owner_class`` itself --
    must resolve the same as one declared directly on ``owner_class``.

    This previously fell through to ``None`` because the property lookup checked
    ``owner_class.__dict__`` directly, which inherited (rather than redeclared)
    descriptors never populate.
    """
    assert get_field_type_endpoint(_Named, "label") == str
    assert get_field_type_endpoint(_NamedTool, "label") == str
