import enum
from dataclasses import dataclass
from typing import assert_never, Self

from random_events.set import Set

from typing_extensions import List, Any, Union

from random_events.variable import Variable, Symbolic, Continuous, Integer

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Index,
    Attribute,
    MappedVariable,
)

AttributeAccessLike = Union[Index, Attribute]


@dataclass
class ObjectAccessVariable:
    """
    Class to represent a variable that accesses an object field.
    """

    variable: Variable
    """
    The random events variable used to represent the object field.
    """

    attribute: AttributeAccessLike
    """
    The list of access paths used to access the object field.
    """

    @property
    def access_path(self) -> List[AttributeAccessLike]:
        """
        :return: The access path of the variable as a list of operations.
        """
        current = self.attribute
        result = [current]
        while isinstance(current, MappedVariable):
            current = current._child_
            result.append(current)
        return result[:-1][::-1]

    def set_value(self, obj: Any, value: Any):
        """
        Set the field of the object at the access path to the given value.

        :param obj: The object to be updated.
        :param value: The value to set.
        """

        current = obj
        for domain_mapping in self.access_path[:-1]:
            current = next(domain_mapping._apply_mapping_(current))

        if isinstance(self.attribute, Index):
            current[self.attribute._key_] = value
        elif isinstance(self.attribute, Attribute):
            setattr(current, self.attribute._attribute_name_, value)

    def __hash__(self):
        return hash(self.variable)

    def __eq__(self, other):
        return self.variable == other.variable

    @classmethod
    def from_attribute_access_and_type(
        cls, attribute_access: AttributeAccessLike, type_: type
    ) -> Self:
        """
        Create an ObjectAccessVariable from the given attribute and type.

        :param attribute_access: A symbolic expression representing access to the attribute.
        :param type_: The type to construct the variable for
        :return: The ObjectAccessVariable
        """
        if issubclass(type_, enum.Enum):
            result = Symbolic(str(attribute_access), Set.from_iterable(type_))
        elif issubclass(type_, bool):
            result = Symbolic(str(attribute_access), Set.from_iterable([True, False]))
        elif issubclass(type_, int):
            result = Integer(str(attribute_access))
        elif issubclass(type_, float):
            result = Continuous(str(attribute_access))
        else:
            assert_never((attribute_access, type_))

        return cls(result, attribute_access)

    @classmethod
    def from_expression(
        cls, attribute_access: AttributeAccessLike, expression: SymbolicExpression
    ) -> Self:
        """
        Create an ObjectAccessVariable from the given attribute and a symbolic expression.
        The symbolic expression is converted to a list and used for the domain of the variable.

        :param attribute_access: A symbolic expression representing access to the attribute.
        :param expression: The symbolic expression to use for the domain
        :return: The ObjectAccessVariable
        """
        domain = expression.tolist()
        variable = Symbolic(attribute_access._name_, Set.from_iterable(domain))
        return cls(variable, attribute_access)
