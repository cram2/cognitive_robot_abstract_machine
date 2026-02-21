from dataclasses import dataclass

from typing_extensions import List, Any, Union

from random_events.variable import Variable


from ..entity_query_language.core.mapped_variable import Index, Attribute

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
        while current._child_ is not None:
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
