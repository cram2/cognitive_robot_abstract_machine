from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.predicate import Symbol
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


@dataclass
class UnMonitoredContainerTypeForDescriptor(Exception):
    """
    Raised when a descriptor is used on a field with a container type that is not
    monitored (i.e., is not a subclass of MonitoredContainer).

    This happens when your type hint of the field is using a container type that is not
    supported.
    """

    clazz: type
    field_name: str
    container_type: type

    def __post_init__(self):
        super().__init__(
            f"Unmonitored container type '{self.container_type.__name__}' used for field '{self.field_name}' "
            f"in class '{self.clazz.__name__}'."
        )


@dataclass
class NoObjectOfIndividualDeclaresProperty(DataclassException):
    """
    Raised when a fact is stored for an individual none of whose objects, its root role
    taker and its roles, has the field of the property.
    """

    individual_object: Symbol
    """
    An object of the individual.
    """

    property_descriptor_class: type[PropertyDescriptor]
    """
    The property of the fact.
    """

    def error_message(self) -> str:
        return (
            f"No object of the individual of {self.individual_object!r} declares the "
            f"property {self.property_descriptor_class.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Give the individual a role, or a class, that declares a field managed by "
            f"{self.property_descriptor_class.__name__}."
        )
