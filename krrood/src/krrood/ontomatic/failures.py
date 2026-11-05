from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


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
class DescriptorNotDeclaredForDomain(DataclassException):
    """
    Raised when neither a class nor any class it inherits fields from declares a field
    managed by a descriptor type.
    """

    descriptor_type: type
    """
    The descriptor type.
    """

    domain_type: type
    """
    The class without such a field.
    """

    def error_message(self) -> str:
        return (
            f"No field of {self.domain_type.__name__} is managed by "
            f"{self.descriptor_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Declare the field on {self.domain_type.__name__} or on a class it inherits from, "
            f"and assign a {self.descriptor_type.__name__} to it."
        )


@dataclass
class NoObjectOfIndividualDeclaresProperty(DataclassException):
    """
    Raised when a fact is stored for an individual none of whose objects, its root role
    taker and its roles, has the field of the property.
    """

    individual_object: object
    """
    An object of the individual.
    """

    property_descriptor_class: type
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
