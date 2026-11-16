"""
The errors raised while reading an ontology and while keeping its facts on Python
objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import rdflib
from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.predicate import Symbol
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class AmbiguousDescribedClass(DataclassException):
    """
    Raised when an anonymous class expression is declared under, or over, more than one
    named class, so it describes no single class.
    """

    class_expression_name: str
    """
    The name given to the anonymous class expression.
    """

    candidates: list[str]
    """
    The names of the named classes the expression could describe.
    """

    def error_message(self) -> str:
        return (
            f"The class expression {self.class_expression_name} describes more than "
            f"one named class: {', '.join(self.candidates)}."
        )

    def suggest_correction(self) -> str:
        return (
            "Declare the class expression under, or over, a single named class, or "
            "state it once for each class it describes."
        )


@dataclass(kw_only=True)
class UnsupportedClassExpression(DataclassException):
    """
    Raised when a blank node is named as an anonymous class expression but is none of
    the kinds of class expression known to the converter.
    """

    node: rdflib.BNode
    """
    The blank node.
    """

    predicates: list[rdflib.URIRef]
    """
    The predicates stated on the blank node.
    """

    def error_message(self) -> str:
        return (
            f"The blank node {self.node} is no known kind of anonymous class "
            f"expression; it states {', '.join(map(str, self.predicates))}."
        )

    def suggest_correction(self) -> str:
        return (
            "Name only blank nodes that are anonymous class expressions, or add a "
            "subclass of ClassExpression for this kind."
        )
