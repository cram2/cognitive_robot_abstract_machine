"""Core type definitions for Robokudo.

This module provides the base type hierarchy for Robokudo's type system.
It is comparable to uima.cas.TOP in uimacpp and provides the foundation
for all other types in the system.

The module defines:

* Base Type class
* Common type mixins (Identifiable, Nameable)
* Query and Annotation types
"""

from dataclasses import dataclass


@dataclass
class Type:
    """Base class for all Robokudo types.

    This is the root of the type hierarchy, comparable to uima.cas.TOP in uimacpp.
    All other types in the system should inherit from this class.
    """

    ...


@dataclass
class Identifiable(Type):
    """Mixin class for types that need unique identification.

    Adds an ID field to types that need to be uniquely identified.
    """

    id: str = str("")
    """
    Unique identifier string
    """


@dataclass
class Nameable(Type):
    """Mixin class for types that need a name.

    Adds a name field to types that need to be named.
    """

    name: str = str("")
    """
    Name of the instance
    """


@dataclass
class Query(Type):
    """Type representing a query in the system.

    Encapsulates query information for processing.
    """

    query: str = str("")
    """
    Query string to be processed
    """


@dataclass
class Annotation(Type):
    """Base class for all annotations.

    Represents metadata or annotations added to data during processing.
    """

    source: str = ""
    """
    Name of the annotator that created the annotation
    """


@dataclass
class IdentifiableAnnotation(Annotation, Identifiable):
    """Annotation type with unique identification.

    Combines annotation capabilities with unique identification.
    Inherits source from Annotation and id from Identifiable.
    """

    ...
