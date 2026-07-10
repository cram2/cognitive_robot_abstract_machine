from dataclasses import dataclass

from typing_extensions import Type, Optional

from krrood.exceptions import DataclassException


@dataclass
class ClassIsUnMappedInClassDiagram(DataclassException):
    """
    Raised when a class is not mapped in the class diagram.
    """

    class_: Type
    """
    The class that is not mapped in the class diagram.
    """

    def error_message(self) -> str:
        return f"Class {self.class_} is not mapped in the class diagram"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MissingContainedTypeOfContainer(DataclassException):
    """
    Raised when a container type is missing its contained type.
    For example, List without a specified type.
    """

    class_: Type
    """
    The class that is missing the contained type.
    """
    field_name: str
    """
    The name of the field that is missing the contained type.
    """
    container_type: Type
    """
    The container type that is missing its contained type.
    """

    def error_message(self) -> str:
        return (
            f"Container type {self.container_type} is missing its contained type"
            f" for field '{self.field_name}' of class {self.class_}, please specify it."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class CouldNotResolveType(DataclassException):
    """
    Raised when a type cannot be resolved.
    """

    type_name: str
    """
    The name of the type that could not be resolved.
    """
    error: Optional[Exception] = None
    """
    The exception that was raised when resolving the type.
    """
    extra_information: str = ""
    """
    Additional information about the error.
    """

    def error_message(self) -> str:
        return (
            f"Could not resolve type {self.type_name}.\n{self.extra_information}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MockedClassInstantiationError(DataclassException):
    """
    Raised when an attempt is made to instantiate a MockedClass.
    """

    def error_message(self) -> str:
        return "MockedClass cannot be instantiated directly"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class FactoryMethodDecoratorError(DataclassException):
    """
    Raised when ``@factory_method`` does not wrap a ``classmethod``.

    The decorator records the factory through Python's ``__set_name__`` descriptor hook, which a
    ``classmethod`` does not forward to the object it wraps. The marker must therefore be the
    outermost decorator, applied above ``@classmethod``; any other placement would skip
    registration silently, so it is rejected here instead.
    """

    decorated_name: str
    """The name of the function the marker was applied to."""

    def error_message(self) -> str:
        return f"@factory_method was applied to {self.decorated_name!r}, which is not a classmethod"

    def suggest_correction(self) -> str:
        return "Apply @factory_method as the outermost decorator, directly above @classmethod."
