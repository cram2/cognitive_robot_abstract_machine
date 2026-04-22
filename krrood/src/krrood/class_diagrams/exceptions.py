from dataclasses import dataclass
from typing_extensions import Type, Optional

from krrood.utils import DataclassException


@dataclass
class ClassIsUnMappedInClassDiagram(DataclassException):
    """
    Raised when a class is not mapped in the class diagram.
    """

    class_: Type
    """
    The class that is not mapped in the class diagram.
    """

    def __post_init__(self):
        self.message = f"Class {self.class_} is not mapped in the class diagram"
        super().__post_init__()


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

    def __post_init__(self):
        self.message = (
            f"Container type {self.container_type} is missing its contained type"
            f" for field '{self.field_name}' of class {self.class_}, please specify it."
        )
        super().__post_init__()


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

    def __post_init__(self):
        self.message = (
            f"Could not resolve type {self.type_name}.\n{self.extra_information}"
        )
        super().__post_init__()
