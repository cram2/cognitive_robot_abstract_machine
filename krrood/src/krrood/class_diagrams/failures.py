from dataclasses import dataclass
from typing_extensions import Type


@dataclass(init=False)
class ClassIsUnMappedInClassDiagram(Exception):
    """
    Raised when a class is not mapped in the class diagram.
    """

    def __init__(self, class_: Type):
        super().__init__(f"Class {class_} is not mapped in the class diagram")


@dataclass
class MissingContainedTypeOfContainer(Exception):
    """
    Raised when a container type is missing its contained type.
    For example, List without a specified type.
    """

    class_: Type
    field_name: str
    container_type: Type

    def __post_init__(self):
        super().__init__(
            f"Container type {self.container_type} is missing its contained type"
            f" for field '{self.field_name}' of class {self.class_}, please specify it."
        )
