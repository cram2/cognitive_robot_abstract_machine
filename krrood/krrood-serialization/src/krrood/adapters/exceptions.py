from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing_extensions import Any, TYPE_CHECKING

from krrood.adapters.json_field import JSONField
from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    import numpy as np


@dataclass(kw_only=True)
class JSONSerializationError(DataclassException):
    """
    Base exception for JSON (de)serialization errors.
    """


@dataclass(kw_only=True)
class MissingTypeError(JSONSerializationError):
    """
    Raised when :attr:`JSONField.TYPE` is missing in the JSON data.
    """

    def error_message(self) -> str:
        return f"Missing {JSONField.TYPE} field in JSON data"

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class InvalidTypeFormatError(JSONSerializationError):
    """
    Raised when the value of :attr:`JSONField.TYPE` is not a fully qualified class name.
    """

    invalid_type_value: str

    def error_message(self) -> str:
        return f"Invalid type format: {self.invalid_type_value}"

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class UnknownModuleError(JSONSerializationError):
    """
    Raised when the module named by :attr:`JSONField.TYPE` cannot be imported.
    """

    module_name: str

    def error_message(self) -> str:
        return f"Unknown module in type: {self.module_name}"

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class ClassNotFoundError(JSONSerializationError):
    """
    Raised when the class named by :attr:`JSONField.TYPE` cannot be found in the module.
    """

    class_name: str
    module_name: str

    def error_message(self) -> str:
        return f"Class '{self.class_name}' not found in module '{self.module_name}'"

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class ClassNotSerializableError(JSONSerializationError):
    """
    Raised when the class specified cannot be JSON-serialized.
    """

    clazz: type

    def error_message(self) -> str:
        return f"Class '{self.clazz.__name__}' cannot be serialized"

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class ClassNotDeserializableError(JSONSerializationError):
    """
    Raised when the class specified cannot be JSON-deserialized.
    """

    clazz: type

    def error_message(self) -> str:
        return f"Class '{self.clazz.__name__}' cannot be deserialized"

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class UntrackedObjectError(JSONSerializationError):
    """
    Raised when a JSON document refers to an object by a key that no object was
    deserialized with.
    """

    key: Any
    """
    The key the document refers to the object with.
    """

    name: Any = None
    """
    The name the reference to the object went by, which says which object was meant.
    """

    def error_message(self) -> str:
        return (
            f"No object was deserialized with the key '{self.key}' "
            f"(referred to as '{self.name}')."
        )

    def suggest_correction(self) -> str:
        return (
            "Deserialize the referenced object before the objects that refer to it, "
            "within the same JSON document."
        )


@dataclass(kw_only=True)
class ArrayElementTypeSerializationError(JSONSerializationError, ABC):
    """
    Raised when a numpy array is serialized whose element type the serializer cannot
    write down faithfully.
    """

    element_type: np.dtype
    """
    The element type of the array.
    """

    def error_message(self) -> str:
        return f"Serializing a numpy array of element type {self.element_type} is not supported."


@dataclass(kw_only=True)
class ObjectArraySerializationError(ArrayElementTypeSerializationError):
    """
    Raised when a numpy array of Python objects is serialized, whose bytes are
    references rather than values.
    """

    def suggest_correction(self) -> str:
        return "Consider converting the array to a list first and serializing that."


@dataclass(kw_only=True)
class StructuredArraySerializationError(ArrayElementTypeSerializationError):
    """
    Raised when a numpy array of records is serialized, whose field names its element
    type's byte layout does not carry.
    """

    def suggest_correction(self) -> str:
        return (
            "Consider serializing each field of the array as an array of its own, or, if appropriate, see if"
            "you could contribute to krrood by adding support for this kind of array"
        )
