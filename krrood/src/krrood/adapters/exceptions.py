from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, Type

from krrood.adapters.json_field import JSONField
from krrood.exceptions import DataclassException


@dataclass
class JSONSerializationError(DataclassException):
    """
    Base exception for JSON (de)serialization errors.
    """


@dataclass
class MissingTypeError(JSONSerializationError):
    """
    Raised when :attr:`JSONField.TYPE` is missing in the JSON data.
    """

    def error_message(self) -> str:
        return f"Missing {JSONField.TYPE} field in JSON data"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidTypeFormatError(JSONSerializationError):
    """
    Raised when the value of :attr:`JSONField.TYPE` is not a fully qualified class name.
    """

    invalid_type_value: str

    def error_message(self) -> str:
        return f"Invalid type format: {self.invalid_type_value}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnknownModuleError(JSONSerializationError):
    """
    Raised when the module named by :attr:`JSONField.TYPE` cannot be imported.
    """

    module_name: str

    def error_message(self) -> str:
        return f"Unknown module in type: {self.module_name}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
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


@dataclass
class ClassNotSerializableError(JSONSerializationError):
    """
    Raised when the class specified cannot be JSON-serialized.
    """

    clazz: Type

    def error_message(self) -> str:
        return f"Class '{self.clazz.__name__}' cannot be serialized"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ClassNotDeserializableError(JSONSerializationError):
    """
    Raised when the class specified cannot be JSON-deserialized.
    """

    clazz: Type

    def error_message(self) -> str:
        return f"Class '{self.clazz.__name__}' cannot be deserialized"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UntrackedObjectError(JSONSerializationError):
    """
    Raised when a JSON document refers to an object by a key that no object was
    deserialized with.
    """

    key: Any
    """
    The key the document refers to the object with.
    """

    def error_message(self) -> str:
        return f"No object was deserialized with the key '{self.key}'."

    def suggest_correction(self) -> str:
        return (
            "Deserialize the referenced object before the objects that refer to it, "
            "within the same JSON document."
        )
