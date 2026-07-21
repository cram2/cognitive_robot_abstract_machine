from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Type

from krrood.exceptions import DataclassException

JSON_TYPE_NAME = "__json_type__"  # the key used in JSON dicts to identify the class


@dataclass
class JSONSerializationError(DataclassException):
    """
    Base exception for JSON (de)serialization errors.
    """


@dataclass
class MissingTypeError(JSONSerializationError):
    """
    Raised when the 'type' field is missing in the JSON data.
    """

    def error_message(self) -> str:
        return f"Missing {JSON_TYPE_NAME} field in JSON data"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidTypeFormatError(JSONSerializationError):
    """
    Raised when the 'type' field value is not a fully qualified class name.
    """

    invalid_type_value: str

    def error_message(self) -> str:
        return f"Invalid type format: {self.invalid_type_value}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnknownModuleError(JSONSerializationError):
    """
    Raised when the module specified in the 'type' field cannot be imported.
    """

    module_name: str

    def error_message(self) -> str:
        return f"Unknown module in type: {self.module_name}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ClassNotFoundError(JSONSerializationError):
    """
    Raised when the class specified in the 'type' field cannot be found in the module.
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
