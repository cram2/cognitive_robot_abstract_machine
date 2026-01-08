from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Type

from ..utils import DataclassException

JSON_TYPE_NAME = "__json_type__"  # the key used in JSON dicts to identify the class


@dataclass
class JSONSerializationError(DataclassException):
    """Base exception for JSON (de)serialization errors."""


@dataclass
class MissingTypeError(JSONSerializationError):
    """Raised when the 'type' field is missing in the JSON data."""

    def __post_init__(self):
        self.message = f"Missing {JSON_TYPE_NAME} field in JSON data"


@dataclass
class InvalidTypeFormatError(JSONSerializationError):
    """Raised when the 'type' field value is not a fully qualified class name."""

    invalid_type_value: str

    def __post_init__(self):
        self.message = f"Invalid type format: {self.invalid_type_value}"


@dataclass
class UnknownModuleError(JSONSerializationError):
    """Raised when the module specified in the 'type' field cannot be imported."""

    module_name: str

    def __post_init__(self):
        self.message = f"Unknown module in type: {self.module_name}"


@dataclass
class ClassNotFoundError(JSONSerializationError):
    """Raised when the class specified in the 'type' field cannot be found in the module."""

    class_name: str
    module_name: str

    def __post_init__(self):
        self.message = (
            f"Class '{self.class_name}' not found in module '{self.module_name}'"
        )


@dataclass
class ClassNotSerializableError(JSONSerializationError):
    """Raised when the class specified cannot be JSON-serialized."""

    clazz: Type

    def __post_init__(self):
        self.message = f"Class '{self.clazz.__name__}' cannot be serialized"


@dataclass
class ClassNotDeserializableError(JSONSerializationError):
    """Raised when the class specified cannot be JSON-deserialized."""

    clazz: Type

    def __post_init__(self):
        self.message = f"Class '{self.clazz.__name__}' cannot be deserialized"
