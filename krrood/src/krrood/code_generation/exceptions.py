"""
Exceptions raised by the code-generation package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, Callable

from krrood.exceptions import DataclassException

# %%
# Exceptions


@dataclass(kw_only=True)
class FunctionMissingAnnotationsError(DataclassException):
    """
    Raised when a callable lacks a type annotation required for generation.
    """

    function: Callable
    """
    The callable that is missing an annotation.
    """

    missing_parameter_name: str | None = None
    """
    The parameter whose annotation is missing, or ``None`` when the return annotation is
    the one missing.
    """

    def error_message(self) -> str:
        if self.missing_parameter_name is None:
            return (
                f"Function '{self.function.__qualname__}' lacks a return "
                f"type annotation."
            )
        return (
            f"Parameter '{self.missing_parameter_name}' of "
            f"'{self.function.__qualname__}' lacks a type annotation."
        )

    def suggest_correction(self) -> str:
        return "Add a type annotation to the reported callable."


@dataclass
class ObjectNotSerializableError(DataclassException):
    """
    Raised when an object, or one of its field values, has no constructor-source form.
    """

    value: Any
    """
    The value that could not be serialized.
    """

    supported_types: tuple[type, ...]
    """
    The scalar types the serializer supports, besides ``None`` and nested dataclasses.
    """

    def error_message(self) -> str:
        type_names = ", ".join(
            supported_type.__name__ for supported_type in self.supported_types
        )
        return (
            f"Cannot serialize value of type {type(self.value).__name__!r} to Python "
            f"constructor source. Only None, {type_names} members, and nested "
            "dataclasses are supported."
        )

    def suggest_correction(self) -> str:
        return "For other types, implement a custom ObjectSourceSerializer."
