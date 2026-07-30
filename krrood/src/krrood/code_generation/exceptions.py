"""
Exceptions raised by the code-generation package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Optional, Callable

from krrood.exceptions import DataclassException

# %%
# Exceptions


@dataclass
class FunctionMissingAnnotationsError(DataclassException):
    """
    Raised when a callable lacks a type annotation required for generation.
    """

    function: Callable
    """
    The callable that is missing an annotation.
    """

    missing_parameter_name: Optional[str] = None
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
