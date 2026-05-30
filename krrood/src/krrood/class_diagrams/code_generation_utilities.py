"""
Deprecated — use :mod:`krrood.code_generation.utils` instead.

This module has been moved.  The re-exports below are kept for backward
compatibility and will be removed in a future release.
"""

from __future__ import annotations

from krrood.code_generation.utils import (
    FunctionMissingAnnotationsError,
    function_to_dataclass_source,
    generate_callable_import,
    to_camel_case,
    validate_annotations,
)

__all__ = [
    "FunctionMissingAnnotationsError",
    "function_to_dataclass_source",
    "generate_callable_import",
    "to_camel_case",
    "validate_annotations",
]
