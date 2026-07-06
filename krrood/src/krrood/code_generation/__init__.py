"""Code generation utilities for the CRAM krrood package.

This package provides general infrastructure for generating Python source code:
naming conventions, import extraction, type-hint serialisation, source-code
formatting, and a :class:`CodeGenerator` base class for Jinja2-based generation.

Domain-specific Jinja2 templates live alongside their respective packages
(e.g. ``entity_query_language/rdr/templates/``).
"""

from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.source_extraction_utils import (
    extract_function_or_class_file,
    extract_function_or_class_from_source,
    extract_imports_from,
)
from krrood.code_generation.utils import (
    # naming
    str_to_snake_case,
    to_camel_case,
    to_variable_name,
    # imports
    generate_callable_import,
    generate_relative_import,
    get_imports_from_types,
    get_type_names_per_module_from_types,
    get_types_to_import_from_func_type_hints,
    get_types_to_import_from_type_hints,
    # type-hint serialisation
    stringify_hint,
    stringify_type_hint,
    value_to_source,
    # FunctionCase generation
    function_to_dataclass_source,
    validate_annotations,
    FunctionMissingAnnotationsError,
    # formatting
    run_black_on_file,
    run_ruff_on_file,
)

__all__ = [
    # naming
    "str_to_snake_case",
    "to_camel_case",
    "to_variable_name",
    # imports
    "extract_imports_from",
    "generate_callable_import",
    "generate_relative_import",
    "get_imports_from_types",
    "get_type_names_per_module_from_types",
    "get_types_to_import_from_func_type_hints",
    "get_types_to_import_from_type_hints",
    # type-hint serialisation
    "stringify_hint",
    "stringify_type_hint",
    "value_to_source",
    # FunctionCase generation
    "function_to_dataclass_source",
    "validate_annotations",
    "FunctionMissingAnnotationsError",
    # source extraction
    "extract_function_or_class_file",
    "extract_function_or_class_from_source",
    # formatting
    "run_black_on_file",
    "run_ruff_on_file",
    # generator
    "CodeGenerator",
]
