"""Import-line generation and inspection of callables for code generation."""

from __future__ import annotations

import inspect
import sys
from collections import defaultdict
from dataclasses import dataclass
from importlib.util import resolve_name
from typing import Callable, Dict, Iterable, List, Optional, Set, Type

from krrood.code_generation.enums import PythonBuiltinParameterNames
from krrood.code_generation.exceptions import FunctionMissingAnnotationsError
from krrood.utils import (
    get_function_import_data,
    get_relative_import,
    is_typing_type,
)

# %%
# Callable inspection


@dataclass
class ImportData:
    """Import data required to reference a symbol from generated source."""

    import_line: str
    """The ``from … import …`` statement that makes the symbol available."""

    access_expression: str
    """The name expression used to reference the symbol after the import."""


def generate_import_statement_for_callable(function: Callable) -> ImportData:
    """Return the :class:`ImportData` for *function*.

    :param function: The callable to generate an import for.
    :returns: An :class:`ImportData` bundling the ``from … import …`` line and
        the name expression used to reference the callable after that import.

    Module-level function ``distance`` in ``my.module``::

        ImportData("from my.module import distance", "distance")

    Method ``MyClass.distance`` in ``my.module``::

        ImportData("from my.module import MyClass", "MyClass.distance")
    """
    module_name = function.__module__
    qualified_name = function.__qualname__
    qualified_name_parts = qualified_name.split(".")

    parent_segment = (
        qualified_name_parts[-2] if len(qualified_name_parts) >= 2 else None
    )
    is_method = (
        parent_segment is not None
        and parent_segment.isidentifier()
        and "<" not in parent_segment
    )

    if is_method:
        class_name = parent_segment
        import_line = f"from {module_name} import {class_name}"
        access_expression = f"{class_name}.{function.__name__}"
    else:
        import_line = f"from {module_name} import {function.__name__}"
        access_expression = function.__name__

    return ImportData(import_line=import_line, access_expression=access_expression)


def validate_annotations(function: Callable) -> None:
    """Raise :exc:`FunctionMissingAnnotationsError` if any required annotation is absent.

    Unannotated ``self`` and ``cls`` parameters are silently excluded.
    """
    signature = inspect.signature(function)
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name in PythonBuiltinParameterNames:
            continue
        if parameter.annotation is inspect.Parameter.empty:
            raise FunctionMissingAnnotationsError(
                function_qualified_name=function.__qualname__,
                missing_parameter_name=parameter_name,
            )
    if signature.return_annotation is inspect.Parameter.empty:
        raise FunctionMissingAnnotationsError(
            function_qualified_name=function.__qualname__,
            missing_parameter_name=None,
        )


# %%
# Import extraction / generation


def get_type_names_per_module_from_types(
    type_objects: Iterable[Type],
    excluded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Get a dictionary of type names grouped by module.

    :param type_objects: A list of type objects to format.
    :param excluded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A dictionary of type names grouped by module.
    """
    excluded_modules = [] if excluded_modules is None else excluded_modules
    excluded_names = [] if excluded_names is None else excluded_names
    module_to_types: Dict[str, List[str]] = defaultdict(list)
    for type_object in type_objects:
        try:
            if isinstance(type_object, type) or is_typing_type(type_object):
                module_name = type_object.__module__
                name = type_object.__qualname__
            elif callable(type_object):
                module_name, name = get_function_import_data(type_object)
            elif hasattr(type(type_object), "__module__"):
                module_name = type(type_object).__module__
                name = type(type_object).__qualname__
            else:
                continue
            if name == "NoneType":
                module_name = "types"
            if (
                module_name is None
                or module_name == "builtins"
                or module_name.startswith("_")
                or module_name in sys.builtin_module_names
                or module_name in excluded_modules
                or "<" in module_name
                or name in excluded_names
                or "site-packages" in module_name.split(".")
            ):
                continue
            if module_name == "typing":
                module_name = "typing_extensions"
            module_to_types[module_name].append(name)
        except AttributeError:
            continue
    return module_to_types


def get_imports_from_types(
    type_objects: Iterable[Type],
    target_file_path: Optional[str] = None,
    package_name: Optional[str] = None,
    excluded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> List[str]:
    """Format import lines from type objects.

    :param type_objects: A list of type objects to format.
    :param target_file_path: The file path to which the imports should be relative.
    :param package_name: The name of the package to use for relative imports.
    :param excluded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A list of formatted import lines.
    """
    module_to_types = get_type_names_per_module_from_types(
        type_objects, excluded_names, excluded_modules
    )

    lines: List[str] = []
    stem_imports: List[str] = []
    for module, names in module_to_types.items():
        filtered_names: Set[str] = set()
        for name in set(names):
            if "." in name:
                stem = ".".join(name.split(".")[1:])
                name_to_import = name.split(".")[0]
                filtered_names.add(name_to_import)
                stem_imports.append(f"{stem} = {name_to_import}.{stem}")
            else:
                filtered_names.add(name)
        joined = ", ".join(sorted(set(filtered_names)))
        import_path = module
        if (
            (target_file_path is not None)
            and (package_name is not None)
            and (package_name in module)
        ):
            import_path = get_relative_import(
                target_file_path, module_name=module, package_name=package_name
            )
        lines.append(f"from {import_path} import {joined}")
    lines.extend(stem_imports)
    return lines
