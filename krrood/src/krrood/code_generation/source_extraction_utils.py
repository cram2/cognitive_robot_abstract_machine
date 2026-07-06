"""
Utilities for extracting source code from Python modules, files, and ASTs.

These are infrastructure functions used by multiple krrood subsystems
(RDR, EQL-RDR, class_diagrams) to extract function/class source code and
parse import statements.  They are distinct from the code-generation
utilities in :mod:`krrood.code_generation.utils`.
"""

from __future__ import annotations

import ast
import inspect
import logging
import os
import textwrap
import types
from collections import defaultdict
from importlib.util import resolve_name
from typing import Dict, List, Optional, Set, Tuple, Union

from krrood.exceptions import (
    ModuleNotFoundForConvertingImportsToAbsolute,
    NoSourceDataToParseImportsFrom,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def extract_imports_from(
    module: Optional[types.ModuleType] = None,
    file_path: Optional[str] = None,
    source: Optional[str] = None,
    ast_tree: Optional[ast.AST] = None,
    exclude_libraries: Optional[List[str]] = None,
    convert_relative_to_absolute: bool = False,
) -> List[str]:
    """Extract import statements from a module, source, file path, or AST.

    :param module: The module to extract imports from.
    :param file_path: The file path to extract imports from.
    :param source: The source code to extract imports from.
    :param ast_tree: The ast tree to extract imports from.
    :param exclude_libraries: A list of libraries to exclude from the imports.
    :param convert_relative_to_absolute: Whether to convert relative imports
        to absolute.
    :returns: A sorted list of import-line strings.
    """
    exclude_libraries = exclude_libraries or []
    if module is None and source is None and file_path is None and ast_tree is None:
        raise NoSourceDataToParseImportsFrom(
            module=module, file_path=file_path, ast_tree=ast_tree
        )
    current_module_name = None
    if module:
        source = inspect.getsource(module)
        current_module_name = module.__name__
    elif file_path:
        with open(file_path, "r") as f:
            source = f.read()
        current_module_name = os.path.splitext(os.path.basename(file_path))[0]
    elif convert_relative_to_absolute:
        raise ModuleNotFoundForConvertingImportsToAbsolute(
            path=file_path, source_code=source
        )

    tree = ast_tree or ast.parse(source)

    import_modules: Set[str] = set()
    from_imports: Dict[str, Set[str]] = defaultdict(set)

    for node in ast.walk(tree):
        # import x
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in exclude_libraries:
                    continue
                if alias.asname:
                    import_modules.add(f"{name} as {alias.asname}")
                else:
                    import_modules.add(name)

        # from x import y
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module_name = node.module or ""
            full_module = f"{prefix}{module_name}"

            if convert_relative_to_absolute and node.level > 0:
                full_module = resolve_name(full_module, current_module_name)

            if node.module and node.module in exclude_libraries:
                continue

            for alias in node.names:
                if alias.asname:
                    from_imports[full_module].add(f"{alias.name} as {alias.asname}")
                else:
                    from_imports[full_module].add(alias.name)

    result: Set[str] = set()

    for mod in sorted(import_modules):
        result.add(f"import {mod}")

    for mod, names in sorted(from_imports.items()):
        joined = ", ".join(sorted(names))
        result.add(f"from {mod} import {joined}")

    return sorted(result)


# ---------------------------------------------------------------------------
# Function / class source extraction
# ---------------------------------------------------------------------------


def extract_function_or_class_file(
    file_path: str,
    function_names: List[str],
    join_lines: bool = True,
    return_line_numbers: bool = False,
    include_signature: bool = True,
    as_list: bool = False,
    is_class: bool = False,
) -> Union[
    Dict[str, Union[str, List[str]]],
    Tuple[Dict[str, Union[str, List[str]]], Dict[str, Tuple[int, int]]],
]:
    """Extract the source code of a function/class from a file.

    :param file_path: The path to the file.
    :param function_names: The names of the functions/classes to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers.
    :param include_signature: Whether to include the function/class signature.
    :param as_list: Whether to return a list of sources instead of a dict.
    :param is_class: Whether to also look for class definitions.
    :return: A dictionary mapping names to source code, or a tuple with
        line numbers.
    """
    with open(file_path, "r") as f:
        source = f.read()

    return extract_function_or_class_from_source(
        source,
        function_names,
        join_lines=join_lines,
        return_line_numbers=return_line_numbers,
        include_signature=include_signature,
        as_list=as_list,
        is_class=is_class,
    )


def extract_function_or_class_from_source(
    source: str,
    function_names: List[str],
    join_lines: bool = True,
    return_line_numbers: bool = False,
    include_signature: bool = True,
    as_list: bool = False,
    is_class: bool = False,
) -> Union[
    Dict[str, Union[str, List[str]]],
    Tuple[Dict[str, Union[str, List[str]]], Dict[str, Tuple[int, int]]],
]:
    """Extract the source code of a function/class from source text.

    :param source: The string containing the source code.
    :param function_names: The names of the functions/classes to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers.
    :param include_signature: Whether to include the function/class signature.
    :param as_list: Whether to return a list of sources instead of a dict.
    :param is_class: Whether to also look for class definitions.
    :return: A dictionary mapping names to source code, or a tuple with
        line numbers.
    """
    # Ensure function_names is a list (avoid circular import from rdr.utils).
    # Keep the same semantics as the original ``make_list``: strings are
    # treated as a single element, not iterated over character-by-character.
    if isinstance(function_names, str) or not isinstance(function_names, list):
        try:
            iter(function_names)
            function_names = (
                list(function_names)
                if not isinstance(function_names, str)
                else [function_names]
            )
        except TypeError:
            function_names = [function_names]

    tree = ast.parse(source)
    functions_source: Dict[str, Union[str, List[str]]] = {}
    functions_source_list: List[Union[str, List[str]]] = []
    line_numbers: Dict[str, Tuple[int, int]] = {}
    line_numbers_list: List[Tuple[int, int]] = []
    look_for_type = ast.ClassDef if is_class else ast.FunctionDef

    for node in tree.body:
        if isinstance(node, look_for_type) and (
            node.name in function_names or len(function_names) == 0
        ):
            lines = source.splitlines()
            func_lines = lines[node.lineno - 1 : node.end_lineno]
            if not include_signature:
                func_lines = func_lines[1:]
            if as_list:
                line_numbers_list.append((node.lineno, node.end_lineno))
            else:
                line_numbers[node.name] = (node.lineno, node.end_lineno)
            parsed_function = (
                textwrap.dedent("\n".join(func_lines)) if join_lines else func_lines
            )
            if as_list:
                functions_source_list.append(parsed_function)
            else:
                functions_source[node.name] = parsed_function
            if len(function_names) > 0:
                if len(functions_source) >= len(function_names) or len(
                    functions_source_list
                ) >= len(function_names):
                    break
    if len(functions_source) < len(function_names) and len(functions_source_list) < len(
        function_names
    ):
        logger.warning(
            f"Could not find all functions: {function_names} not found, "
            f"functions not found: "
            f"{set(function_names) - set(functions_source.keys())}"
        )
    if return_line_numbers:
        return functions_source if not as_list else functions_source_list, (
            line_numbers if not as_list else line_numbers_list
        )
    return functions_source if not as_list else functions_source_list
