"""Utilities for extracting source code from Python modules, files, and ASTs.

These functions extract function/class source code and parse import statements.
They are distinct from the code-generation utilities in
:mod:`krrood.code_generation`.
"""

from __future__ import annotations

import ast
import inspect
import logging
import os
import textwrap
import types
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.util import resolve_name
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from krrood.exceptions import (
    ModuleNotFoundForConvertingImportsToAbsolute,
    NoSourceDataToParseImportsFrom,
    SourceDataNotProvided,
)

logger = logging.getLogger(__name__)

# %%
# Extraction result types


@dataclass
class LineSpan:
    """The inclusive start and end line numbers of a definition in its source."""

    start_line: int
    """The one-based line number where the definition begins."""

    end_line: int
    """The one-based line number where the definition ends."""


@dataclass
class ExtractedDefinition:
    """A single function or class definition extracted from source code."""

    name: str
    """The name of the extracted definition."""

    source: Union[str, List[str]]
    """The definition source, joined into a single string or kept as a list of
    lines depending on the extraction request."""

    line_span: LineSpan
    """The start/end line numbers the definition occupies in its source."""


@dataclass
class ExtractedSource:
    """The definitions extracted from a piece of Python source code."""

    definitions: List[ExtractedDefinition] = field(default_factory=list)
    """The extracted definitions, in the order they appear in the source."""

    def source_of(self, name: str) -> Union[str, List[str]]:
        """Return the extracted source of the definition named *name*.

        :param name: The definition name to look up.
        :raises KeyError: If no extracted definition carries that name.
        """
        for definition in self.definitions:
            if definition.name == name:
                return definition.source
        raise KeyError(name)


# %%
# Import extraction


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
    tree, current_module_name = _resolve_import_tree(
        module, file_path, source, ast_tree, convert_relative_to_absolute
    )
    import_modules, from_imports = _collect_imports_from_tree(
        tree, exclude_libraries, convert_relative_to_absolute, current_module_name
    )
    return _format_import_lines(import_modules, from_imports)


def _resolve_import_tree(
    module: Optional[types.ModuleType],
    file_path: Optional[str],
    source: Optional[str],
    ast_tree: Optional[ast.AST],
    convert_relative_to_absolute: bool,
) -> Tuple[ast.AST, Optional[str]]:
    """Resolve the AST to walk and the enclosing module name for relative-import conversion.

    :param module: The module to extract imports from.
    :param file_path: The file path to extract imports from.
    :param source: The source code to extract imports from.
    :param ast_tree: The ast tree to extract imports from.
    :param convert_relative_to_absolute: Whether relative imports will be converted
        to absolute, which requires a module or file to anchor them to.
    :raises NoSourceDataToParseImportsFrom: If no module, file, source, or AST is given.
    :raises ModuleNotFoundForConvertingImportsToAbsolute: If asked to convert relative
        imports to absolute without a module or file to anchor them to.
    :returns: The AST to walk, and the enclosing module name (if resolvable).
    """
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
    return ast_tree or ast.parse(source), current_module_name


def _collect_imports_from_tree(
    tree: ast.AST,
    exclude_libraries: List[str],
    convert_relative_to_absolute: bool,
    current_module_name: Optional[str],
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Walk *tree* and collect plain (``import x``) and ``from``-imports separately.

    :param tree: The AST to walk.
    :param exclude_libraries: Module names to exclude from the result.
    :param convert_relative_to_absolute: Whether to resolve relative ``from``-imports
        to their absolute module name.
    :param current_module_name: The enclosing module name relative imports resolve against.
    :returns: The plain-imported module names, and a mapping of ``from``-module name
        to the names imported from it.
    """
    import_modules: Set[str] = set()
    from_imports: Dict[str, Set[str]] = defaultdict(set)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _collect_plain_import(node, exclude_libraries, import_modules)
        elif isinstance(node, ast.ImportFrom):
            _collect_from_import(
                node,
                exclude_libraries,
                convert_relative_to_absolute,
                current_module_name,
                from_imports,
            )
    return import_modules, from_imports


def _collect_plain_import(
    node: ast.Import, exclude_libraries: List[str], import_modules: Set[str]
) -> None:
    """Add the module names of an ``import x`` statement to *import_modules*.

    :param node: The ``import`` AST node.
    :param exclude_libraries: Module names to skip.
    :param import_modules: The set of collected plain-import module names, updated in place.
    """
    for alias in node.names:
        if alias.name in exclude_libraries:
            continue
        if alias.asname:
            import_modules.add(f"{alias.name} as {alias.asname}")
        else:
            import_modules.add(alias.name)


def _collect_from_import(
    node: ast.ImportFrom,
    exclude_libraries: List[str],
    convert_relative_to_absolute: bool,
    current_module_name: Optional[str],
    from_imports: Dict[str, Set[str]],
) -> None:
    """Add the names of a ``from x import y`` statement to *from_imports*, keyed by module.

    :param node: The ``from ... import`` AST node.
    :param exclude_libraries: Module names to skip.
    :param convert_relative_to_absolute: Whether to resolve a relative *node.module*
        to its absolute module name.
    :param current_module_name: The enclosing module name relative imports resolve against.
    :param from_imports: The collected ``from``-module to imported-names mapping, updated in place.
    """
    if node.module and node.module in exclude_libraries:
        return
    prefix = "." * node.level
    full_module = f"{prefix}{node.module or ''}"
    if convert_relative_to_absolute and node.level > 0:
        full_module = resolve_name(full_module, current_module_name)
    for alias in node.names:
        if alias.asname:
            from_imports[full_module].add(f"{alias.name} as {alias.asname}")
        else:
            from_imports[full_module].add(alias.name)


def _format_import_lines(
    import_modules: Set[str], from_imports: Dict[str, Set[str]]
) -> List[str]:
    """Render collected import data as sorted ``import``/``from ... import`` line strings.

    :param import_modules: The plain-imported module names.
    :param from_imports: The ``from``-module name to imported-names mapping.
    :returns: A sorted list of import-line strings.
    """
    result: Set[str] = set()
    for module_name in sorted(import_modules):
        result.add(f"import {module_name}")
    for module_name, names in sorted(from_imports.items()):
        joined = ", ".join(sorted(names))
        result.add(f"from {module_name} import {joined}")
    return sorted(result)


# %%
# Function / class source extraction


def extract_function_source(
    names: List[str],
    file_path: Optional[str] = None,
    source: Optional[str] = None,
    join_lines: bool = True,
    include_signature: bool = True,
) -> ExtractedSource:
    """Extract the source code of functions from a file and/or source string.

    :param names: The names of the functions to extract; empty extracts all.
    :param file_path: The path to the file to read the source from.
    :param source: The source code to extract from, when no file is given.
    :param join_lines: Whether to join the lines of each definition.
    :param include_signature: Whether to include the function signature.
    :returns: The extracted function definitions.
    """
    return _extract_definitions(
        ast.FunctionDef,
        names,
        file_path=file_path,
        source=source,
        join_lines=join_lines,
        include_signature=include_signature,
    )


def extract_class_source(
    names: List[str],
    file_path: Optional[str] = None,
    source: Optional[str] = None,
    join_lines: bool = True,
    include_signature: bool = True,
) -> ExtractedSource:
    """Extract the source code of classes from a file and/or source string.

    :param names: The names of the classes to extract; empty extracts all.
    :param file_path: The path to the file to read the source from.
    :param source: The source code to extract from, when no file is given.
    :param join_lines: Whether to join the lines of each definition.
    :param include_signature: Whether to include the class signature.
    :returns: The extracted class definitions.
    """
    return _extract_definitions(
        ast.ClassDef,
        names,
        file_path=file_path,
        source=source,
        join_lines=join_lines,
        include_signature=include_signature,
    )


def _extract_definitions(
    definition_type: Type[ast.AST],
    names: List[str],
    file_path: Optional[str],
    source: Optional[str],
    join_lines: bool,
    include_signature: bool,
) -> ExtractedSource:
    """Walk the top-level AST and collect definitions of *definition_type*.

    :param definition_type: The AST node type to collect (function or class).
    :param names: The names to collect; empty collects every matching node.
    :param file_path: The file to read the source from, when *source* is absent.
    :param source: The source code to walk.
    :param join_lines: Whether to join the lines of each definition.
    :param include_signature: Whether to include the definition signature.
    :raises SourceDataNotProvided: If neither a file path nor source is given.
    :returns: The extracted definitions in source order.
    """
    if file_path is None and source is None:
        raise SourceDataNotProvided(file_path=file_path, source_code=source)
    if source is None:
        with open(file_path, "r") as source_file:
            source = source_file.read()

    lines = source.splitlines()
    tree = ast.parse(source)
    extracted = ExtractedSource()

    for node in tree.body:
        if not isinstance(node, definition_type):
            continue
        if names and node.name not in names:
            continue
        definition_lines = lines[node.lineno - 1 : node.end_lineno]
        if not include_signature:
            definition_lines = definition_lines[1:]
        definition_source = (
            textwrap.dedent("\n".join(definition_lines))
            if join_lines
            else definition_lines
        )
        extracted.definitions.append(
            ExtractedDefinition(
                name=node.name,
                source=definition_source,
                line_span=LineSpan(node.lineno, node.end_lineno),
            )
        )
        if names and len(extracted.definitions) >= len(names):
            break

    if names and len(extracted.definitions) < len(names):
        found_names = {definition.name for definition in extracted.definitions}
        logger.warning(
            "Could not find all definitions: %s not found, definitions not found: %s",
            names,
            set(names) - found_names,
        )
    return extracted
