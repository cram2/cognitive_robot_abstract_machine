from __future__ import annotations

import ast
import inspect
import textwrap

from typing_extensions import Dict, Optional, Any

from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.patterns.caching import weak_key_cache


def first_docstring_line(documented_object: Any) -> Optional[str]:
    """:return: The first non-empty line of *documented_object*'s docstring, or ``None``."""
    if documented_object is None:
        return None
    docstring = inspect.getdoc(documented_object)
    if not docstring:
        return None
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _annotated_target_name(node: ast.AST) -> Optional[str]:
    """:return: The target name when *node* is an annotated assignment with a simple name target."""
    if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
        return None
    return node.target.id


def _string_expression_first_line(node: ast.AST) -> Optional[str]:
    """:return: The first stripped line when *node* is a bare string expression (a PEP 257
    attribute docstring), else ``None``."""
    if (
        not isinstance(node, ast.Expr)
        or not isinstance(node.value, ast.Constant)
        or not isinstance(node.value.value, str)
    ):
        return None
    for line in node.value.value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@weak_key_cache
def _attribute_docstrings(cls: type) -> Dict[str, str]:
    """
    :param cls: The class whose own body to scan.
    :return: A mapping of field name to its first PEP 257 attribute docstring line, or an empty
        mapping when the source is unavailable (e.g. C-extension classes).
    """
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return {}
    try:
        class_definition = ast.parse(source).body[0]
    except (SyntaxError, IndexError):
        return {}
    body = getattr(class_definition, "body", [])
    docstrings: Dict[str, str] = {}
    for current, following in zip(body, body[1:]):
        name = _annotated_target_name(current)
        if name is None:
            continue
        line = _string_expression_first_line(following)
        if line is None:
            continue
        docstrings[name] = line
    return docstrings


def docstring_for_source_ref(source_reference: SourceReference) -> Optional[str]:
    """
    Attribute documentation follows the project convention of a bare string expression
    immediately below the field definition (a PEP 257 attribute docstring).

    :param source_reference: The source reference to document.
    :return: The first docstring line for the class or field *source_reference* points at, or ``None``
        when no documentation is found.
    """
    if source_reference.attribute is None:
        return first_docstring_line(source_reference.owner_type)
    for clazz in source_reference.owner_type.__mro__:
        line = _attribute_docstrings(clazz).get(source_reference.attribute)
        if line is not None:
            return line
    return None
