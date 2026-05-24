"""
Persist an EQL-native RDR as a Python module — no JSON, no strings-as-rules.

The rule tree is a live EQL expression DAG. To save it we *unparse* the DAG back into
the same ``with refinement(...) / alternative(...): add(...)`` syntax used to author rule
trees by hand; loading is just importing that module and reading the rebuilt DAG.
"""

from __future__ import annotations

import enum
import importlib.util
import operator
import uuid
from textwrap import indent as _indent

from typing_extensions import Any, Dict, List, Tuple

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    Next,
    Refinement,
)

_SELECTORS = (Refinement, Alternative, Next)

_SELECTOR_FACTORY = {
    Refinement: "refinement",
    Alternative: "alternative",
    Next: "next_rule",
}

_COMPARATOR_SYMBOL = {
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
}

_FACTORY_IMPORT = (
    "from krrood.entity_query_language.factories import (\n"
    "    variable,\n    entity,\n    add,\n    refinement,\n    alternative,\n"
    "    next_rule,\n    and_,\n    or_,\n    not_,\n)"
)


class UnsupportedNodeForSerialization(Exception):
    """Raised when the rule-tree DAG contains a node the serializer cannot emit."""

    def __init__(self, node: Any) -> None:
        super().__init__(
            f"Cannot serialize node of type {type(node).__name__!r} to Python source."
        )
        self.node = node


# --------------------------------------------------------------------------- #
# DAG decomposition
# --------------------------------------------------------------------------- #


def _decompose(node) -> Tuple[Any, List[Tuple[type, Any]]]:
    """
    Flatten a left-nested chain of conclusion selectors into a base condition plus an
    ordered list of ``(selector_type, branch_condition)`` pairs (insertion order).
    """
    branches: List[Tuple[type, Any]] = []
    while isinstance(node, _SELECTORS):
        branches.append((type(node), node.right))
        node = node.left
    branches.reverse()
    return node, branches


def _conclusion_value(condition_node) -> Any:
    """:return: The single value concluded at ``condition_node`` (its ``Add``)."""
    for conclusion in condition_node._conclusions_:
        if isinstance(conclusion, Add):
            target = conclusion.right
            return target._value_ if isinstance(target, Literal) else target
    raise UnsupportedNodeForSerialization(condition_node)


# --------------------------------------------------------------------------- #
# Expression -> source
# --------------------------------------------------------------------------- #


def _emit_value(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    raise UnsupportedNodeForSerialization(value)


def _emit_expr(node, var_names: Dict[uuid.UUID, str]) -> str:
    if isinstance(node, Literal):
        return _emit_value(node._value_)
    if isinstance(node, Attribute):
        return f"{_emit_expr(node._child_, var_names)}.{node._attribute_name_}"
    if isinstance(node, Variable):
        if node._id_ not in var_names:
            raise UnsupportedNodeForSerialization(node)
        return var_names[node._id_]
    if isinstance(node, Comparator):
        symbol = _COMPARATOR_SYMBOL.get(node.operation)
        if symbol is None:
            raise UnsupportedNodeForSerialization(node)
        return (
            f"{_emit_expr(node.left, var_names)} {symbol} "
            f"{_emit_expr(node.right, var_names)}"
        )
    if isinstance(node, AND):
        operands = _condition_operands(node)
        return "and_(" + ", ".join(_emit_expr(c, var_names) for c in operands) + ")"
    if isinstance(node, OR):
        operands = _condition_operands(node)
        return "or_(" + ", ".join(_emit_expr(c, var_names) for c in operands) + ")"
    if isinstance(node, Not):
        return f"not_({_emit_expr(node._children_[0], var_names)})"
    raise UnsupportedNodeForSerialization(node)


def _condition_operands(node) -> List[Any]:
    """
    The logical operands of a connective, excluding any conclusions. When a rule's
    condition is the connective itself (e.g. an ``and_`` root), its ``Add`` conclusions are
    attached as children too; those are not part of the boolean expression.
    """
    conclusion_ids = {c._id_ for c in getattr(node, "_conclusions_", ())}
    return [child for child in node._children_ if child._id_ not in conclusion_ids]


# --------------------------------------------------------------------------- #
# Rule tree -> source
# --------------------------------------------------------------------------- #


def _emit_rule_body(
    condition_node,
    var_names: Dict[uuid.UUID, str],
    conclusion_var_source: str,
    referenced_types: set,
) -> str:
    """Emit the ``add(...)`` plus any nested refinement/alternative blocks for a rule."""
    main, branches = _decompose(condition_node)
    value = _conclusion_value(main)
    if isinstance(value, enum.Enum):
        referenced_types.add(type(value))

    lines = [f"add({conclusion_var_source}, {_emit_value(value)})"]
    for selector_type, branch in branches:
        factory = _SELECTOR_FACTORY[selector_type]
        branch_main, _ = _decompose(branch)
        condition_src = _emit_expr(branch_main, var_names)
        body = _emit_rule_body(
            branch, var_names, conclusion_var_source, referenced_types
        )
        lines.append(f"with {factory}({condition_src}):")
        lines.append(_indent(body, "    "))
    return "\n".join(lines)


def rdr_to_python(rdr) -> str:
    """
    Serialize an :class:`EQLSingleClassRDR` to importable Python source.

    :param rdr: A fitted RDR (must have at least one rule).
    :return: Python module source that rebuilds the same rule-tree DAG on import.
    """
    if rdr.query is None:
        raise ValueError("Cannot serialize an empty RDR (no rules have been added).")

    case_type = rdr.case_type
    var_name = case_type.__name__[0].lower() + case_type.__name__[1:]
    var_names = {rdr.case_variable._id_: var_name}
    conclusion_var_source = f"{var_name}.{rdr.conclusion_attribute_name}"
    referenced_types = {case_type}

    root_main, _ = _decompose(rdr.query._conditions_root_)
    base_condition = _emit_expr(root_main, var_names)
    body = _emit_rule_body(
        rdr.query._conditions_root_, var_names, conclusion_var_source, referenced_types
    )

    # Group imported names by module so e.g. Animal and Species share one import line.
    imports_by_module: Dict[str, set] = {}
    for t in referenced_types:
        imports_by_module.setdefault(t.__module__, set()).add(t.__name__)
    type_imports = "\n".join(
        f"from {module} import {', '.join(sorted(names))}"
        for module, names in sorted(imports_by_module.items())
    )

    lines = [
        '"""Auto-generated EQL-RDR rule tree. Do not edit by hand."""',
        _FACTORY_IMPORT,
        type_imports,
        "",
        f"{var_name} = variable({case_type.__name__}, domain=[])",
        f"query = entity({var_name}).where({base_condition})",
        "with query:",
        _indent(body, "    "),
        "query.build()",
        "",
        "# Stable handles for loading.",
        "RDR_CASE_TYPE = " + case_type.__name__,
        f"RDR_CONCLUSION_ATTRIBUTE = {rdr.conclusion_attribute_name!r}",
        f"RDR_CASE_VARIABLE = {var_name}",
        "RDR_QUERY = query",
        "",
    ]
    return "\n".join(lines)


def save_rdr(rdr, path: str) -> str:
    """Write the RDR's Python source to ``path`` and return that source."""
    source = rdr_to_python(rdr)
    with open(path, "w") as f:
        f.write(source)
    return source


def load_rdr(path: str):
    """
    Load an :class:`EQLSingleClassRDR` from a module previously written by :func:`save_rdr`.
    """
    from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

    spec = importlib.util.spec_from_file_location("_eql_rdr_loaded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rdr = EQLSingleClassRDR(module.RDR_CASE_TYPE, module.RDR_CONCLUSION_ATTRIBUTE)
    rdr.case_variable = module.RDR_CASE_VARIABLE
    rdr.conclusion_variable = getattr(
        module.RDR_CASE_VARIABLE, module.RDR_CONCLUSION_ATTRIBUTE
    )
    rdr.query = module.RDR_QUERY
    return rdr
