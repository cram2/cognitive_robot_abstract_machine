"""
Persist an EQL-native RDR as a Python module — no JSON, no strings-as-rules.

The rule tree is a live EQL expression DAG. To save it we *unparse* the DAG back into
the same ``with refinement(...) / alternative(...): add(...)`` syntax used to author rule
trees by hand, using the stdlib :mod:`ast` module to guarantee syntactically valid output;
loading is just importing that module and reading the rebuilt DAG.
"""

from __future__ import annotations

import ast
import enum
import operator
import os
from dataclasses import dataclass
from textwrap import indent as _indent
from typing import TYPE_CHECKING
from uuid import UUID

from typing_extensions import Any, Callable, Dict, List, Optional, Set, Type, Union

from krrood.code_generation import ast_helpers
from krrood.code_generation.dataclass_source import ObjectSource
from krrood.code_generation.function_case import FunctionCaseGenerator
from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.imports import get_imports_from_types
from krrood.code_generation.module_loading import load_module_from_path
from krrood.code_generation.naming import camel_case_to_lower_camel_case
from krrood.code_generation.object_to_source import render_dict_literal
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.factories import (
    add,
    alternative,
    and_,
    entity,
    next_rule,
    not_,
    or_,
    refinement,
    variable,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore
from krrood.entity_query_language.rdr.exceptions import (
    EmptyRuleTreeError,
    UnsupportedNodeForSerialization,
)
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    Next,
    Refinement,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

SerializableValue = Union[enum.Enum, bool, int, float, str, None]
"""The literal value types a rule conclusion may hold and that :func:`_emit_value` can emit."""

_CONCLUSION_SELECTORS = (Refinement, Alternative, Next)
"""The conclusion-selector classes whose left-nested chains a rule tree is built from."""

_CONCLUSION_SELECTOR_FACTORY: Dict[type, Callable[..., SymbolicExpression]] = {
    Refinement: refinement,
    Alternative: alternative,
    Next: next_rule,
}
"""Maps each conclusion-selector class to the factory that reconstructs it in generated source."""

_COMPARATOR_AST_OPERATOR: Dict[Callable[..., Any], Type[ast.cmpop]] = {
    operator.eq: ast.Eq,
    operator.ne: ast.NotEq,
    operator.lt: ast.Lt,
    operator.le: ast.LtE,
    operator.gt: ast.Gt,
    operator.ge: ast.GtE,
}
"""Maps each comparator operation to the AST comparison-operator class that renders it."""

_FACTORY_IMPORT = "\n".join(
    get_imports_from_types(
        [variable, entity, add, refinement, alternative, next_rule, and_, or_, not_]
    )
)
"""The import statement that makes every rule-authoring factory available in generated source."""

_CLASS_AND_RULES_SEPARATOR = "\n\n\n"
"""Blank-line separator placed between the generated case class and the rule-tree source."""

_TEMPLATES_DIRECTORY = os.path.join(os.path.dirname(__file__), "templates")
"""Absolute path to the directory holding this package's code-generation templates."""

_RDR_MODULE_TEMPLATE_NAME = "rdr_module.py.jinja"
"""Filename of the Jinja2 template used to render a saved RDR module."""

_LOADED_MODULE_NAME_PREFIX = "_eql_rdr_loaded_"
"""Prefix for the uuid-suffixed module name a saved RDR file is imported under."""

RDR_CASE_TYPE_NAME = "RDR_CASE_TYPE"
"""Name of the generated module attribute holding the RDR's case type."""

RDR_CONCLUSION_ATTRIBUTE_NAME = "RDR_CONCLUSION_ATTRIBUTE"
"""Name of the generated module attribute holding the RDR's conclusion attribute name."""

RDR_CASE_VARIABLE_NAME = "RDR_CASE_VARIABLE"
"""Name of the generated module attribute holding the RDR's case ``Variable``."""

RDR_QUERY_NAME = "RDR_QUERY"
"""Name of the generated module attribute holding the RDR's built query."""

RDR_CORNER_CASES_NAME = "RDR_CORNER_CASES"
"""Name of the generated module attribute holding the RDR's corner-case source dict."""

_class_name_to_variable_name = camel_case_to_lower_camel_case
"""Derives the rule-tree variable name from the case-type class name."""


# %% DAG decomposition


@dataclass
class ConclusionSelectorBranch:
    """One reinsertion step: a conclusion-selector type paired with the branch condition it wraps."""

    conclusion_selector_type: type
    """The selector class (``Refinement``, ``Alternative``, or ``Next``) this branch belongs to."""

    condition: SymbolicExpression
    """The branch's condition sub-tree (an EQL expression DAG node)."""


@dataclass
class DecomposedRuleTree:
    """The result of flattening a left-nested chain of conclusion selectors."""

    base_condition: SymbolicExpression
    """The anchor condition node with no selector wrapping."""

    branches: List[ConclusionSelectorBranch]
    """The selector branches, ordered as the loader must re-insert them."""


def _branch_grew_the_tree_inward(branch: ConclusionSelectorBranch) -> bool:
    """
    Report whether a branch's selector grew the rule tree inward rather than outward.

    A ``Refinement`` re-anchors at the same fixed base node, so successive refinements
    grow *inward* and the left-to-right walk visits them in the order they were added.
    An ``Alternative`` / ``Next`` re-anchors at the moving conditions root, so those grow
    *outward* and the walk visits them in the reverse of the order they were added.

    :param branch: The branch to classify.
    :return: ``True`` when the branch was added by an inward-growing refinement,
        ``False`` for outward-growing alternative/next branches.
    """
    return branch.conclusion_selector_type is Refinement


def _reorder_branches_for_reinsertion(
    branches_added_the_same_direction: List[ConclusionSelectorBranch],
) -> List[ConclusionSelectorBranch]:
    """
    Order one group of same-direction branches the way the loader must re-insert them.

    Inward-growing (refinement) branches are already in the order they were added, so the
    group is kept as-is. Outward-growing (alternative/next) branches were walked in reverse
    of the order they were added, so the group is flipped back.

    :param branches_added_the_same_direction: Consecutive branches that were all added by
        the same kind of selector (all refinements, or all alternatives/nexts).
    :return: The group in the order the loader re-inserts it.
    """
    if not branches_added_the_same_direction:
        return []
    if _branch_grew_the_tree_inward(branches_added_the_same_direction[0]):
        return list(branches_added_the_same_direction)
    return list(reversed(branches_added_the_same_direction))


def _flatten_selector_chain(node: SymbolicExpression) -> DecomposedRuleTree:
    """
    Flatten a left-nested chain of conclusion selectors into a base condition plus an
    ordered list of branches in the order the loader must re-insert them to rebuild the
    identical DAG.

    Because refinement and alternative branches were added growing in opposite directions
    (see :func:`_branch_grew_the_tree_inward`), the walked branches are split into maximal
    groups that were all added the same direction, and each group is ordered independently;
    a blanket reverse would silently swap sibling refinements on every round-trip.

    :param node: The (possibly selector-wrapped) condition node to flatten.
    :return: The decomposed rule tree (base condition plus ordered branches).
    """
    walked_branches: List[ConclusionSelectorBranch] = []
    while isinstance(node, _CONCLUSION_SELECTORS):
        walked_branches.append(ConclusionSelectorBranch(type(node), node.right))
        node = node.left

    branches: List[ConclusionSelectorBranch] = []
    current_group: List[ConclusionSelectorBranch] = []
    for branch in walked_branches:
        # Close the current group whenever this branch was added growing the opposite
        # direction from the branches already in it, so each same-direction group can be
        # ordered on its own.
        breaks_the_current_group = bool(current_group) and (
            _branch_grew_the_tree_inward(branch)
            != _branch_grew_the_tree_inward(current_group[0])
        )
        if breaks_the_current_group:
            branches.extend(_reorder_branches_for_reinsertion(current_group))
            current_group = []
        current_group.append(branch)
    branches.extend(_reorder_branches_for_reinsertion(current_group))
    return DecomposedRuleTree(node, branches)


def walk_rules_in_emission_order(
    conditions_root: Optional[SymbolicExpression],
) -> List[SymbolicExpression]:
    """
    Return condition (leaf) nodes in the same pre-order that ``_emit_rule_body`` visits.

    This is the single authoritative ordering shared by the serializer (save path) and
    the corner-case-store rebuilder (load path). Both must use this function so the two
    orderings can never drift independently.

    :param conditions_root: The root of the rule-tree condition DAG.
    :return: Leaf condition nodes in the order the serializer emits their ``add(...)``
        calls (i.e. the *i*-th element here corresponds to the *i*-th ``add(`` line in
        the file written by :func:`rdr_to_python`).
    """
    result: List[SymbolicExpression] = []

    def _visit(node: SymbolicExpression) -> None:
        decomposed = _flatten_selector_chain(node)
        result.append(decomposed.base_condition)
        for branch in decomposed.branches:
            _visit(branch.condition)

    if conditions_root is not None:
        _visit(conditions_root)
    return result


def _conclusion_value(
    condition_node: SymbolicExpression,
) -> Union[SerializableValue, SymbolicExpression]:
    """Return the single value concluded at a condition node.

    :param condition_node: The rule condition whose ``Add`` conclusion is read.
    :return: The concluded literal value, or the target node when it is not a literal.
    """
    for conclusion in condition_node._conclusions_:
        if isinstance(conclusion, Add):
            target = conclusion.right
            return target._value_ if isinstance(target, Literal) else target
    raise UnsupportedNodeForSerialization(condition_node)


# %% Expression -> AST


def _emit_value(value: SerializableValue) -> ast.expr:
    """Build the AST expression that reconstructs a literal value.

    :param value: The literal (or enum member) to emit.
    :return: The AST expression that evaluates back to *value*.
    """
    match value:
        case enum.Enum():
            return ast_helpers.enum_member(value)
        case bool() | int() | float() | str() | None:
            return ast_helpers.constant(value)
        case _:
            raise UnsupportedNodeForSerialization(value)


def _emit_factory_call(
    factory: Callable[..., SymbolicExpression],
    operands: List[SymbolicExpression],
    variable_names_by_id: Dict[UUID, str],
) -> ast.Call:
    """Build the AST call that invokes a factory on the emitted operands.

    :param factory: The rule-authoring factory function to call by name.
    :param operands: The condition sub-trees to pass as positional arguments.
    :param variable_names_by_id: Maps each bound variable's id to its source name.
    :return: The AST call node.
    """
    return ast_helpers.call(
        factory.__name__,
        [_emit_expression(operand, variable_names_by_id) for operand in operands],
    )


def _emit_comparison(
    node: Comparator, variable_names_by_id: Dict[UUID, str]
) -> ast.Compare:
    """Build the AST comparison that reconstructs a comparator node.

    :param node: The comparator sub-tree to emit.
    :param variable_names_by_id: Maps each bound variable's id to its source name.
    :return: The AST comparison node.
    """
    comparison_operator = _COMPARATOR_AST_OPERATOR.get(node.operation)
    if comparison_operator is None:
        raise UnsupportedNodeForSerialization(node)
    return ast_helpers.compare(
        _emit_expression(node.left, variable_names_by_id),
        comparison_operator,
        _emit_expression(node.right, variable_names_by_id),
    )


def _emit_expression(
    node: SymbolicExpression, variable_names_by_id: Dict[UUID, str]
) -> ast.expr:
    """Build the AST expression that reconstructs a condition sub-tree.

    :param node: The condition sub-tree to emit.
    :param variable_names_by_id: Maps each bound variable's id to its source name.
    :return: The AST expression that rebuilds *node*.
    """
    match node:
        case Literal():
            return _emit_value(node._value_)
        case Attribute():
            return ast_helpers.attribute_access(
                _emit_expression(node._child_, variable_names_by_id),
                node._attribute_name_,
            )
        case Variable() if node._id_ in variable_names_by_id:
            return ast_helpers.load_name(variable_names_by_id[node._id_])
        case Comparator():
            return _emit_comparison(node, variable_names_by_id)
        case AND():
            return _emit_factory_call(
                and_, _condition_operands(node), variable_names_by_id
            )
        case OR():
            return _emit_factory_call(
                or_, _condition_operands(node), variable_names_by_id
            )
        case Not():
            return _emit_factory_call(
                not_, [node._children_[0]], variable_names_by_id
            )
        case _:
            raise UnsupportedNodeForSerialization(node)


def _condition_operands(node: SymbolicExpression) -> List[SymbolicExpression]:
    """
    The logical operands of a connective, excluding any conclusions. When a rule's
    condition is the connective itself (e.g. an ``and_`` root), its ``Add`` conclusions are
    attached as children too; those are not part of the boolean expression.

    :param node: The connective node to read operands from.
    :return: The operand children that form the boolean expression.
    """
    conclusion_ids = {conclusion._id_ for conclusion in node._conclusions_}
    return [child for child in node._children_ if child._id_ not in conclusion_ids]


# %% Rule tree -> AST


def _emit_rule_body(
    condition_node: SymbolicExpression,
    variable_names_by_id: Dict[UUID, str],
    conclusion_target: ast.expr,
    referenced_types: Set[type],
) -> List[ast.stmt]:
    """Emit the ``add(...)`` statement plus any nested refinement/alternative blocks for a rule.

    :param condition_node: The rule condition to emit statements for.
    :param variable_names_by_id: Maps each bound variable's id to its source name.
    :param conclusion_target: The AST expression the conclusion value is assigned to.
    :param referenced_types: Accumulates enum types that need importing; mutated in place.
    :return: The AST statements forming this rule's body.
    """
    decomposed = _flatten_selector_chain(condition_node)
    value = _conclusion_value(decomposed.base_condition)
    if isinstance(value, enum.Enum):
        referenced_types.add(type(value))

    statements: List[ast.stmt] = [
        ast.Expr(value=ast_helpers.call(add.__name__, [conclusion_target, _emit_value(value)]))
    ]
    for branch in decomposed.branches:
        factory = _CONCLUSION_SELECTOR_FACTORY[branch.conclusion_selector_type]
        branch_base_condition = _flatten_selector_chain(branch.condition).base_condition
        selector_call = _emit_factory_call(
            factory, [branch_base_condition], variable_names_by_id
        )
        nested_body = _emit_rule_body(
            branch.condition, variable_names_by_id, conclusion_target, referenced_types
        )
        statements.append(ast_helpers.with_block(selector_call, nested_body))
    return statements


def _render_corner_cases_source(
    corner_case_sources: Dict[int, ObjectSource], referenced_types: Set[type]
) -> str:
    """Render the ``{index: constructor, ...}`` corner-case dict literal.

    :param corner_case_sources: The per-rule corner-case constructor sources, keyed by
        emission index.
    :param referenced_types: Accumulates the types each corner case needs imported;
        mutated in place.
    :return: The corner-case dict source (``"{}"`` when there are none).
    """
    for case_source in corner_case_sources.values():
        referenced_types.update(case_source.referenced_types)
    return render_dict_literal(
        (repr(positional_index), case_source.source)
        for positional_index, case_source in sorted(corner_case_sources.items())
    )


def _render_type_imports(
    referenced_types: Set[type], case_type: type, case_type_is_local: bool
) -> str:
    """Render the import block for every type the generated module references.

    :param referenced_types: The types referenced by the rule tree and corner cases.
    :param case_type: The RDR's case type.
    :param case_type_is_local: When ``True``, omit the case type's own import (its class
        is already defined in the same file).
    :return: The newline-joined import statements.
    """
    if case_type_is_local:
        types_to_import = [
            referenced_type
            for referenced_type in referenced_types
            if referenced_type is not case_type
        ]
    else:
        types_to_import = list(referenced_types)
    return "\n".join(get_imports_from_types(types_to_import))


def rdr_to_python(rdr: EQLSingleClassRDR, case_type_is_local: bool = False) -> str:
    """
    Serialize an :class:`EQLSingleClassRDR` to importable Python source.

    :param rdr: A fitted RDR (must have at least one rule).
    :param case_type_is_local: When ``True``, skip emitting the import for the case type
        itself. Use this when ``save_rdr_with_case`` has already written the class
        definition at the top of the same file.
    :return: Python module source that rebuilds the same rule-tree DAG on import.
    """
    if rdr.query is None:
        raise EmptyRuleTreeError()

    case_type = rdr.case_type
    variable_name = _class_name_to_variable_name(case_type.__name__)
    variable_names_by_id = {rdr.case_variable._id_: variable_name}
    conclusion_target = ast_helpers.attribute_access(
        ast_helpers.load_name(variable_name), rdr.conclusion_attribute_name
    )
    referenced_types: Set[type] = {case_type}

    decomposed = _flatten_selector_chain(rdr.query._conditions_root_)
    base_condition = ast_helpers.unparse_expression(
        _emit_expression(decomposed.base_condition, variable_names_by_id)
    )
    body_statements = _emit_rule_body(
        rdr.query._conditions_root_, variable_names_by_id, conclusion_target, referenced_types
    )
    body = _indent(ast_helpers.unparse_statements(body_statements), "    ")

    ordered_nodes = walk_rules_in_emission_order(rdr.query._conditions_root_)
    corner_cases_dict_src = _render_corner_cases_source(
        rdr.corner_cases.to_ordered_sources(ordered_nodes), referenced_types
    )
    type_imports = _render_type_imports(referenced_types, case_type, case_type_is_local)

    generator = CodeGenerator(template_directory=_TEMPLATES_DIRECTORY)
    return generator.render(
        template_name=_RDR_MODULE_TEMPLATE_NAME,
        factory_import=_FACTORY_IMPORT,
        type_imports=type_imports,
        variable_name=variable_name,
        case_type_name=case_type.__name__,
        base_condition=base_condition,
        body=body,
        conclusion_attribute_name=rdr.conclusion_attribute_name,
        corner_cases_dict_src=corner_cases_dict_src,
        rdr_case_type_name=RDR_CASE_TYPE_NAME,
        rdr_conclusion_attribute_name=RDR_CONCLUSION_ATTRIBUTE_NAME,
        rdr_case_variable_name=RDR_CASE_VARIABLE_NAME,
        rdr_query_name=RDR_QUERY_NAME,
        rdr_corner_cases_name=RDR_CORNER_CASES_NAME,
    )


def save_rdr(rdr: EQLSingleClassRDR, path: str) -> str:
    """Write the RDR's Python source to a file and return that source.

    :param rdr: The fitted RDR to serialize.
    :param path: Destination ``.py`` file path.
    :return: The source written to disk.
    """
    source = rdr_to_python(rdr)
    with open(path, "w") as file:
        file.write(source)
    return source


def save_rdr_with_case(rdr: EQLSingleClassRDR, path: str) -> str:
    """
    Write a combined class-header + rule-tree file to ``path``.

    When ``rdr.case_type`` is a :class:`FunctionCase` subclass the file begins with the
    ``@dataclass`` class definition (generated from the original function stored in
    ``case_type.function``), followed by the rule-tree section which omits the case-type
    import (the class is already defined above). For any other case type the function
    falls back to plain :func:`save_rdr`.

    :param rdr: A fitted :class:`EQLSingleClassRDR`.
    :param path: Destination ``.py`` file path.
    :return: The source written to disk.
    """
    from krrood.entity_query_language.rdr.function_case import FunctionCase

    if isinstance(rdr.case_type, type) and issubclass(rdr.case_type, FunctionCase):
        class_source = FunctionCaseGenerator().generate(
            function=rdr.case_type.function,
            class_name=rdr.case_type.__name__,
        )
        rule_source = rdr_to_python(rdr, case_type_is_local=True)
        source = class_source + _CLASS_AND_RULES_SEPARATOR + rule_source
    else:
        source = rdr_to_python(rdr)

    with open(path, "w") as file:
        file.write(source)
    return source


def load_rdr(path: str) -> EQLSingleClassRDR:
    """Load an :class:`EQLSingleClassRDR` from a module previously written by :func:`save_rdr`.

    :param path: Path to a ``.py`` file produced by the save path.
    :return: The rebuilt RDR.
    """
    from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

    module = load_module_from_path(path, _LOADED_MODULE_NAME_PREFIX)

    case_type = getattr(module, RDR_CASE_TYPE_NAME)
    conclusion_attribute_name = getattr(module, RDR_CONCLUSION_ATTRIBUTE_NAME)
    rdr = EQLSingleClassRDR(case_type, conclusion_attribute_name)
    rdr.case_variable = getattr(module, RDR_CASE_VARIABLE_NAME)
    rdr.conclusion_variable = getattr(rdr.case_variable, conclusion_attribute_name)
    rdr.query = getattr(module, RDR_QUERY_NAME)

    cases_by_index = getattr(module, RDR_CORNER_CASES_NAME, {})
    ordered_nodes = walk_rules_in_emission_order(rdr.conditions_root)
    rdr.corner_cases = CornerCaseStore.from_ordered_cases(ordered_nodes, cases_by_index)

    return rdr
