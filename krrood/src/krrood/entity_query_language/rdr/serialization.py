"""Persist an EQL-native RDR as a Python module — no JSON, no strings-as-rules.

The rule tree is a live EQL expression DAG. To save it we *unparse* the
DAG back into the same ``with refinement(...) / alternative(...):
add(...)`` syntax used to author rule trees by hand, using the stdlib
:mod:`ast` module to guarantee syntactically valid output; loading is
just importing that module and reading the rebuilt DAG.
"""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from textwrap import indent
from uuid import UUID

from typing_extensions import TYPE_CHECKING, Any, Callable

from krrood.code_generation import ast_helpers
from krrood.code_generation.dataclass_source import DataclassConstructorSerializer
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
    ConclusionSelector,
    Next,
    Refinement,
)
from krrood.entity_query_language.rules.decomposition import DecomposedRuleTree
from krrood.ormatic.utils import classproperty

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR


# %% names a saved module defines


class RDRModuleAttribute(StrEnum):
    """The attributes a saved RDR module defines so that loading it can rebuild
    the RDR."""

    CASE_TYPE = "RDR_CASE_TYPE"
    """The RDR's case type."""

    CONCLUSION_ATTRIBUTE = "RDR_CONCLUSION_ATTRIBUTE"
    """The name of the case-type attribute the RDR concludes."""

    CASE_VARIABLE = "RDR_CASE_VARIABLE"
    """The ``Variable`` the rules are written over."""

    QUERY = "RDR_QUERY"
    """The built query holding the rule tree."""

    CORNER_CASES = "RDR_CORNER_CASES"
    """The corner cases, keyed by the position of their rule in pre-order."""

# %% rule tree -> AST


@dataclass(kw_only=True)
class RuleTreeSourceEmitter:
    """Builds the :mod:`ast` statements that rebuild a rule tree in the syntax
    rule trees are authored in by hand."""

    variable_names_by_id: dict[UUID, str]
    """The source name of each bound variable, keyed by the variable's id."""

    conclusion_target: ast.expr
    """The expression every rule's ``add(...)`` adds its conclusion to."""

    referenced_types: set[type] = field(default_factory=set)
    """The enum types of the emitted conclusions, which the generated module
    has to import."""

    @classproperty
    def conclusion_selector_factories(
        cls,
    ) -> dict[type[ConclusionSelector], Callable[..., SymbolicExpression]]:
        """The rule-authoring factory that adds a branch of each conclusion-
        selector type."""
        return {Refinement: refinement, Alternative: alternative, Next: next_rule}

    def emit_rule_body(self, *, rule_tree: DecomposedRuleTree) -> list[ast.stmt]:
        """Emit the ``add(...)`` statement of a rule, followed by a ``with``
        block for each of its branches.

        :param rule_tree: The rule and its branches.
        :return: The statements that rebuild the rule.
        """
        value = self.conclusion_value(condition=rule_tree.base_condition)
        if isinstance(value, enum.Enum):
            self.referenced_types.add(type(value))
        statements: list[ast.stmt] = [
            ast.Expr(
                value=ast_helpers.call(
                    add.__name__, [self.conclusion_target, self.emit_value(value=value)]
                )
            )
        ]
        for branch in rule_tree.branches:
            selector_call = self.emit_factory_call(
                factory=self.conclusion_selector_factories[
                    branch.conclusion_selector_type
                ],
                operands=[branch.subtree.base_condition],
            )
            nested_body = self.emit_rule_body(rule_tree=branch.subtree)
            statements.append(ast_helpers.with_block(selector_call, nested_body))
        return statements

    def emit_expression(self, *, node: SymbolicExpression) -> ast.expr:
        """Build the expression that rebuilds a condition sub-tree.

        :param node: The condition sub-tree.
        :return: The expression that rebuilds ``node``.
        """
        match node:
            case Literal():
                return self.emit_value(value=node._value_)
            case Attribute():
                return ast_helpers.attribute_access(
                    self.emit_expression(node=node._child_), node._attribute_name_
                )
            case Variable() if node._id_ in self.variable_names_by_id:
                return ast_helpers.load_name(self.variable_names_by_id[node._id_])
            case Comparator():
                return self.emit_comparison(node=node)
            case AND():
                return self.emit_factory_call(
                    factory=and_, operands=self.condition_operands(connective=node)
                )
            case OR():
                return self.emit_factory_call(
                    factory=or_, operands=self.condition_operands(connective=node)
                )
            case Not():
                return self.emit_factory_call(
                    factory=not_, operands=[node._children_[0]]
                )
            case _:
                raise UnsupportedNodeForSerialization(node=node)

    def emit_comparison(self, *, node: Comparator) -> ast.Compare:
        """Build the comparison that rebuilds a comparator node.

        :param node: The comparator sub-tree.
        :return: The comparison that rebuilds ``node``.
        """
        comparison_operator = ast_helpers.comparison_operator_class(node.operation)
        if comparison_operator is None:
            raise UnsupportedNodeForSerialization(node=node)
        return ast_helpers.compare(
            self.emit_expression(node=node.left),
            comparison_operator,
            self.emit_expression(node=node.right),
        )

    def emit_factory_call(
        self,
        *,
        factory: Callable[..., SymbolicExpression],
        operands: list[SymbolicExpression],
    ) -> ast.Call:
        """Build the call of a rule-authoring factory on the emitted operands.

        :param factory: The factory, called by its name.
        :param operands: The condition sub-trees passed as positional
            arguments.
        :return: The call.
        """
        return ast_helpers.call(
            factory.__name__,
            [self.emit_expression(node=operand) for operand in operands],
        )

    def emit_value(self, *, value: Any) -> ast.expr:
        """Build the expression that rebuilds a literal value.

        :param value: ``None``, or a value of one of
            :attr:`DataclassConstructorSerializer.supported_scalar_types`.
        :return: The expression that evaluates back to ``value``.
        """
        if value is not None and not isinstance(
            value, DataclassConstructorSerializer.supported_scalar_types
        ):
            raise UnsupportedNodeForSerialization(node=value)
        return ast_helpers.literal(value)

    def conclusion_value(self, *, condition: SymbolicExpression) -> Any:
        """The value a rule concludes.

        :param condition: The rule's condition, which holds its ``Add``
            conclusion.
        :return: The concluded literal value, or the target node when it
            is no literal.
        """
        for conclusion in condition._conclusions_:
            if isinstance(conclusion, Add):
                target = conclusion.right
                return target._value_ if isinstance(target, Literal) else target
        raise UnsupportedNodeForSerialization(node=condition)

    def condition_operands(
        self, *, connective: SymbolicExpression
    ) -> list[SymbolicExpression]:
        """The operands of a logical connective, without the ``Add``
        conclusions attached to it as children when it is a rule's whole
        condition.

        :param connective: The ``AND`` or ``OR`` node.
        :return: The operands that form the boolean expression.
        """
        conclusion_ids = {conclusion._id_ for conclusion in connective._conclusions_}
        return [
            child for child in connective._children_ if child._id_ not in conclusion_ids
        ]


# %% RDR <-> module


@dataclass(kw_only=True)
class RDRSerializer:
    """Writes an :class:`EQLSingleClassRDR` as a Python module that rebuilds it
    on import, and loads such a module back."""

    rdr: EQLSingleClassRDR
    """The RDR to write; it needs at least one rule."""

    @classproperty
    def rule_authoring_import(cls) -> str:
        """The import statement for every factory the generated rule tree
        calls."""
        return "\n".join(
            get_imports_from_types(
                [
                    variable,
                    entity,
                    add,
                    refinement,
                    alternative,
                    next_rule,
                    and_,
                    or_,
                    not_,
                ]
            )
        )

    @classproperty
    def templates_directory(cls) -> Path:
        """The directory holding this package's code-generation templates."""
        return Path(__file__).parent / "templates"

    @classproperty
    def module_template_name(cls) -> str:
        """The file name of the template a saved RDR module is rendered
        from."""
        return "rdr_module.py.jinja"

    @classproperty
    def case_class_and_rules_separator(cls) -> str:
        """The blank lines between a generated case class and the rule tree
        after it."""
        return "\n\n\n"

    @classproperty
    def loaded_module_name_prefix(cls) -> str:
        """The prefix of the unique name a saved RDR module is imported
        under."""
        return "_eql_rdr_loaded_"

    def to_source(self, *, case_type_is_local: bool = False) -> str:
        """Python module source that rebuilds the same rule tree on import.

        :param case_type_is_local: Whether the case type is defined in
            the same file, so the module must not import it.
        :return: The module source.
        """
        if self.rdr.query is None:
            raise EmptyRuleTreeError()

        case_type = self.rdr.case_type
        variable_name = camel_case_to_lower_camel_case(case_type.__name__)
        emitter = RuleTreeSourceEmitter(
            variable_names_by_id={self.rdr.case_variable._id_: variable_name},
            conclusion_target=ast_helpers.attribute_access(
                ast_helpers.load_name(variable_name), self.rdr.conclusion_attribute_name
            ),
            referenced_types={case_type},
        )
        rule_tree = DecomposedRuleTree.from_condition(
            condition=self.rdr.query._conditions_root_
        )
        base_condition = ast_helpers.unparse_expression(
            emitter.emit_expression(node=rule_tree.base_condition)
        )
        body = indent(
            ast_helpers.unparse_statements(emitter.emit_rule_body(rule_tree=rule_tree)),
            "    ",
        )
        corner_cases = self._render_corner_cases(
            rule_tree=rule_tree, referenced_types=emitter.referenced_types
        )
        type_imports = self._render_type_imports(
            referenced_types=emitter.referenced_types,
            case_type_is_local=case_type_is_local,
        )

        generator = CodeGenerator(template_directory=str(self.templates_directory))
        return generator.render(
            template_name=self.module_template_name,
            factory_import=self.rule_authoring_import,
            type_imports=type_imports,
            variable_name=variable_name,
            case_type_name=case_type.__name__,
            base_condition=base_condition,
            body=body,
            conclusion_attribute_name=self.rdr.conclusion_attribute_name,
            corner_cases_dict_src=corner_cases,
            module_attributes=RDRModuleAttribute,
        )

    def _render_corner_cases(
        self, *, rule_tree: DecomposedRuleTree, referenced_types: set[type]
    ) -> str:
        """Render the ``{index: constructor, ...}`` dict literal of the corner
        cases.

        :param rule_tree: The decomposed rule tree, whose rule order
            sets the indices.
        :param referenced_types: Extended in place with the types the
            corner cases need.
        :return: The dict literal source, ``"{}"`` when there are no
            corner cases.
        """
        corner_case_sources = self.rdr.corner_cases.to_ordered_sources(
            rule_tree.rule_conditions()
        )
        for case_source in corner_case_sources.values():
            referenced_types.update(case_source.referenced_types)
        return render_dict_literal(
            (repr(index), case_source.source)
            for index, case_source in sorted(corner_case_sources.items())
        )

    def _render_type_imports(
        self, *, referenced_types: set[type], case_type_is_local: bool
    ) -> str:
        """Render the import statements for every type the generated module
        references.

        :param referenced_types: The types the rule tree and the corner
            cases reference.
        :param case_type_is_local: Whether to leave out the case type's
            own import.
        :return: The newline-joined import statements.
        """
        types_to_import = [
            referenced_type
            for referenced_type in referenced_types
            if not (case_type_is_local and referenced_type is self.rdr.case_type)
        ]
        return "\n".join(get_imports_from_types(types_to_import))

    def save(self, *, path: str) -> str:
        """Write the RDR's module to ``path``.

        :param path: The destination ``.py`` file.
        :return: The source written.
        """
        source = self.to_source()
        Path(path).write_text(source)
        return source

    def save_with_case(self, *, path: str) -> str:
        """Write the RDR's module to ``path``, preceded by the definition of
        its case type when that is a :class:`FunctionCase`; otherwise the same
        as :meth:`save`.

        :param path: The destination ``.py`` file.
        :return: The source written.
        """
        from krrood.entity_query_language.rdr.function_case import FunctionCase

        case_type = self.rdr.case_type
        if not (isinstance(case_type, type) and issubclass(case_type, FunctionCase)):
            return self.save(path=path)
        class_source = FunctionCaseGenerator().generate(
            function=case_type.function, class_name=case_type.__name__
        )
        source = (
            class_source
            + self.case_class_and_rules_separator
            + self.to_source(case_type_is_local=True)
        )
        Path(path).write_text(source)
        return source

    @classmethod
    def load(cls, *, path: str) -> EQLSingleClassRDR:
        """Load an RDR from a module written by :meth:`save` or
        :meth:`save_with_case`.

        :param path: The ``.py`` file.
        :return: The rebuilt RDR.
        """
        from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

        module = load_module_from_path(path, cls.loaded_module_name_prefix)
        module_attributes = vars(module)

        conclusion_attribute_name = module_attributes[
            RDRModuleAttribute.CONCLUSION_ATTRIBUTE
        ]
        rdr = EQLSingleClassRDR(
            module_attributes[RDRModuleAttribute.CASE_TYPE], conclusion_attribute_name
        )
        rdr.case_variable = module_attributes[RDRModuleAttribute.CASE_VARIABLE]
        rdr.conclusion_variable = getattr(rdr.case_variable, conclusion_attribute_name)
        rdr.query = module_attributes[RDRModuleAttribute.QUERY]

        rule_conditions = DecomposedRuleTree.from_condition(
            condition=rdr.conditions_root
        ).rule_conditions()
        rdr.corner_cases = CornerCaseStore.from_ordered_cases(
            rule_conditions, module_attributes.get(RDRModuleAttribute.CORNER_CASES, {})
        )
        return rdr
