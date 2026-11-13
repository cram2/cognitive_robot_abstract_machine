"""Persist an EQL-native RDR as a Python module — no JSON, no strings-as-rules.

The rule tree is a live EQL expression DAG. To save it we *unparse* the
DAG back into the same ``with refinement(...) / alternative(...):
add(...)`` syntax used to author rule trees by hand, using the stdlib
:mod:`ast` module to guarantee syntactically valid output; loading is
just importing that module and reading the rebuilt DAG.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from textwrap import indent

from typing_extensions import TYPE_CHECKING

from krrood.code_generation.ast_helpers import (
    ast_attribute_access,
    ast_call,
    ast_load_name,
    ast_unparse_expression,
    ast_unparse_statements,
    ast_with_block,
)
from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.imports import get_imports_from_types
from krrood.code_generation.naming import camel_case_to_lower_camel_case
from krrood.code_generation.object_to_source import render_dict_literal
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import add, entity, variable
from krrood.entity_query_language.rdr.exceptions import EmptyRuleTreeError
from krrood.entity_query_language.exceptions import UnsupportedNodeForSerialization
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.decomposition import DecomposedRuleTree
from krrood.entity_query_language.source_emission import (
    BranchSourceEmitter,
    ConditionEmissionContext,
    ConditionSourceEmitter,
    FactoryCallSourceEmitter,
)
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

    context: ConditionEmissionContext
    """The state shared by every emitter of the generated source."""

    conclusion_target: ast.expr
    """The expression every rule's ``add(...)`` adds its conclusion to."""

    def emit_rule_body(self, *, rule_tree: DecomposedRuleTree) -> list[ast.stmt]:
        """Emit the ``add(...)`` statement of a rule, followed by a ``with``
        block for each of its branches.

        :param rule_tree: The rule and its branches.
        :return: The statements that rebuild the rule.
        """
        conclusion = self.conclusion_node(condition=rule_tree.base_condition)
        statements: list[ast.stmt] = [
            ast.Expr(
                value=ast_call(
                    add.__name__,
                    [
                        self.conclusion_target,
                        ConditionSourceEmitter.emit_source(
                            subject=conclusion, context=self.context
                        ),
                    ],
                )
            )
        ]
        for branch in rule_tree.branches:
            branch_opening = BranchSourceEmitter.emit_source(
                subject=branch, context=self.context
            )
            nested_body = self.emit_rule_body(rule_tree=branch.subtree)
            statements.append(ast_with_block(branch_opening, nested_body))
        return statements

    def conclusion_node(self, *, condition: SymbolicExpression) -> SymbolicExpression:
        """The node holding the value a rule concludes.

        :param condition: The rule's condition, which holds its ``Add``
            conclusion.
        :return: The node the ``Add`` conclusion adds.
        """
        for conclusion in condition._conclusions_:
            if isinstance(conclusion, Add):
                return conclusion.right
        raise UnsupportedNodeForSerialization(node=condition)


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
        emitted_factories = [
            emitter.factory
            for emitter in ConditionSourceEmitter.family_members
            + BranchSourceEmitter.family_members
            if issubclass(emitter, (FactoryCallSourceEmitter, BranchSourceEmitter))
        ]
        return "\n".join(
            get_imports_from_types([variable, entity, add, *emitted_factories])
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
        context = ConditionEmissionContext(
            variable_names_by_id={self.rdr.case_variable._id_: variable_name},
            referenced_types={case_type},
        )
        emitter = RuleTreeSourceEmitter(
            context=context,
            conclusion_target=ast_attribute_access(
                ast_load_name(variable_name), self.rdr.conclusion_attribute_name
            ),
        )
        rule_tree = DecomposedRuleTree.from_condition(
            condition=self.rdr.query._conditions_root_
        )
        base_condition = ast_unparse_expression(
            ConditionSourceEmitter.emit_source(
                subject=rule_tree.base_condition, context=context
            )
        )
        body = indent(
            ast_unparse_statements(emitter.emit_rule_body(rule_tree=rule_tree)),
            "    ",
        )
        corner_cases = self._render_corner_cases(
            rule_tree=rule_tree, referenced_types=context.referenced_types
        )
        type_imports = self._render_type_imports(
            referenced_types=context.referenced_types,
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
