"""
Emitters that build the :mod:`ast` expression rebuilding an EQL condition, or a rule-
tree branch, in the syntax conditions and rule trees are authored in by hand.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from typing_extensions import Any, Callable, TypeVar

from krrood.code_generation.ast_helpers import (
    ast_attribute_access,
    ast_call,
    ast_compare,
    ast_comparison_operator_class,
    ast_load_name,
)
from krrood.code_generation.source_emission import (
    SourceEmissionContext,
    SourceEmitter,
    ValueSourceEmitter,
)
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.factories import (
    alternative,
    and_,
    next_rule,
    not_,
    or_,
    refinement,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    LogicalBinaryOperator,
    Not,
)
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    ConclusionSelector,
    Next,
    Refinement,
)
from krrood.entity_query_language.rules.decomposition import ConclusionSelectorBranch
from krrood.ormatic.utils import classproperty

NodeType = TypeVar("NodeType", bound=SymbolicExpression)

ConnectiveType = TypeVar("ConnectiveType", bound=LogicalBinaryOperator)

ConclusionSelectorType = TypeVar("ConclusionSelectorType", bound=ConclusionSelector)

# %% emission context


@dataclass(kw_only=True)
class ConditionEmissionContext(SourceEmissionContext):
    """
    What the emitters of one generated condition source share.
    """

    variable_names_by_id: dict[UUID, str]
    """
    The source name of each bound variable, keyed by the variable's id.
    """


# %% conditions


@dataclass(kw_only=True)
class ConditionSourceEmitter(SourceEmitter[NodeType], ABC):
    """
    The family of emitters for the nodes of a condition.
    """

    context: ConditionEmissionContext
    """
    The state shared by every emitter of the generated source.
    """


@dataclass(kw_only=True)
class LiteralSourceEmitter(ConditionSourceEmitter[Literal]):
    """
    Emits a literal as the value it holds.
    """

    def emit(self, *, subject: Literal) -> ast.expr:
        return ValueSourceEmitter.emit_source(
            subject=subject._value_, context=self.context
        )


@dataclass(kw_only=True)
class VariableSourceEmitter(ConditionSourceEmitter[Variable]):
    """
    Emits a bound variable as its source name.
    """

    @classmethod
    def can_emit_source_for_object(
        cls, *, subject: Any, context: ConditionEmissionContext
    ) -> bool:
        """
        Like :meth:`SourceEmitter.can_emit_source_for_object`, and only for a variable
        :attr:`ConditionEmissionContext.variable_names_by_id` names.
        """
        return (
            super().can_emit_source_for_object(subject=subject, context=context)
            and subject._id_ in context.variable_names_by_id
        )

    def emit(self, *, subject: Variable) -> ast.expr:
        return ast_load_name(self.context.variable_names_by_id[subject._id_])


@dataclass(kw_only=True)
class AttributeSourceEmitter(ConditionSourceEmitter[Attribute]):
    """
    Emits an attribute access on the emitted expression it reads from.
    """

    def emit(self, *, subject: Attribute) -> ast.expr:
        return ast_attribute_access(
            self.emit_source(subject=subject._child_, context=self.context),
            subject._attribute_name_,
        )


@dataclass(kw_only=True)
class ComparisonSourceEmitter(ConditionSourceEmitter[Comparator]):
    """
    Emits a comparator as a comparison of its emitted operands.
    """

    @classmethod
    def can_emit_source_for_object(
        cls, *, subject: Any, context: ConditionEmissionContext
    ) -> bool:
        """
        Like :meth:`SourceEmitter.can_emit_source_for_object`, and only for a comparator
        whose operation has a Python comparison operator.
        """
        return (
            super().can_emit_source_for_object(subject=subject, context=context)
            and ast_comparison_operator_class(subject.operation) is not None
        )

    def emit(self, *, subject: Comparator) -> ast.expr:
        return ast_compare(
            self.emit_source(subject=subject.left, context=self.context),
            ast_comparison_operator_class(subject.operation),
            self.emit_source(subject=subject.right, context=self.context),
        )


@dataclass(kw_only=True)
class FactoryCallSourceEmitter(ConditionSourceEmitter[NodeType], ABC):
    """
    Emits a node as a call of the factory that builds it on its emitted operands.
    """

    @classproperty
    @abstractmethod
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        The factory that builds a node of :attr:`emitted_type`.
        """

    @abstractmethod
    def factory_operands(self, *, subject: NodeType) -> list[SymbolicExpression]:
        """
        The operands :attr:`factory` is called on to rebuild ``subject``.

        :param subject: The node to rebuild.
        :return: The operands, in the order the factory takes them.
        """

    def emit(self, *, subject: NodeType) -> ast.expr:
        return ast_call(
            self.factory.__name__,
            [
                self.emit_source(subject=operand, context=self.context)
                for operand in self.factory_operands(subject=subject)
            ],
        )


@dataclass(kw_only=True)
class ConnectiveSourceEmitter(FactoryCallSourceEmitter[ConnectiveType], ABC):
    """
    Emits a binary logical connective as a factory call on its two operands.
    """

    def factory_operands(self, *, subject: ConnectiveType) -> list[SymbolicExpression]:
        return [subject.left, subject.right]


@dataclass(kw_only=True)
class ConjunctionSourceEmitter(ConnectiveSourceEmitter[AND]):
    """
    Emits a conjunction as an ``and_(...)`` call.
    """

    @classproperty
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        :func:`~krrood.entity_query_language.factories.and_`.
        """
        return and_


@dataclass(kw_only=True)
class DisjunctionSourceEmitter(ConnectiveSourceEmitter[OR]):
    """
    Emits a disjunction as an ``or_(...)`` call.
    """

    @classproperty
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        :func:`~krrood.entity_query_language.factories.or_`.
        """
        return or_


@dataclass(kw_only=True)
class NegationSourceEmitter(FactoryCallSourceEmitter[Not]):
    """
    Emits a negation as a ``not_(...)`` call.
    """

    @classproperty
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        :func:`~krrood.entity_query_language.factories.not_`.
        """
        return not_

    def factory_operands(self, *, subject: Not) -> list[SymbolicExpression]:
        return [subject._child_]


# %% rule-tree branches


@dataclass(kw_only=True)
class BranchSourceEmitter(SourceEmitter[ConclusionSelectorType], ABC):
    """
    The family of emitters for the factory call that opens a rule-tree branch, such as
    ``refinement(condition)``, for the conclusion-selector type that added the branch.
    """

    context: ConditionEmissionContext
    """
    The state shared by every emitter of the generated source.
    """

    @classproperty
    @abstractmethod
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        The factory that adds a branch of :attr:`emitted_type`.
        """

    @classmethod
    def can_emit_source_for_object(
        cls, *, subject: Any, context: SourceEmissionContext
    ) -> bool:
        """
        Whether ``subject`` is a branch added by a conclusion selector of
        :attr:`emitted_type`.
        """
        return isinstance(subject, ConclusionSelectorBranch) and issubclass(
            subject.conclusion_selector_type, cls.emitted_type
        )

    def emit(self, *, subject: ConclusionSelectorBranch) -> ast.expr:
        return ast_call(
            self.factory.__name__,
            [
                ConditionSourceEmitter.emit_source(
                    subject=subject.subtree.base_condition, context=self.context
                )
            ],
        )


@dataclass(kw_only=True)
class RefinementBranchSourceEmitter(BranchSourceEmitter[Refinement]):
    """
    Emits a refinement branch as a ``refinement(...)`` call.
    """

    @classproperty
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        :func:`~krrood.entity_query_language.factories.refinement`.
        """
        return refinement


@dataclass(kw_only=True)
class AlternativeBranchSourceEmitter(BranchSourceEmitter[Alternative]):
    """
    Emits an alternative branch as an ``alternative(...)`` call.
    """

    @classproperty
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        :func:`~krrood.entity_query_language.factories.alternative`.
        """
        return alternative


@dataclass(kw_only=True)
class NextBranchSourceEmitter(BranchSourceEmitter[Next]):
    """
    Emits a next-rule branch as a ``next_rule(...)`` call.
    """

    @classproperty
    def factory(cls) -> Callable[..., SymbolicExpression]:
        """
        :func:`~krrood.entity_query_language.factories.next_rule`.
        """
        return next_rule
