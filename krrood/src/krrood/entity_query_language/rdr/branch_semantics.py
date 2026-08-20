"""
Per-selector branch semantics for backward inference over EQL-RDR rule trees.

Each :class:`~krrood.entity_query_language.rules.conclusion_selector.ConclusionSelector`
answers two questions when a rule tree is traversed backwards, and this module holds one
class per selector answering both:

* *As a competing sibling branch, what leaf predicates capture whether that branch was
  taken?* — :meth:`SelectorBranchSemantics.sibling_guards`.
* *Descending into this selector, which children continue the walk and what does entering
  each one add to the accumulated guards?* — :meth:`SelectorBranchSemantics.branches`.

Keeping both on one class per selector is what makes them impossible to change apart, and
lets a new selector participate by defining a class here rather than by editing the
traversal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import (
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.rdr.exceptions import AmbiguousBranchSemanticsError
from krrood.entity_query_language.rdr.guard_condition import GuardCondition
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    ConclusionSelector,
    Next,
    Refinement,
)
from krrood.patterns.specificity_ranking import (
    concrete_subclasses,
    mro_depth,
    sole_maximum,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

SelectorType = TypeVar("SelectorType", bound=ConclusionSelector)
"""
The conclusion-selector class a branch-semantics class handles.
"""

LeafGuardDecomposition = Callable[[SymbolicExpression, bool], List[GuardCondition]]
"""
Decompose an expression into leaf guards at a given polarity.
"""

# %% branch value object


@dataclass(frozen=True)
class GuardedBranch:
    """
    One child of a conclusion selector, paired with what entering it implies.
    """

    child_expression: SymbolicExpression
    """
    The child expression the traversal continues into.
    """

    entry_guards: Tuple[GuardCondition, ...]
    """
    The guards that reaching :attr:`child_expression` through this selector adds to the
    path.
    """


# %% the family


@dataclass
class SelectorBranchSemantics(Generic[SelectorType], SubClassSafeGeneric, ABC):
    """
    One conclusion selector's branch-choice semantics, as seen by backward inference.

    A concrete subclass binds :data:`SelectorType` to the selector it handles and
    implements both halves; it is discovered automatically, so adding a selector needs no
    edit to the traversal.

    ..note:: Every method is a classmethod — the semantics carry no state, so the
        traversal dispatches to the class itself and never constructs one.
    """

    @classmethod
    def selector_type(cls) -> Type[ConclusionSelector]:
        """:return: The conclusion-selector class this semantics handles."""
        return cls.get_generic_type_parameters()[0]

    @classmethod
    def most_specific_for(
        cls,
        expression: SymbolicExpression,
    ) -> Optional[Type[SelectorBranchSemantics]]:
        """
        Find the semantics governing *expression*.

        Ranks by the specificity of each candidate's :meth:`selector_type`, so a
        semantics for a subclass of some selector outranks the one it refines.

        :param expression: Any rule-tree node.
        :return: The matching semantics class, or ``None`` when *expression* is not a
            conclusion selector.
        :raises AmbiguousBranchSemanticsError: Two candidates are equally specific.
        """
        applicable = [
            candidate
            for candidate in concrete_subclasses(cls)
            if isinstance(expression, candidate.selector_type())
        ]
        return sole_maximum(
            applicable,
            key=lambda candidate: mro_depth(candidate.selector_type()),
            collision_error=lambda tied: AmbiguousBranchSemanticsError(
                selector=expression, candidates=tied
            ),
        )

    @classmethod
    @abstractmethod
    def sibling_guards(
        cls,
        selector: ConclusionSelector,
        negated: bool,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardCondition]:
        """
        Decompose *selector*, standing as a competing sibling branch, into leaf guards.

        The result is always leaf-level, never a selector, so guards stay readable and
        directly evaluable.

        ..note:: The returned guards are conjoined by the caller, so a selector whose
            positive reading is a disjunction cannot be represented exactly; see
            :class:`AlternativeBranchSemantics`.

        :param selector: The selector node to decompose.
        :param negated: Whether the guard polarity is negated.
        :param decompose: The recursion continuation for child expressions.
        :return: The flat list of leaf guards.
        """

    @classmethod
    @abstractmethod
    def branches(
        cls,
        selector: ConclusionSelector,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardedBranch]:
        """
        Enumerate the children the traversal continues into, with their entry guards.

        :param selector: The selector node being descended into.
        :param decompose: The recursion continuation for child expressions.
        :return: One :class:`GuardedBranch` per child, in traversal order.
        """


# %% concrete selectors


@dataclass
class RefinementBranchSemantics(SelectorBranchSemantics[Refinement]):
    """
    ``Refinement(left, right)`` — *right* refines *left*, overriding it when it applies.

    As a sibling, the refinement branch was taken exactly when *left* passed; *right* is
    a separate rule subtree rather than a condition on *left* having been reached.
    """

    @classmethod
    def sibling_guards(
        cls,
        selector: Refinement,
        negated: bool,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardCondition]:
        return decompose(selector.left, negated)

    @classmethod
    def branches(
        cls,
        selector: Refinement,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardedBranch]:
        return [
            GuardedBranch(selector.left, tuple(decompose(selector.right, True))),
            GuardedBranch(selector.right, tuple(decompose(selector.left, False))),
        ]


@dataclass
class AlternativeBranchSemantics(SelectorBranchSemantics[Alternative]):
    """
    ``Alternative(left, right)`` — an "else if": *right* applies only when *left* did
    not.

    Negated, the branch was not taken when neither side passed, which De Morgan turns into
    the conjunction ``NOT(left) AND NOT(right)`` the caller expects.

    ..warning:: Positively, the branch was taken when ``left OR right`` passed, which the
        caller's conjunction cannot express. The traversal never asks for that reading — a
        positive decomposition is only requested for a ``Refinement``'s left child, and an
        ``Alternative`` is always spliced above the conditions root rather than under a
        refinement — so the two sides are returned unchanged rather than approximated.
    """

    @classmethod
    def sibling_guards(
        cls,
        selector: Alternative,
        negated: bool,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardCondition]:
        return decompose(selector.left, negated) + decompose(selector.right, negated)

    @classmethod
    def branches(
        cls,
        selector: Alternative,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardedBranch]:
        return [
            GuardedBranch(selector.left, ()),
            GuardedBranch(selector.right, tuple(decompose(selector.left, True))),
        ]


@dataclass
class NextBranchSemantics(SelectorBranchSemantics[Next]):
    """
    ``Next(...)`` — independent rules at the same depth, evaluated without cross-guards.
    """

    @classmethod
    def sibling_guards(
        cls,
        selector: Next,
        negated: bool,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardCondition]:
        guards: List[GuardCondition] = []
        for child in selector._operation_children_:
            guards.extend(decompose(child, negated))
        return guards

    @classmethod
    def branches(
        cls,
        selector: Next,
        decompose: LeafGuardDecomposition,
    ) -> List[GuardedBranch]:
        return [GuardedBranch(child, ()) for child in selector._operation_children_]
