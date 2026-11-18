"""
Decomposition of a rule tree, a left-nested chain of :class:`ConclusionSelector` nodes,
into the branches it was built from.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property

from typing_extensions import Self

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.rules.conclusion_selector import ConclusionSelector

# %% branch


@dataclass(kw_only=True)
class ConclusionSelectorBranch:
    """
    One branch of a rule tree: the conclusion-selector type that added it and the
    condition sub-tree it holds.
    """

    conclusion_selector_type: type[ConclusionSelector]
    """
    The conclusion-selector class that added this branch.
    """

    condition: SymbolicExpression
    """
    The branch's condition sub-tree, itself possibly wrapped in conclusion selectors.
    """

    @cached_property
    def subtree(self) -> DecomposedRuleTree:
        """
        The rule tree this branch opens, decomposed.
        """
        return DecomposedRuleTree.from_condition(condition=self.condition)


# %% decomposed rule tree


@dataclass(kw_only=True)
class DecomposedRuleTree:
    """
    A rule tree split into the base condition it is anchored on and its branches, in the
    order they have to be added again to rebuild the identical tree.
    """

    base_condition: SymbolicExpression
    """
    The condition no conclusion selector wraps.
    """

    branches: list[ConclusionSelectorBranch]
    """
    The branches, in the order they have to be added again.
    """

    @classmethod
    def from_condition(cls, *, condition: SymbolicExpression) -> Self:
        """
        Decompose the rule tree rooted at ``condition``.

        :param condition: The root of the rule tree, possibly a bare condition.
        :return: The decomposed rule tree.
        """
        walked_branches = []
        node = condition
        while isinstance(node, ConclusionSelector):
            walked_branches.append(
                ConclusionSelectorBranch(
                    conclusion_selector_type=type(node), condition=node.right
                )
            )
            node = node.left
        return cls(
            base_condition=node,
            branches=cls._order_branches_for_reinsertion(
                walked_branches=walked_branches
            ),
        )

    @classmethod
    def _order_branches_for_reinsertion(
        cls, *, walked_branches: list[ConclusionSelectorBranch]
    ) -> list[ConclusionSelectorBranch]:
        """Order the branches as walked from the root into the order they were
        added: the walk meets inward-growing branches in that order and
        outward-growing ones in reverse, so each run of same-direction
        branches is ordered on its own.

        :param walked_branches: The branches in the order the walk from
            the root met them.
        :return: The branches in the order they have to be added again.
        """
        branches = []
        for grows_rule_tree_inward, run in itertools.groupby(
            walked_branches,
            key=lambda branch: branch.conclusion_selector_type.grows_rule_tree_inward,
        ):
            run = list(run)
            branches.extend(run if grows_rule_tree_inward else reversed(run))
        return branches

    def rule_conditions(self) -> list[SymbolicExpression]:
        """
        The condition of every rule in this tree, in pre-order: the base condition, then
        the rules of each branch in the order the branches have to be added again.
        """
        conditions = [self.base_condition]
        for branch in self.branches:
            conditions.extend(branch.subtree.rule_conditions())
        return conditions
