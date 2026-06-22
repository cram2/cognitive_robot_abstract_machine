from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.core.expression_structure import (
    chain_ends_in_boolean_attribute,
    walk_chain,
)
from krrood.entity_query_language.verbalization.navigation_path import (
    PathStep,
    build_path_parts,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    relational_verb,
)
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner


@dataclass(frozen=True)
class ChainPlan:
    """
    A ``MappedVariable`` chain analysed once into the values its rendering needs: the walked
    chain, its root, the display path-parts, and whether it ends in a boolean attribute.
    """

    chain: List[MappedVariable]
    """The access path, root-adjacent first."""

    root: SymbolicExpression
    """The chain root (first non-``MappedVariable`` node)."""

    parts: List[PathStep]
    """The display path-parts."""

    is_boolean_terminal: bool
    """``True`` when the chain ends in a ``bool``-typed attribute (predicative form)."""

    @property
    def is_single_variable_attribute(self) -> bool:
        """:return: ``True`` when the chain is a single attribute on a variable (the bare-plural
        *"attributes of Roots"* form)."""
        return (
            isinstance(self.root, Variable)
            and len(self.chain) == 1
            and isinstance(self.chain[0], Attribute)
        )

    def renders_as_plural_attribute(self, number: Number) -> bool:
        """
        The chain precedence policy in one place: a single attribute on a variable, asked for in
        the plural, renders as the bare-plural noun phrase *"attributes of Roots"* — which takes
        precedence over the predicative form a boolean terminal would otherwise produce.

        :param number: The grammatical number requested for the chain.
        :return: ``True`` when the chain renders as the bare-plural attribute form.
        """
        return number is Number.PLURAL and self.is_single_variable_attribute


@dataclass
class ChainPlanner(Planner[MappedVariable, ChainPlan]):
    """
    Analyse a ``MappedVariable`` chain into a ``ChainPlan``: its root, the display path-parts, and
    whether it ends in a boolean attribute (predicative form) — the chain decisions of *what to
    say*, before any surface form is chosen.

    Reference: Reiter & Dale (2000) — content/structure determination (microplanning).

    >>> ChainPlanner(variable(Task, []).completed).plan().is_boolean_terminal
    True
    """

    def plan(self) -> ChainPlan:
        """:return: The chain plan: the walked chain, its root, display parts, and boolean form."""
        chain, root = walk_chain(self.node)
        return ChainPlan(
            chain=chain,
            root=root,
            parts=build_path_parts(chain, relational_verb),
            is_boolean_terminal=chain_ends_in_boolean_attribute(chain),
        )
