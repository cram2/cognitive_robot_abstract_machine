from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

TSymbolicExpression = TypeVar("TSymbolicExpression", bound=SymbolicExpression)
"""The EQL node type the assembler realises."""

TPlan = TypeVar("TPlan")
"""The plan (data record) the assembler realises."""


@dataclass
class Planner(Generic[TSymbolicExpression, TPlan], SubClassSafeGeneric):
    """
    Pure analysis of a single EQL *node* into a plan of type ``P``.

    A planner inspects an EQL node and produces a *plan*: a plain, frozen data record of the
    decisions about *what to say*. It does not build fragments, touch the verbalization context,
    or recurse — the content/structure stage of the Reiter & Dale microplanning model.

    Reference: :cite:t:`reiter2000building` — microplanning vs. surface realisation.
    """

    node: TSymbolicExpression
    """The EQL expression being analysed."""

    @abstractmethod
    def plan(self) -> TPlan:
        """:return: The plan computed from the node."""
