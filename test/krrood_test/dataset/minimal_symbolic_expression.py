from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Iterator

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
)


@dataclass(eq=False)
class MinimalSymbolicExpression(SymbolicExpression):
    """A real :class:`SymbolicExpression` with no children, carrying the inherited ``_id_``.

    It stands in for any identified query node in tests that only exercise behaviour keyed on a
    node's identity (the fold's binding-override lookup, the discourse model's scope keys), without
    pulling in a concrete grammar construct.
    """

    def _evaluate__(self, sources: OperationResult) -> Iterator[OperationResult]:
        """Never evaluated by identity-only tests."""
        return iter(())

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ) -> None:
        """No child fields to rewire on a leaf stand-in."""

    @property
    def _name_(self) -> str:
        return type(self).__name__
