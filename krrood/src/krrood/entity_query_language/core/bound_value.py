"""
The bound-value contract shared by symbolic callables and read by the query evaluator.

Kept in its own leaf module (depending only on the standard library) so that both
:mod:`krrood.entity_query_language.core.variable` and
:mod:`krrood.entity_query_language.predicate` can import it without forming an import
cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Any


class HasBoundValue(ABC):
    """
    A type whose contribution to a query result is computed from keyword arguments.

    :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable` binds a callable class's
    value through :meth:`_bound_value_` — the constructed instance for a boolean operation, the
    constructed-and-called value for a value operation — rather than probing for the method by name. A
    plain function or type does not implement this contract, so the evaluator calls it directly instead.
    """

    @classmethod
    @abstractmethod
    def _bound_value_(cls, **kwargs: Any) -> Any:
        """:return: the value this operation contributes to a query result for the given argument
        values."""
