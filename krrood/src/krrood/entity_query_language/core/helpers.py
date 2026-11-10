from __future__ import annotations

from typing_extensions import Any, Iterable

from krrood.entity_query_language.cache_data import InstanceFilteredDomain
from krrood.entity_query_language.core.variable import InstancesOfType, Literal
from krrood.entity_query_language.utils import T, is_iterable
from krrood.symbol_graph.symbol_graph import Symbol


def unwrap_if_literal(value: Any) -> Any:
    """
    Unwrap a :class:`~krrood.entity_query_language.core.variable.Literal` wrapper to get
    the raw value; a value that is not a :class:`Literal` is returned as it is.
    """
    return value._value_ if isinstance(value, Literal) else value


def _resolve_domain(type_: type[T], domain: Iterable[T] | None) -> Iterable[T] | None:
    """
    Resolve a variable's domain: what :func:`variable` and
    :meth:`~krrood.entity_query_language.query.match.Match._create_or_update_variable_`
    both range a variable's values over.

    :param type_: The type of the variable the domain is for.
    :param domain: Iterable of potential values for the variable, or None. If None, the
        domain is every instance of a Symbol type in the evaluation, else should not be
        evaluated by EQL but by another evaluator (e.g., EQL To SQL converter in
        Ormatic).
    :return: The given ``domain`` filtered to instances of ``type_``. For a Symbol type
        with no domain given, every instance of ``type_`` in the evaluation instead.
        Otherwise ``domain`` unchanged.
    """
    if domain is not None and is_iterable(domain):
        return InstanceFilteredDomain(type_, domain)
    if domain is None and issubclass(type_, Symbol):
        return InstancesOfType(type_=type_)
    return domain
