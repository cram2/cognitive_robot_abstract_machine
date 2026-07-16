from __future__ import annotations

from typing_extensions import Type, Optional, TYPE_CHECKING

from krrood.entity_query_language.cache_data import InstanceFilteredDomain
from krrood.entity_query_language.utils import T, is_iterable
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

if TYPE_CHECKING:
    from krrood.entity_query_language.core.variable import DomainType


def _resolve_domain(type_: Type[T], domain: Optional[DomainType]) -> Optional[DomainType]:
    """
    Resolve a variable's domain: what :func:`variable` and
    :meth:`~krrood.entity_query_language.query.match.Match.create_or_update_variable`
    both range a variable's values over.

    :param type_: The type of the variable the domain is for.
    :param domain: Iterable of potential values for the variable, or None. If None, the
        domain is inferred from the SymbolGraph for Symbol types, else should not be
        evaluated by EQL but by another evaluator (e.g., EQL To SQL converter in
        Ormatic).
    :return: The given ``domain`` filtered to instances of ``type_``. For a Symbol type
        with no domain given, the ``SymbolGraph``'s instances of ``type_`` instead.
        Otherwise ``domain`` unchanged.
    """
    if domain is not None and is_iterable(domain):
        return InstanceFilteredDomain(type_, domain)
    if domain is None and issubclass(type_, Symbol):
        return SymbolGraph().get_instances_of_type(type_)
    return domain
