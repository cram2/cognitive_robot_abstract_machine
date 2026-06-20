from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import ClassVar

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role, role_taker_field
from krrood.symbol_graph.symbol_graph import SymbolGraph


@dataclass(eq=False)
class PersistentEntityForCaching(Symbol):
    name: str

    def __hash__(self) -> int:
        return id(self)


@dataclass(eq=False)
class RoleWithDefaultCaching(Role[PersistentEntityForCaching]):
    entity: PersistentEntityForCaching = role_taker_field()


@dataclass(eq=False)
class RoleOptingIntoInstanceCaching(Role[PersistentEntityForCaching]):
    _cache_instances_: ClassVar[bool] = True

    entity: PersistentEntityForCaching = role_taker_field()


def test_a_role_is_not_cached_as_a_symbol_graph_instance_by_default():
    entity = PersistentEntityForCaching(name="entity")
    role = RoleWithDefaultCaching(entity=entity)

    assert SymbolGraph().get_wrapped_instance(role) is None
    assert role not in set(SymbolGraph().get_instances_of_type(RoleWithDefaultCaching))


def test_a_subclass_can_opt_into_instance_caching():
    entity = PersistentEntityForCaching(name="entity")
    role = RoleOptingIntoInstanceCaching(entity=entity)

    assert SymbolGraph().get_wrapped_instance(role) is not None
    assert role in set(
        SymbolGraph().get_instances_of_type(RoleOptingIntoInstanceCaching)
    )


def test_membership_queries_work_regardless_of_instance_caching():
    entity = PersistentEntityForCaching(name="entity")
    uncached_role = RoleWithDefaultCaching(entity=entity)
    cached_role = RoleOptingIntoInstanceCaching(entity=entity)

    assert Role.has_role(entity, RoleWithDefaultCaching)
    assert Role.has_role(entity, RoleOptingIntoInstanceCaching)
    assert uncached_role in Role.roles_for(entity, RoleWithDefaultCaching)
    assert cached_role in Role.roles_for(entity, RoleOptingIntoInstanceCaching)
