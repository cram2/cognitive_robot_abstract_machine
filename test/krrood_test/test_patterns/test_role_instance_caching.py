from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import ClassVar

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import SymbolGraph


@dataclass(kw_only=True, eq=False)
class PersistentEntityForCaching(Symbol):
    name: str

    def __hash__(self) -> int:
        return id(self)


@dataclass(kw_only=True, eq=False)
class RoleWithDefaultCaching(Role[PersistentEntityForCaching]): ...


@dataclass(kw_only=True, eq=False)
class RoleOptingIntoInstanceCaching(Role[PersistentEntityForCaching]):
    _cache_instances_: ClassVar[bool] = True


def test_a_role_is_not_cached_as_a_symbol_graph_instance_by_default():
    entity = PersistentEntityForCaching(name="entity")
    role = RoleWithDefaultCaching(role_taker=entity)

    assert SymbolGraph().get_wrapped_instance(instance=role) is None
    assert role not in set(SymbolGraph().get_instances_of_type(type_=RoleWithDefaultCaching))


def test_a_subclass_can_opt_into_instance_caching():
    entity = PersistentEntityForCaching(name="entity")
    role = RoleOptingIntoInstanceCaching(role_taker=entity)

    assert SymbolGraph().get_wrapped_instance(instance=role) is not None
    assert role in set(
        SymbolGraph().get_instances_of_type(type_=RoleOptingIntoInstanceCaching)
    )


def test_membership_queries_work_regardless_of_instance_caching():
    entity = PersistentEntityForCaching(name="entity")
    uncached_role = RoleWithDefaultCaching(role_taker=entity)
    cached_role = RoleOptingIntoInstanceCaching(role_taker=entity)

    assert Role.has_role(role_taker=entity, role_types=RoleWithDefaultCaching)
    assert Role.has_role(role_taker=entity, role_types=RoleOptingIntoInstanceCaching)
    assert uncached_role in Role.roles_for(role_taker=entity, role_type=RoleWithDefaultCaching)
    assert cached_role in Role.roles_for(role_taker=entity, role_type=RoleOptingIntoInstanceCaching)
