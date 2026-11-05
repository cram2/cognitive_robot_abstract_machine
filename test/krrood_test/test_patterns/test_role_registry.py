from __future__ import annotations

import gc
from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role
from krrood.patterns.role_registry import RoleRegistry


@dataclass(kw_only=True, eq=False)
class PersistentEntityWithValueEquality(Symbol):
    name: str

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PersistentEntityWithValueEquality)
            and other.name == self.name
        )

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(kw_only=True, eq=False)
class RoleOverEntity(Role[PersistentEntityWithValueEquality]): ...


@dataclass(kw_only=True, eq=False)
class RoleOverRole(Role[RoleOverEntity]): ...


def test_a_registered_role_is_retrievable_by_its_taker():
    registry = RoleRegistry()
    entity = PersistentEntityWithValueEquality(name="entity")
    role = RoleOverEntity(role_taker=entity)

    registry.register(role=role)

    assert role in set(registry.roles_of(role_taker=entity))


def test_a_role_is_retrievable_by_every_taker_in_its_chain():
    registry = RoleRegistry()
    entity = PersistentEntityWithValueEquality(name="entity")
    inner = RoleOverEntity(role_taker=entity)
    outer = RoleOverRole(role_taker=inner)

    registry.register(role=outer)

    assert outer in set(registry.roles_of(role_taker=inner))
    assert outer in set(registry.roles_of(role_taker=entity))


def test_roles_are_keyed_by_taker_identity_not_equality():
    registry = RoleRegistry()
    first_entity = PersistentEntityWithValueEquality(name="same")
    second_entity = PersistentEntityWithValueEquality(name="same")
    assert first_entity == second_entity and first_entity is not second_entity

    role = RoleOverEntity(role_taker=first_entity)
    registry.register(role=role)

    assert role in set(registry.roles_of(role_taker=first_entity))
    assert role not in set(registry.roles_of(role_taker=second_entity))


def test_dead_roles_are_pruned_from_the_index():
    registry = RoleRegistry()
    entity = PersistentEntityWithValueEquality(name="entity")
    role = RoleOverEntity(role_taker=entity)
    registry.register(role=role)
    assert list(registry.roles_of(role_taker=entity))

    del role
    gc.collect()

    assert list(registry.roles_of(role_taker=entity)) == []


def test_a_fresh_registry_shares_no_state_with_the_role_default():
    isolated_registry = RoleRegistry()
    entity = PersistentEntityWithValueEquality(name="entity")

    RoleOverEntity(role_taker=entity)

    assert list(isolated_registry.roles_of(role_taker=entity)) == []
