from dataclasses import dataclass

import pytest

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.exceptions import RoleAttributeNotDeclaredError
from krrood.patterns.role import Role, role_taker_field


@dataclass(eq=False)
class PersistentEntityWithName(Symbol):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(eq=False)
class RoleWithOwnField(Role[PersistentEntityWithName]):
    entity: PersistentEntityWithName = role_taker_field()
    own_field: str = ""


def test_reading_a_taker_attribute_through_a_role_delegates_to_the_taker():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    assert role.name == "original"


def test_assigning_a_role_native_field_sets_it_on_the_role():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    role.own_field = "value"

    assert role.own_field == "value"
    assert not hasattr(entity, "own_field")


def test_assigning_an_undeclared_name_raises_and_leaves_the_taker_unchanged():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    with pytest.raises(RoleAttributeNotDeclaredError):
        role.name = "shadow"

    assert entity.name == "original"


def test_modifying_the_taker_requires_going_through_role_taker():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    role.role_taker.name = "changed"

    assert entity.name == "changed"
    assert role.name == "changed"


def test_private_attributes_can_be_set_on_the_role():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    role._scratch = 1

    assert role._scratch == 1
