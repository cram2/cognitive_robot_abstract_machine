from dataclasses import dataclass, field, fields
import pytest
from krrood.patterns.role import Role
from krrood.entity_query_language.factories import variable_from
from typing import TypeVar

T = TypeVar("T")


@dataclass
class BaseEntity:
    root: str = "default_root"


@dataclass
class RoleTaker(BaseEntity):
    name: str = "taker"


@dataclass
class MyRole(Role[RoleTaker], RoleTaker):
    role_taker_attr: RoleTaker = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls):
        return variable_from(cls).role_taker_attr


def test_reproduce_attribute_error_on_uninitialized_role():
    # check if 'root' is indeed a property
    assert isinstance(MyRole.root, property)

    instance = MyRole.__new__(MyRole)
    with pytest.raises(AttributeError):
        val = instance.root
    instance = MyRole(role_taker_attr=RoleTaker())
    assert instance.root == "default_root"
    assert "root" in [f.name for f in fields(MyRole)]
    assert "root" in [f.name for f in fields(BaseEntity)]
    root_field = next(f for f in fields(MyRole) if f.name == "root")
    root_field_in_base = next(f for f in fields(BaseEntity) if f.name == "root")
    assert root_field.init is True
    assert root_field_in_base.init is True
