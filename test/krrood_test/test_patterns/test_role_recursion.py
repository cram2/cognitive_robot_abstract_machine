import pytest

from krrood.patterns.role import Role
from ..dataset.role_and_ontology.classes_for_testing_role_recursion_error import (
    PersonForRoleRecursion,
    StudentForRoleRecursion,
    TeacherForRoleRecursion,
    BaseForRoleRecursion,
    IntermediateForRoleRecursion,
    TopForRoleRecursion,
)


def test_role_attribute_resolution():
    p = PersonForRoleRecursion(name="John")
    s = StudentForRoleRecursion(student_id="S123", role_taker=p)
    t = TeacherForRoleRecursion(employee_id="T456", role_taker=p)

    # Taker attr accessible from role via __getattr__ delegation.
    assert s.name == "John"
    assert t.name == "John"

    # Role-native attrs are accessed directly from the role.
    assert s.student_id == "S123"
    assert t.employee_id == "T456"

    # Sibling role attrs are accessed via the shared taker's roles dict.
    assert Role.roles_for(p, TeacherForRoleRecursion)[0].employee_id == "T456"
    assert Role.roles_for(p, StudentForRoleRecursion)[0].student_id == "S123"

    # Non-existent attribute should raise AttributeError, not RecursionError.
    with pytest.raises(AttributeError):
        s.non_existent_attr


def test_role_recursion_with_chained_roles():
    b = BaseForRoleRecursion()
    i = IntermediateForRoleRecursion(role_taker=b)
    top = TopForRoleRecursion(role_taker=i)

    # Role-native attrs on each role directly.
    assert top.top_attr == "top"
    assert i.inter_attr == "inter"
    assert b.base_attr == "base"

    # Taker attrs accessible from role via __getattr__ delegation.
    assert top.inter_attr == "inter"
    assert i.base_attr == "base"
    assert top.base_attr == "base"

    # Roles dict tracks each role in the chain.
    assert Role.roles_for(i, TopForRoleRecursion)[0] is top
    assert Role.roles_for(b, IntermediateForRoleRecursion)[0] is i

    with pytest.raises(AttributeError):
        top.none
