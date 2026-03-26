from dataclasses import dataclass

import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role
from ..dataset.role_and_ontology.classes_for_testing_role_recursion_error import PersonForRoleRecursion, \
    StudentForRoleRecursion, TeacherForRoleRecursion, BaseForRoleRecursion, IntermediateForRoleRecursion, \
    TopForRoleRecursion


def test_role_attribute_resolution():
    diagram = ClassDiagram([PersonForRoleRecursion, StudentForRoleRecursion, TeacherForRoleRecursion])

    p = PersonForRoleRecursion(name="John")
    s = StudentForRoleRecursion(student_id="S123", person=p)
    t = TeacherForRoleRecursion(employee_id="T456", person=p)

    # Access attribute from role taker
    assert s.name == "John"
    assert t.name == "John"

    # Access attribute from sibling role
    assert s.employee_id == "T456"
    assert t.student_id == "S123"

    # Non-existent attribute should raise AttributeError, not RecursionError
    with pytest.raises(AttributeError):
        s.non_existent_attr


def test_role_recursion_with_chained_roles():
    diagram = ClassDiagram([BaseForRoleRecursion, IntermediateForRoleRecursion, TopForRoleRecursion])

    b = BaseForRoleRecursion()
    i = IntermediateForRoleRecursion(base=b)
    top = TopForRoleRecursion(inter=i)

    assert top.top_attr == "top"
    assert top.inter_attr == "inter"
    assert top.base_attr == "base"

    with pytest.raises(AttributeError):
        top.none
