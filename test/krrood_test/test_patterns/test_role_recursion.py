from dataclasses import dataclass

import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role


@dataclass
class Person:
    name: str


@dataclass(eq=False)
class Student(Role[Person]):
    student_id: str
    _person: Person

    @classmethod
    def role_taker_attribute(cls) -> Person:
        return variable_from(cls)._person


@dataclass(eq=False)
class Teacher(Role[Person]):
    employee_id: str
    _person: Person

    @classmethod
    def role_taker_attribute(cls) -> Person:
        return variable_from(cls)._person


def test_role_attribute_resolution():
    diagram = ClassDiagram([Person, Student, Teacher])

    p = Person(name="John")
    s = Student(student_id="S123", _person=p)
    t = Teacher(employee_id="T456", _person=p)

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
    @dataclass
    class Base:
        base_attr: str = "base"

    @dataclass(eq=False)
    class Intermediate(Role[Base]):
        _base: Base
        inter_attr: str = "inter"

        @classmethod
        def role_taker_attribute(cls) -> Base:
            return variable_from(cls)._base

    @dataclass(eq=False)
    class Top(Role[Intermediate]):
        _inter: Intermediate
        top_attr: str = "top"

        @classmethod
        def role_taker_attribute(cls) -> Intermediate:
            return variable_from(cls)._inter

    diagram = ClassDiagram([Base, Intermediate, Top])

    b = Base()
    i = Intermediate(_base=b)
    top = Top(_inter=i)

    assert top.top_attr == "top"
    assert top.inter_attr == "inter"
    assert top.base_attr == "base"

    with pytest.raises(AttributeError):
        top.none
