from dataclasses import is_dataclass

import pytest

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import (
    HasRoleTaker,
    AssociationThroughRoleTaker,
)
from krrood.class_diagrams.utils import classes_of_module
from ..dataset.role_and_ontology import university_ontology_like_classes
from ..dataset.role_and_ontology.university_ontology_like_classes_without_descriptors import (
    Person,
    CEOAsFirstRole,
    Company,
    ProfessorAsFirstRole,
    Course,
    RepresentativeAsSecondRole,
)


def test_getting_and_setting_attribute_for_role_and_role_taker():
    person = Person(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    ceo.head_of = Company(name="BassCo")

    assert ceo.person.name == person.name

    # access attribute of role-taker (Person) directly from a role (CEO)
    assert ceo.name == person.name

    # access attribute of a role (CEO) directly from a role-taker (Person)
    assert ceo.head_of is person.head_of
    assert ceo.person.head_of is ceo.head_of


def test_getting_and_setting_attribute_between_sibling_roles():
    person = Person(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    ceo.head_of = Company(name="BassCo")
    professor = ProfessorAsFirstRole(person=person)
    professor.teacher_of.append(Course(name="BassCourse"))

    assert professor.person is ceo.person
    assert professor.teacher_of[0].name == "BassCourse"

    # access attribute of sibling roles (CEO and Professor) directly from each other.
    assert professor.head_of.name == "BassCo"
    assert professor.head_of is ceo.head_of
    assert ceo.teacher_of[0].name == "BassCourse"
    assert ceo.teacher_of is professor.teacher_of
    assert person.teacher_of is professor.teacher_of


def test_accessing_attribute_of_role_from_role_taker_when_role_does_not_exist_and_the_attribute_has_default():
    person = Person(name="Bass")
    with pytest.raises(AttributeError):
        head_of = person.head_of
    assert hasattr(person, "head_of") is False


def test_accessing_attribute_of_role_from_role_taker_when_role_does_not_exist_and_the_attribute_has_default_factory():
    person = Person(name="Bass")
    with pytest.raises(AttributeError):
        teacher_of = person.teacher_of
    assert hasattr(person, "teacher_of") is False


def test_roles_are_equal_and_has_same_hash_as_each_other():
    person = Person(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    representative = RepresentativeAsSecondRole(ceo=ceo)
    professor = ProfessorAsFirstRole(person=person)
    assert hash(ceo) == hash(person)
    assert ceo == person
    assert ceo == representative
    assert ceo == professor


def test_role_taker_associations():

    classes = [
        cls
        for cls in classes_of_module(university_ontology_like_classes)
        if is_dataclass(cls)
    ]
    diagram = ClassDiagram(classes)
    assert len(diagram._dependency_graph.edges()) == 29
    assert (
        len(
            [
                e
                for e in diagram._dependency_graph.edges()
                if isinstance(e, HasRoleTaker)
            ]
        )
        == 3
    )
    assert len(diagram._dependency_graph.nodes()) == 14
    assert (
        len(
            [
                e
                for e in diagram._dependency_graph.edges()
                if isinstance(e, AssociationThroughRoleTaker)
            ]
        )
        == 9
    )
    # diagram.to_dot("class_diagram.svg")
