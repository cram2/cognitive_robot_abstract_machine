from __future__ import annotations

from dataclasses import is_dataclass, dataclass, field
from typing import Type

import pytest

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import (
    HasRoleTaker,
    AssociationThroughRoleTaker,
)
from krrood.class_diagrams.utils import classes_of_module, T
from krrood.patterns.role.role import Role
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from ..dataset.role_and_ontology import university_ontology_like_classes
from ..dataset.role_and_ontology.university_ontology_like_classes_without_descriptors import (
    PersonInRoleAndOntology,
    CEOAsFirstRole,
    Company,
    ProfessorAsFirstRole,
    Course,
    RepresentativeAsSecondRole, DelegateAsThirdRole, InDirectDiamondShapedInheritanceWhereOneIsRole,
    DirectDiamondShapedInheritanceWhereOneIsRole,
)


def test_getting_and_setting_attribute_for_role_and_role_taker():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    ceo.head_of = Company(name="BassCo")

    assert ceo.person.name == person.name

    # access attribute of role-taker (PersonInRoleAndOntology) directly from a role (CEO)
    assert ceo.name == person.name

    # access attribute of a role (CEO) directly from a role-taker (PersonInRoleAndOntology)
    assert ceo.head_of is person.head_of
    assert ceo.person.head_of is ceo.head_of


def test_getting_and_setting_attribute_between_sibling_roles():
    person = PersonInRoleAndOntology(name="Bass")
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
    person = PersonInRoleAndOntology(name="Bass")
    with pytest.raises(AttributeError):
        head_of = person.head_of
    assert hasattr(person, "head_of") is False


def test_accessing_attribute_of_role_from_role_taker_when_role_does_not_exist_and_the_attribute_has_default_factory():
    person = PersonInRoleAndOntology(name="Bass")
    with pytest.raises(AttributeError):
        teacher_of = person.teacher_of
    assert hasattr(person, "teacher_of") is False


def test_roles_are_equal_and_has_same_hash_as_each_other():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    representative = RepresentativeAsSecondRole(ceo=ceo)
    professor = ProfessorAsFirstRole(person=person)
    assert hash(ceo) == hash(person)
    assert ceo == person
    assert ceo == representative
    assert ceo == professor
    assert len({person, ceo, representative, professor}) == 1


def test_mappings_between_roles_and_role_takers():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    representative = RepresentativeAsSecondRole(ceo=ceo)
    professor = ProfessorAsFirstRole(person=person)
    delegate = DelegateAsThirdRole(representative=representative)
    roles = [ceo, representative, professor, delegate]

    for role_ in roles:
        assert all(role in role_.role_taker_roles for role in roles)

    delegate_role_takers = [representative, ceo, person]
    assert len(delegate_role_takers) == len(delegate.all_role_takers)
    assert all(EntityAndType(role_taker) in map(EntityAndType, delegate.all_role_takers) for role_taker in
               delegate_role_takers)

    professor_role_takers = [person]
    assert len(professor_role_takers) == len(professor.all_role_takers)
    assert all(EntityAndType(role_taker) in map(EntityAndType, professor.all_role_takers) for role_taker in
               professor_role_takers)


def test_has_role():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    representative = RepresentativeAsSecondRole(ceo=ceo)
    professor = ProfessorAsFirstRole(person=person)

    assert Role.has_role(person, CEOAsFirstRole)
    assert Role.has_role(person, RepresentativeAsSecondRole)
    assert not Role.has_role(person, DelegateAsThirdRole)
    assert Role.has_role(person, ProfessorAsFirstRole)


def test_get_roles_of_type():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    representative = RepresentativeAsSecondRole(ceo=ceo)
    professor = ProfessorAsFirstRole(person=person)

    assert isinstance(Role.get_taker_roles_of_type(person, CEOAsFirstRole)[0], CEOAsFirstRole)
    assert isinstance(Role.get_taker_roles_of_type(person, RepresentativeAsSecondRole)[0], RepresentativeAsSecondRole)
    assert Role.get_taker_roles_of_type(person, DelegateAsThirdRole) == []
    assert isinstance(Role.get_taker_roles_of_type(person, ProfessorAsFirstRole)[0], ProfessorAsFirstRole)
    assert Role.get_taker_roles_of_type(person, PersonInRoleAndOntology) == []


def test_role_that_inherits_from_class_that_role_taker_inherits_from_that_has_default_attributes():
    person = PersonInRoleAndOntology(name="Bass", default_name="BassDefualt")
    ceo = DirectDiamondShapedInheritanceWhereOneIsRole(person=person)
    assert ceo.name == person.name
    assert ceo.default_name == person.default_name


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


@dataclass
class EntityAndType(SubClassSafeGeneric[T]):
    entity: T
    type: Type[T] = field(init=False)

    def __post_init__(self):
        self.type = type(self.entity)

    def __hash__(self):
        return hash((self.entity, self.type))

    def __eq__(self, other):
        return hash(self) == hash(other)
