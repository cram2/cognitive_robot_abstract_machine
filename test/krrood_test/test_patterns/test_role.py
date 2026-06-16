from __future__ import annotations

import doctest
from dataclasses import is_dataclass, dataclass, field
from typing import Type

import pytest

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import (
    HasRoleTaker,
    AssociationThroughRoleTaker,
)
from krrood.class_diagrams.utils import classes_of_module, T
from krrood.patterns.role import Role, role_taker_field
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from ..dataset.role_and_ontology import university_ontology_like_classes
from ..dataset.role_and_ontology.university_ontology_like_classes_without_descriptors import (
    PersonInRoleAndOntology,
    CEOAsFirstRole,
    Company,
    ProfessorAsFirstRole,
    Course,
    RepresentativeAsSecondRole,
    DelegateAsThirdRole,
)


def test_role_class_docstring_examples_execute_with_documented_results():
    # Runs the doctest embedded in the Role class docstring and verifies that every
    # documented result matches reality, so the docstring example cannot silently drift.
    #
    # The example defines a role class inline, so it runs in a dedicated namespace
    # rather than the role module's own namespace. The role module enables
    # ``from __future__ import annotations``; reusing that namespace would turn the
    # example annotations into strings the class diagram cannot resolve for inline
    # classes. A fresh namespace keeps annotations as real objects.
    namespace = {
        "Role": Role,
        "role_taker_field": role_taker_field,
        "__name__": "role_docstring_example",
    }
    docstring_test = doctest.DocTestParser().get_doctest(
        Role.__doc__, namespace, name="Role", filename=__file__, lineno=0
    )
    results = doctest.DocTestRunner(verbose=False).run(docstring_test)
    assert results.failed == 0


def test_getting_and_setting_attribute_for_role_and_role_taker():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    ceo.head_of = Company(name="BassCo")

    assert ceo.person.name == person.name

    # shared-base attr (from HasName) delegates from role to taker via DelegatorFor mixin property
    assert ceo.name == person.name

    # role-native attr lives on the role; access it directly from the role
    assert ceo.head_of == Company(name="BassCo")


def test_role_is_not_an_instance_of_its_taker():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    representative = RepresentativeAsSecondRole(ceo=ceo)

    # Pure composition: a role is not an instance of its role-taker type.
    assert not isinstance(ceo, PersonInRoleAndOntology)
    assert not isinstance(representative, PersonInRoleAndOntology)
    assert not isinstance(representative, CEOAsFirstRole)

    # A role is an instance of its own role type, and the taker is plainly itself.
    assert isinstance(ceo, CEOAsFirstRole)
    assert isinstance(representative, RepresentativeAsSecondRole)
    assert isinstance(person, PersonInRoleAndOntology)
    assert isinstance(person, CEOAsFirstRole) is False
    assert isinstance(person, RepresentativeAsSecondRole) is False

    # Role membership is expressed through has_role, not inheritance.
    assert Role.has_role(person, CEOAsFirstRole)
    assert Role.has_role(ceo, RepresentativeAsSecondRole)


def test_role_native_attr_accessible_from_roles_dict():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person, head_of=Company(name="BassCo"))
    assert Role.roles_for(person, CEOAsFirstRole)[0] is ceo
    assert Role.roles_for(person, CEOAsFirstRole)[0].head_of == Company(name="BassCo")


def test_getting_and_setting_attribute_between_sibling_roles():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(person=person)
    ceo.head_of = Company(name="BassCo")
    professor = ProfessorAsFirstRole(person=person)
    professor.teacher_of.append(Course(name="BassCourse"))

    # both roles share the same taker instance
    assert professor.person is ceo.person

    # role-native attrs on each role are directly accessible from that role
    assert professor.teacher_of[0].name == "BassCourse"
    assert ceo.head_of.name == "BassCo"

    # sibling role attrs are accessible via the roles dict on the shared taker
    assert Role.roles_for(person, CEOAsFirstRole)[0].head_of.name == "BassCo"
    assert (
        Role.roles_for(person, ProfessorAsFirstRole)[0].teacher_of[0].name
        == "BassCourse"
    )


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
    assert all(
        EntityAndType(role_taker) in map(EntityAndType, delegate.all_role_takers)
        for role_taker in delegate_role_takers
    )

    professor_role_takers = [person]
    assert len(professor_role_takers) == len(professor.all_role_takers)
    assert all(
        EntityAndType(role_taker) in map(EntityAndType, professor.all_role_takers)
        for role_taker in professor_role_takers
    )


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

    assert isinstance(
        Role.get_taker_roles_of_type(person, CEOAsFirstRole)[0], CEOAsFirstRole
    )
    assert isinstance(
        Role.get_taker_roles_of_type(person, RepresentativeAsSecondRole)[0],
        RepresentativeAsSecondRole,
    )
    assert Role.get_taker_roles_of_type(person, DelegateAsThirdRole) == []
    assert isinstance(
        Role.get_taker_roles_of_type(person, ProfessorAsFirstRole)[0],
        ProfessorAsFirstRole,
    )
    # All three roles (ceo, representative, professor) resolve back to the same taker.
    assert len(Role.roles_for(person)) == 3


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
