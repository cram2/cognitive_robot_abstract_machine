from __future__ import annotations

import doctest
from dataclasses import is_dataclass, dataclass, field
from typing import Type

import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.class_diagram import (
    HasRoleTaker,
    AssociationThroughRoleTaker,
)
from krrood.class_diagrams.method_classifier import (
    factory_method_names,
    is_factory_method,
)
from krrood.class_diagrams.utils import classes_of_module, T
from krrood.patterns.exceptions import DelegatedFactoryMethodError
from krrood.patterns.role import Role
from krrood.patterns.role_predicates import IsSameSemanticEntity
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic
from ..dataset.role_and_ontology import (
    university_ontology_like_classes,
    university_ontology_like_classes_without_descriptors,
)
from ..dataset.role_and_ontology.university_ontology_like_classes_without_descriptors import (
    PersonInRoleAndOntology,
    CEOAsFirstRole,
    CEOThatOverridesFactory,
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
        "__name__": "role_docstring_example",
    }
    docstring_test = doctest.DocTestParser().get_doctest(
        Role.__doc__, namespace, name="Role", filename=__file__, lineno=0
    )
    results = doctest.DocTestRunner(verbose=False).run(docstring_test)
    assert results.failed == 0


def test_getting_and_setting_attribute_for_role_and_role_taker():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    ceo.head_of = Company(name="BassCo")

    assert ceo.role_taker.name == person.name

    # shared-base attr (from HasName) delegates from role to taker via DelegatorFor mixin property
    assert ceo.name == person.name

    # role-native attr lives on the role; access it directly from the role
    assert ceo.head_of == Company(name="BassCo")


def test_role_is_not_an_instance_of_its_taker():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    representative = RepresentativeAsSecondRole(role_taker=ceo)

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
    ceo = CEOAsFirstRole(role_taker=person, head_of=Company(name="BassCo"))
    assert Role.roles_for(person, CEOAsFirstRole)[0] is ceo
    assert Role.roles_for(person, CEOAsFirstRole)[0].head_of == Company(name="BassCo")


def test_getting_and_setting_attribute_between_sibling_roles():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    ceo.head_of = Company(name="BassCo")
    professor = ProfessorAsFirstRole(role_taker=person)
    professor.teacher_of.append(Course(name="BassCourse"))

    # both roles share the same taker instance
    assert professor.role_taker is ceo.role_taker

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


def test_roles_are_distinct_objects_but_share_same_entity():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    representative = RepresentativeAsSecondRole(role_taker=ceo)
    professor = ProfessorAsFirstRole(role_taker=person)

    # A role is an ordinary object: equal only to itself, distinct from its taker and
    # from sibling roles, so all four stay distinct in a set.
    assert ceo != person
    assert ceo != representative
    assert ceo != professor
    assert len({person, ceo, representative, professor}) == 4

    # ...yet the IsSameEntity predicate sees through the role chain to the one root entity.
    assert IsSameSemanticEntity(ceo, person)
    assert IsSameSemanticEntity(ceo, representative)
    assert IsSameSemanticEntity(ceo, professor)
    assert IsSameSemanticEntity(representative, professor)


def test_multiple_roles_of_same_type_on_same_taker():
    person = PersonInRoleAndOntology(name="Bass")
    first = CEOAsFirstRole(role_taker=person, head_of=Company(name="Acme"))
    second = CEOAsFirstRole(role_taker=person, head_of=Company(name="Globex"))

    # Both same-type roles are retrievable and keep their own role-native state.
    retrieved = Role.roles_for(person, CEOAsFirstRole)
    assert len(retrieved) == 2
    assert {role.head_of.name for role in retrieved} == {"Acme", "Globex"}

    # They are distinct objects (no collapse in a set) but the same underlying entity.
    assert first is not second
    assert first != second
    assert len({first, second}) == 2
    assert IsSameSemanticEntity(first, second)


def test_is_same_entity_predicate():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    professor = ProfessorAsFirstRole(role_taker=person)
    representative = RepresentativeAsSecondRole(role_taker=ceo)
    delegate = DelegateAsThirdRole(role_taker=representative)
    other_person = PersonInRoleAndOntology(name="Other")

    # Two roles on one taker, and a role versus its taker, are the same entity.
    assert IsSameSemanticEntity(ceo, professor)
    assert IsSameSemanticEntity(ceo, person)
    # A role chain resolves to its root entity, in either operand order.
    assert IsSameSemanticEntity(delegate, person)
    assert IsSameSemanticEntity(person, delegate)
    # Unrelated entities are not the same; a plain object is the same as itself.
    assert not IsSameSemanticEntity(ceo, other_person)
    assert not IsSameSemanticEntity(person, other_person)
    assert IsSameSemanticEntity(person, person)


def test_mappings_between_roles_and_role_takers():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    representative = RepresentativeAsSecondRole(role_taker=ceo)
    professor = ProfessorAsFirstRole(role_taker=person)
    delegate = DelegateAsThirdRole(role_taker=representative)
    roles = [ceo, representative, professor, delegate]

    # Every role in the chain registers an edge to the root taker, so the root taker can
    # retrieve all of them (by identity), and each role reports the same underlying entity.
    roles_of_person = Role.roles_for(person)
    assert all(
        any(role is retrieved for retrieved in roles_of_person) for role in roles
    )
    assert all(IsSameSemanticEntity(role, person) for role in roles)

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
    ceo = CEOAsFirstRole(role_taker=person)
    representative = RepresentativeAsSecondRole(role_taker=ceo)
    professor = ProfessorAsFirstRole(role_taker=person)

    assert Role.has_role(person, CEOAsFirstRole)
    assert Role.has_role(person, RepresentativeAsSecondRole)
    assert not Role.has_role(person, DelegateAsThirdRole)
    assert Role.has_role(person, ProfessorAsFirstRole)


def test_get_roles_of_type():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)
    representative = RepresentativeAsSecondRole(role_taker=ceo)
    professor = ProfessorAsFirstRole(role_taker=person)

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


def test_delegated_factory_method_raises_on_call_through_role():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)

    # Accessing the delegated factory is fine; the guard fires only on call.
    delegated_factory = ceo.from_name
    with pytest.raises(DelegatedFactoryMethodError):
        delegated_factory("Other")
    with pytest.raises(DelegatedFactoryMethodError):
        ceo.from_name("Other")

    # The @factory_method-marked classmethod is guarded the same way.
    with pytest.raises(DelegatedFactoryMethodError):
        ceo.spawn()


def test_factory_method_still_callable_on_taker_and_root_persistent_entity():
    person = PersonInRoleAndOntology(name="Bass")
    ceo = CEOAsFirstRole(role_taker=person)

    built_from_taker = ceo.role_taker.from_name("Made")
    built_from_root = ceo.root_persistent_entity.from_name("Made")

    assert isinstance(built_from_taker, PersonInRoleAndOntology)
    assert built_from_taker.name == "Made"
    assert isinstance(built_from_root, PersonInRoleAndOntology)


def test_role_overriding_factory_method_shadows_the_guard():
    # A role that overrides the factory keeps the role instead of dropping it. Normal attribute
    # lookup finds the override, so __getattr__ (and the guard) never runs.
    from_class = CEOThatOverridesFactory.from_name("Bass")
    assert isinstance(from_class, CEOThatOverridesFactory)
    assert from_class.name == "Bass"

    existing = CEOThatOverridesFactory(role_taker=PersonInRoleAndOntology(name="Seed"))
    from_instance = existing.from_name("Bass")
    assert isinstance(from_instance, CEOThatOverridesFactory)
    assert from_instance.name == "Bass"


def test_factory_method_guard_applies_through_nested_roles():
    person = PersonInRoleAndOntology(name="Bass")
    representative = RepresentativeAsSecondRole(
        role_taker=CEOAsFirstRole(role_taker=person)
    )
    with pytest.raises(DelegatedFactoryMethodError):
        representative.from_name("Other")


def test_ordinary_taker_method_still_delegates_through_role():
    person = PersonInRoleAndOntology(name="Bass", works_for=Company(name="BassCo"))
    ceo = CEOAsFirstRole(role_taker=person)

    # An ordinary (non-factory) classmethod delegates to the taker and runs normally.
    assert ceo.describe() == "PersonInRoleAndOntology"
    # An instance method also keeps delegating.
    assert ceo.method_in_person() == Company(name="BassCo")


def test_is_factory_method_classification():
    assert is_factory_method(PersonInRoleAndOntology, "from_name")  # -> Self annotation
    assert is_factory_method(PersonInRoleAndOntology, "spawn")  # @factory_method marker
    assert not is_factory_method(PersonInRoleAndOntology, "describe")  # -> str
    assert not is_factory_method(
        PersonInRoleAndOntology, "method_in_person"
    )  # instance
    assert not is_factory_method(PersonInRoleAndOntology, "name")  # not a method

    names = factory_method_names(PersonInRoleAndOntology)
    assert "from_name" in names and "spawn" in names
    assert "describe" not in names and "method_in_person" not in names


def test_wrapped_class_exposes_factory_methods():
    classes = [
        cls
        for cls in classes_of_module(
            university_ontology_like_classes_without_descriptors
        )
        if is_dataclass(cls)
    ]
    diagram = ClassDiagram(classes)
    wrapped = diagram.get_wrapped_class(PersonInRoleAndOntology)
    assert "from_name" in wrapped.factory_methods
    assert "spawn" in wrapped.factory_methods
    assert "describe" not in wrapped.factory_methods


@dataclass
class EntityAndType(Generic[T], SubClassSafeGeneric):
    entity: T
    type: Type[T] = field(init=False)

    def __post_init__(self):
        self.type = type(self.entity)

    def __hash__(self):
        return hash((self.entity, self.type))

    def __eq__(self, other):
        return hash(self) == hash(other)


def test_role_taker_is_a_required_keyword_only_field():
    person = PersonInRoleAndOntology(name="Bass")

    # Keyword-only: the inherited taker cannot be passed positionally.
    with pytest.raises(TypeError):
        CEOThatOverridesFactory(person)

    # Required: constructing a role without its taker is an error.
    with pytest.raises(TypeError):
        CEOThatOverridesFactory()

    role = CEOThatOverridesFactory(role_taker=person)
    assert role.role_taker is person


def test_assigning_role_taker_rebinds_the_underlying_entity():
    first = PersonInRoleAndOntology(name="First")
    second = PersonInRoleAndOntology(name="Second")
    ceo = CEOAsFirstRole(role_taker=first)

    ceo.role_taker = second

    assert ceo.role_taker is second
    # Delegated reads now resolve against the new taker.
    assert ceo.name == "Second"


def test_role_taker_field_factory_and_marker_are_removed():
    import krrood.patterns.role as role_module

    # The taker is now an inherited field; the old factory and marker class no longer exist.
    assert not hasattr(role_module, "role_taker_field")
    assert not hasattr(role_module, "RoleTakerField")
