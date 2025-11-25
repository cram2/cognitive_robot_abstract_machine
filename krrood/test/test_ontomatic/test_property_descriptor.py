from __future__ import annotations

from test.dataset.university_ontology_like_classes import Company, Person, CEO
from krrood.entity_query_language.symbol_graph import SymbolGraph

SymbolGraph().clear()
SymbolGraph()


def test_set_non_container_property():
    company = Company(name="BassCo")
    person1 = Person(name="Bass1")

    person1.works_for = company
    assert person1.works_for == company
    assert person1 in company.members

    person2 = Person(name="Bass2")
    another_company = Company(name="AnotherBassCo")
    person2.works_for = another_company
    assert person2.works_for == another_company
    assert person2 in another_company.members
    assert person2 not in company.members


def test_set_container_property():
    company = Company(name="BassCo")
    company2 = Company(name="AnotherBassCo")
    person1 = Person(name="Bass1")
    person2 = Person(name="Bass2")

    # test direct setting of a set
    company.members = {person1, person2}
    assert person1 in company.members
    assert person2 in company.members
    assert company in person1.member_of
    assert company in person2.member_of
    assert person1.works_for != company
    assert person2.works_for != company

    person1.member_of.append(company2)
    assert company2 in person1.member_of
    assert person1 in company2.members
    assert person1.works_for != company2


def test_setting_a_role_affects_role_taker():
    company = Company(name="BassCo")
    person1 = Person(name="Bass1")
    ceo1 = CEO(person1)

    ceo1.head_of = company
    assert ceo1.head_of == company
    assert ceo1.person.works_for == company
    assert ceo1 in company.members
    assert company in ceo1.person.member_of


def test_transitive_property():
    company = Company(name="BassCo")
    company2 = Company(name="AnotherBassCo")
    company3 = Company(name="ThirdBassCo")
    company4 = Company(name="FourthBassCo")

    company4.sub_organization_of = company3
    company3.sub_organization_of = company2
    company2.sub_organization_of = company

    assert company in company3.sub_organization_of
    assert company2 in company4.sub_organization_of
    assert company in company4.sub_organization_of
