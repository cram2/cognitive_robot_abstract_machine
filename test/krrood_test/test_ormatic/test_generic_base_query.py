"""
A query for a generic base class selects the objects of all its subclasses, also when a
field typed with one of its parametrizations gives that parametrization a table.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.result_loading import (
    RelationshipLoading,
    ResultLoading,
    SubclassLoading,
)
from ..dataset.classes_with_generic import (
    FirstGeneric,
    RoverBase,
    SubClassGenericThatUpdatesGenericTypeToBuiltInType,
    WheeledBase,
    WheeledBaseHolder,
)

# %% queries


def test_a_query_for_a_generic_base_selects_its_subclasses_objects(session, database):
    """
    Selecting the objects of a generic base class finds those of a subclass binding it.
    """
    session.add(to_dao(RoverBase(label="rover")))
    session.commit()
    base = variable(WheeledBase, domain=None)

    result = eql_to_sql(an(entity(base)), session).evaluate()

    assert [dao.label for dao in result] == ["rover"]


def test_a_query_for_a_field_typed_with_a_parametrized_generic_selects_its_objects(
    session, database
):
    """
    Selecting the objects that a field typed with a parametrization of a generic holds
    finds them.
    """
    session.add(to_dao(WheeledBaseHolder(base=WheeledBase(label="plain"))))
    session.commit()
    holder = variable(WheeledBaseHolder, domain=None)

    result = eql_to_sql(an(entity(holder.base)), session).evaluate()

    assert [dao.label for dao in result] == ["plain"]


def test_a_query_for_a_generic_base_filters_on_a_field_of_its_type_parameter(
    session, database
):
    """
    A condition on a field typed with the type parameter of a generic base selects the
    objects of the subclasses binding it that meet it.
    """
    for value in [3, 4]:
        session.add(
            to_dao(
                SubClassGenericThatUpdatesGenericTypeToBuiltInType(
                    attribute_using_generic=value
                )
            )
        )
    session.commit()
    generic = variable(FirstGeneric, domain=None)
    query = an(entity(generic).where(generic.attribute_using_generic == 3))

    result = eql_to_sql(query, session).evaluate()

    assert [dao.attribute_using_generic for dao in result] == [3]


def test_the_result_loading_of_a_query_for_a_generic_base_loads_by_class():
    """
    The objects of a generic base class may be of its subclasses, so the result loading
    suited to a query for it loads them one statement per class.
    """
    query = an(entity(variable(WheeledBase, domain=None)))

    assert ResultLoading.from_query(query=query) == ResultLoading(
        relationship_loading=RelationshipLoading.ON_ACCESS,
        subclass_loading=SubclassLoading.ONE_QUERY_PER_CLASS,
    )
