"""
A query for a generic base class selects the objects of all its subclasses, also when a
field typed with one of its parametrizations gives that parametrization a table.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from ..dataset.classes_with_generic import RoverBase, WheeledBase

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
