"""
A query stands for what its selection yields, and what it reports as its type is what
every attribute chain taken from it inherits: an ``Attribute`` reads its owning class
off its child, so a query with no type gives back a chain that knows nothing about
itself.
"""

from krrood.entity_query_language.factories import (
    entity,
    set_of,
    variable,
    variable_from,
)

from ...dataset.semantic_world_like_classes import Body

# %% the type a query reports


def test_query_over_one_variable_reports_that_variable_type():
    assert entity(variable(Body, domain=[]))._type_ is Body


def test_query_over_several_variables_reports_no_type():
    """
    A row binding several variables is not a value of any one type, so there is no
    single type to report.
    """
    query = set_of(variable(Body, domain=[]), variable(Body, domain=[]))
    assert query._type_ is None


def test_query_over_a_selection_that_is_not_a_variable_reports_no_type():
    """
    A ``Comparator`` declares no ``_type_``, so reading one off it would be captured as
    a symbolic attribute rather than raising, and the capture reaches back into the
    query while it is still being built.
    """
    value = variable_from([6])
    assert entity(value > 5)._type_ is None


# %% what a chain taken from a query knows


def test_attribute_taken_from_a_query_carries_the_attribute_type():
    """
    The chain a query gives back knows what the same chain taken from the variable
    knows, which is what a condition's re-rooting already assumes of the two.
    """
    body = variable(Body, domain=[])
    assert entity(body).size._type_ is body.size._type_ is int


# %% the variable-rooted form of an expression


def test_chain_taken_from_a_query_has_the_variable_rooted_form_of_its_variable():
    body = variable(Body, domain=[])
    assert entity(body).size._variable_rooted_._name_ == body.size._name_


def test_chain_taken_from_a_variable_is_its_own_variable_rooted_form():
    body = variable(Body, domain=[])
    assert body.size._variable_rooted_ is body.size


def test_expression_that_is_no_chain_is_its_own_variable_rooted_form():
    body = variable(Body, domain=[])
    condition = body.size > 1
    assert condition._variable_rooted_ is condition
