"""
Where an expression is expected, anything that stands for one contributes the expression
it stands for, and only a value standing for nothing symbolic is read as a literal.
"""

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import a, entity, variable

from ...dataset.expression_stand_ins import ExpressionStandIn
from ...dataset.semantic_world_like_classes import Body

# %% what each kind of value contributes


def test_expression_contributes_itself():
    body = variable(Body, domain=[])
    assert SymbolicExpression._as_operand_(body.size) is body.size


def test_match_contributes_the_query_it_stands_for():
    match = a(Body)(name="Body1")
    assert SymbolicExpression._as_operand_(match) is match._symbolic_expression_


def test_value_standing_for_nothing_symbolic_is_a_literal():
    operand = SymbolicExpression._as_operand_(5)
    assert isinstance(operand, Literal)
    assert operand._value_ == 5


# %% a new kind of operand


def test_new_kind_of_operand_contributes_the_expression_it_stands_for():
    body = variable(Body, domain=[])
    stand_in = ExpressionStandIn(body.size)
    assert SymbolicExpression._as_operand_(stand_in) is body.size


def test_new_kind_of_operand_filters_as_the_expression_it_stands_for():
    bodies = [Body("Body1", size=1), Body("Body2", size=2)]
    body = variable(Body, domain=bodies)
    size = ExpressionStandIn(body.size)
    query = entity(body).where(body.size == 2, variable(int, domain=[2]) == size)
    assert query.tolist() == [bodies[1]]
