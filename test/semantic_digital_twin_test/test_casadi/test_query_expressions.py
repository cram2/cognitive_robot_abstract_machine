from krrood.entity_query_language.entity import (
    variable,
    entity,
    contains,
)
from krrood.entity_query_language.entity_result_processors import the, a, an
from semantic_digital_twin.spatial_types import Expression

from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable


def test_querying_equations(world_setup):
    position_var = variable(PositionVariable, domain=None)
    results = list(an(entity(position_var)).evaluate())
    expr = results[0] + results[1]
    e = variable(Expression, domain=None)
    found_expr = the(
        entity(e).where(
            e.is_scalar(),
            contains(e.free_variables(), results[0]),
            contains(e.free_variables(), results[1]),
        )
    ).evaluate()

    assert found_expr is expr
