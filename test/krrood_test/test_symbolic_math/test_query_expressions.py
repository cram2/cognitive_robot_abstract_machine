from krrood.entity_query_language.entity import (
    let,
    entity,
    contains,
)
from krrood.entity_query_language.match import match
from krrood.entity_query_language.quantify_entity import the, a
from semantic_digital_twin.spatial_types import Expression

from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable


def test_querying_equations(world_setup):
    results = list(a(match(PositionVariable)()).evaluate())
    expr = results[0] + results[1]
    found_expr = the(
        entity(
            e := let(Expression, domain=None),
            e.is_scalar(),
            contains(e.free_variables(), results[0]),
            contains(e.free_variables(), results[1]),
        )
    ).evaluate()

    assert found_expr is expr
