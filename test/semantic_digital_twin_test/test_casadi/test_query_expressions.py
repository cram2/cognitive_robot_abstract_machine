from krrood.entity_query_language.entity import (
    let,
    entity,
    contains,
)
from krrood.entity_query_language.entitymatch import matching, select
from krrood.entity_query_language.entity_result_processors import the, a
from semantic_digital_twin.spatial_types import Expression

from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable


def test_querying_equations(world_setup):
    results = list(a(matching(PositionVariable)).evaluate())
    expr = results[0] + results[1]
    e = the(matching(Expression))
    found_expr = (
        select(e)
        .where(
            e.is_scalar(),
            contains(e.free_variables(), results[0]),
            contains(e.free_variables(), results[1]),
        )
        .evaluate()
    )

    assert found_expr is expr
