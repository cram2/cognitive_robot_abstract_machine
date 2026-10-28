"""
Reading back what a statement of where an object may be grasped allows.
"""

from krrood.entity_query_language.query.match import Match
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.interval import Interval


def allowed_intervals(statement: Match) -> dict[str, Interval]:
    """
    :param statement: A statement over a surface grasp.
    :return: For each of the grasp's parameters, by name, the smallest interval holding
        every value the statement allows it.
    """
    allowed = UnderspecifiedParameters(
        statement
    ).truncation_assignments_from_where_conditions.bounding_box()
    return {
        variable.name.rsplit(".", 1)[-1]: allowed[variable]
        for variable in allowed.variables
    }
