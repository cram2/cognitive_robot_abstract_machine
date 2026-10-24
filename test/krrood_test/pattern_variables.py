"""
Lookups of the variables a match's pattern assigns to its attributes, for keying the
bindings a test constructs or scores.
"""

from __future__ import annotations

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.match import Match


def find_assigned_variable(match: Match, access_path_name: str) -> SymbolicExpression:
    """
    :param match: A match whose pattern has been stated.
    :param access_path_name: The name of an attribute's access path, which is also the
        name of the model variable it corresponds to, e.g. ``"Rectangle.width"``.
    :return: The variable the pattern assigns to that attribute.
    """
    [assigned_variable] = [
        attribute_match.assigned_variable
        for attribute_match in match._matches_with_variables_
        if attribute_match.name_from_variable_access_path == access_path_name
    ]
    return assigned_variable
