from __future__ import annotations

import coraplex.perception
import coraplex.plans
import coraplex.robot_plans

from ...cramph_test.test_statechart.keyword_only_constructors import (
    KeywordOnlyConstructorAudit,
)

# %% keyword-only constructors


def test_every_coraplex_node_takes_keyword_arguments_only() -> None:
    audit = KeywordOnlyConstructorAudit(
        [coraplex.perception, coraplex.plans, coraplex.robot_plans]
    )
    assert audit.nodes_with_positional_parameters() == []
