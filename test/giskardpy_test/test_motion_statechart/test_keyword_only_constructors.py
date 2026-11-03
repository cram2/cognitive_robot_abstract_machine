from __future__ import annotations

import giskardpy.motion_statechart

from ...cramph_test.test_statechart.keyword_only_constructors import (
    KeywordOnlyConstructorAudit,
)

# %% keyword-only constructors


def test_every_motion_statechart_node_takes_keyword_arguments_only() -> None:
    audit = KeywordOnlyConstructorAudit([giskardpy.motion_statechart])
    assert audit.nodes_with_positional_parameters() == []
