from __future__ import annotations

import giskardpy.motion_statechart

from ...cramph_test.test_statechart.node_rebuild_audit import NodeRebuildAudit

# %% building a node anew


def test_every_motion_statechart_node_holding_nodes_builds_them_anew() -> None:
    audit = NodeRebuildAudit(modules=[giskardpy.motion_statechart])
    assert audit.nodes_holding_nodes_they_do_not_rebuild() == []
