from __future__ import annotations

import segmind.detectors.agent_event_detector_nodes
import segmind.detectors.atomic_event_detectors_nodes
import segmind.detectors.base
import segmind.detectors.coarse_event_detector_nodes
import segmind.detectors.spatial_relation_detector_nodes

from ...cramph_test.test_statechart.keyword_only_constructors import (
    KeywordOnlyConstructorAudit,
)

# %% keyword-only constructors


def test_every_detector_takes_keyword_arguments_only() -> None:
    audit = KeywordOnlyConstructorAudit(
        [
            segmind.detectors.base,
            segmind.detectors.agent_event_detector_nodes,
            segmind.detectors.atomic_event_detectors_nodes,
            segmind.detectors.coarse_event_detector_nodes,
            segmind.detectors.spatial_relation_detector_nodes,
        ]
    )
    assert audit.nodes_with_positional_parameters() == []
