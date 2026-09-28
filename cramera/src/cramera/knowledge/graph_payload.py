"""
The knowledge-graph overview: nodes, edges, details and presets for the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from typing_extensions import Any, ClassVar, Dict, List, Optional


from cramera.knowledge.enums import EdgeKind, NodeGroup
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.presets import Preset
from cramera.knowledge.scene_bundle import SceneBundle
from cramera.onboard.scene_index import SceneIndexEntry
from cramera.knowledge.subgraph import (
    GraphPanelPayload,
    SubgraphAccumulator,
)


@dataclass(kw_only=True)
class KnowledgeGraphPayload(GraphPanelPayload):
    """
    The knowledge-graph overview: the whole recorded episode in one graph.
    """

    TAB: ClassVar[Optional[str]] = "knowledge"

    status: str = ""
    """
    One-line summary shown above the graph.
    """

    presets: List[Preset] = field(default_factory=list)
    """
    Ready-made EQL queries the query panel offers.
    """

    def panel_options(self) -> Dict[str, Any]:
        """
        The status line and the query presets, which only the overview sends.
        """
        return {
            "status": self.status,
            "presets": [asdict(preset) for preset in self.presets],
        }

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> KnowledgeGraphPayload:
        """
        The knowledge-graph overview: nodes, edges, details and presets.

        :param knowledge_base: Recorded scene entities and inspected architecture.
        """
        view = SubgraphAccumulator()

        # the CRAM architecture cluster: repo root → packages, plus import edges
        if knowledge_base.packages:
            view.add(
                "cram",
                "CRAM architecture",
                NodeGroup.ROOT,
                [
                    "~/cognitive_robot_abstract_machine",
                    "%d packages · %d Python classes"
                    % (len(knowledge_base.packages), len(knowledge_base.classes)),
                ],
            )
            for package in knowledge_base.packages:
                view.add(
                    package.name,
                    package.name,
                    NodeGroup.PACKAGE,
                    [
                        "a Package",
                        package.description,
                        "%d modules · %d classes"
                        % (package.module_count, package.class_count),
                        "double-click to open",
                    ],
                )
                view.add_edge("cram", package.name, EdgeKind.PROPERTY, "contains")
            for subpackage in knowledge_base.subpackages:
                view.add(
                    subpackage.name,
                    subpackage.name.split(".", 1)[1],
                    NodeGroup.SUBPACKAGE,
                    [
                        "a SubPackage of " + subpackage.package,
                        "%d modules · %d classes"
                        % (subpackage.module_count, subpackage.class_count),
                        "double-click to open",
                    ],
                )
                view.add_edge(
                    subpackage.package, subpackage.name, EdgeKind.PROPERTY, "contains"
                )
            for dependency in knowledge_base.package_dependencies:
                view.add_edge(
                    dependency.source, dependency.target, EdgeKind.TYPE, "imports"
                )

        scene = SceneBundle.of_scene(knowledge_base.scene_name).scene
        status = (
            "recorded · %s"
            % SceneIndexEntry.of_scene(
                knowledge_base.scene_name or SceneBundle.active_name() or "", scene
            ).describes()
        )
        return cls(
            status=status,
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
            presets=Preset.of_scene(knowledge_base.scene_name),
        )
