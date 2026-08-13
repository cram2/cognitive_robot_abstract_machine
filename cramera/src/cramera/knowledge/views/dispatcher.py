"""
Dispatch a graph-panel tab name or a double-clicked node id to its view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, Dict, Optional

from cramera.knowledge.graph_payload import KnowledgeGraphPayload
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.subgraph import GraphPanelPayload
from cramera.knowledge.views.architecture import SubgraphViewPayload
from cramera.knowledge.views.chart import ChartViewPayload
from cramera.knowledge.views.kinematics import UrdfViewPayload
from cramera.knowledge.views.plan_tree import PlanViewPayload


@dataclass(kw_only=True)
class UnknownViewPayload(GraphPanelPayload):
    """
    The error payload returned for a graph-panel tab name that does not exist.
    """

    ok: bool = False
    """
    Always ``False``.
    """

    error: str = ""
    """
    Human-readable description of the unknown tab name.
    """

    def panel_options(self) -> Dict[str, Any]:
        """
        The error message; there is no graph to describe.
        """
        return {"error": self.error}


@dataclass
class GraphPanelViews:
    """
    The two ways the graph panel asks for a view: by tab name, or by drilling into a
    node that was double-clicked.

    Holding the knowledge base here is what keeps the individual views free of it: they
    are handed the episode they describe instead of reaching for the process-wide one.
    """

    knowledge_base: EpisodeKnowledgeBase
    """
    The recorded episode every view is built from.
    """

    @classmethod
    def of_active_scene(cls) -> GraphPanelViews:
        """
        The views of the scene bundle the server currently serves.
        """
        return cls.of_scene(None)

    @classmethod
    def of_scene(cls, scene: Optional[str]) -> GraphPanelViews:
        """
        The views of one named scene bundle.

        :param scene: Name of the scene to build views for, or None for the active one.
        """
        return cls(knowledge_base=EpisodeKnowledgeBase.of_scene(scene))

    def for_tab(self, name: str) -> GraphPanelPayload:
        """
        One tab of the graph panel.

        ``knowledge`` is the entity graph (the default, with drill-down); the others are
        structural views of the same demo that the UI can overlay with live status from
        the bridge (see :mod:`cramera.live.http`, ``/plan`` and ``/chart``).

        Every view declares the tab it serves as :attr:`GraphPanelPayload.TAB`, so
        adding one is a matter of subclassing rather than of extending this method.

        :param name: Name of the requested tab.
        """
        for payload_type in GraphPanelPayload.__subclasses__():
            if payload_type.TAB == name:
                return payload_type.of_tab(self.knowledge_base)
        return UnknownViewPayload(error="unknown view: %s" % name)

    def for_node(self, node_id: str) -> Optional[GraphPanelPayload]:
        """
        The inside view of a double-clicked node, or None if it has none.

        Unlike :meth:`for_tab` this cannot dispatch on a declared name: the id is looked
        up in the episode's packages, subpackages and classes in turn.

        :param node_id: Id of the double-clicked node.
        """
        if node_id == self.knowledge_base.robot.name:  # → full URDF kinematic tree
            return UrdfViewPayload.of_tab(self.knowledge_base)
        if node_id == PlanViewPayload.TAB:  # → the executed plan tree
            return PlanViewPayload.of_tab(self.knowledge_base)
        package = next(
            (entry for entry in self.knowledge_base.packages if entry.name == node_id),
            None,
        )
        if package:
            return SubgraphViewPayload.for_package(self.knowledge_base, package)
        subpackage = next(
            (
                entry
                for entry in self.knowledge_base.subpackages
                if entry.name == node_id
            ),
            None,
        )
        if subpackage:
            return SubgraphViewPayload.for_subpackage(self.knowledge_base, subpackage)
        python_class = next(
            (
                entry
                for entry in self.knowledge_base.classes
                if entry.qualified_name == node_id
            ),
            None,
        )
        if python_class:
            return SubgraphViewPayload.for_class(self.knowledge_base, python_class)
        return None
