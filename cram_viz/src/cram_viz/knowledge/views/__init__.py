"""
The graph-panel tabs and drill-down subgraphs of the knowledge package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from typing_extensions import Any, Dict, Optional

from cram_viz.knowledge.graph_payload import KnowledgeGraphPayload, graph_payload
from cram_viz.knowledge.knowledge_base import get_knowledge_base
from cram_viz.knowledge.views.architecture import (
    ArchitectureViews,
    SubgraphViewPayload,
    _class_id,
)
from cram_viz.knowledge.views.kinematics import UrdfViewPayload, _urdf_view
from cram_viz.knowledge.views.plan_tree import PlanViewPayload, _plan_view


@dataclass
class ChartViewPayload:
    """
    The (live-only) statechart tab.

    Motion statecharts only exist while giskardpy executes them: one is compiled per
    merged motion group and thrown away afterwards, and nothing of it is recorded into
    the bundle — the UI fills this view from the bridge's ``/chart`` while attached.
    """

    ok: bool
    """
    Always ``True`` — this view has no failure mode.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend expects.
        """
        return {
            "ok": self.ok,
            "crumb": "motion statechart",
            "nodes": [],
            "edges": [],
            "details": {},
            "layout": "hier",
            "live": "chart",
            "empty": "Motion statecharts are built and ticked at execution time. "
            "Start the demo with cram-viz-live and press ◉ Live — "
            "the statechart of the running motion group appears here, "
            "coloured by its node life cycle.",
        }


@dataclass
class UnknownViewPayload:
    """
    The error payload returned for a graph-panel tab name that does not exist.
    """

    ok: bool
    """
    Always ``False``.
    """

    error: str
    """
    Human-readable description of the unknown tab name.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend expects.
        """
        return {"ok": self.ok, "error": self.error}


#: any payload one of the graph-panel tabs or drill-down subgraphs can return
ViewPayload = Union[
    KnowledgeGraphPayload,
    ChartViewPayload,
    UrdfViewPayload,
    PlanViewPayload,
    SubgraphViewPayload,
]


def view_payload(name: str) -> Union[ViewPayload, UnknownViewPayload]:
    """
    One tab of the graph panel.

    ``knowledge`` is the entity graph (the default, with drill-down); the others are
    structural views of the same demo that the UI can overlay with live status from the
    bridge (see :mod:`cram_viz.live.http`, ``/plan`` and ``/chart``).

    :param name: Name of the requested tab.
    """
    kb = get_knowledge_base()
    if name == "knowledge":
        return graph_payload()
    if name == "kinematics":
        return _urdf_view(kb)
    if name == "plan":
        return _plan_view()
    if name == "chart":
        return _chart_view()
    return UnknownViewPayload(False, "unknown view: %s" % name)


def _chart_view() -> ChartViewPayload:
    """
    The (live-only) statechart tab.
    """
    return ChartViewPayload(True)


def expand_node(node_id: str) -> Optional[ViewPayload]:
    """
    The inside view of a double-clicked node, or None if not drillable.

    :param node_id: Id of the double-clicked node.
    """
    kb = get_knowledge_base()
    if node_id == kb.robot.name:  # robot → full URDF kinematic tree
        return _urdf_view(kb)
    if node_id == "plan":  # → the executed plan tree
        return _plan_view()
    package = next((entry for entry in kb.packages if entry.name == node_id), None)
    if package:
        return ArchitectureViews.package_view(kb, package)
    subpackage = next(
        (entry for entry in kb.subpackages if entry.name == node_id), None
    )
    if subpackage:
        return ArchitectureViews.subpackage_view(kb, subpackage)
    python_class = next(
        (entry for entry in kb.classes if _class_id(entry) == node_id), None
    )
    if python_class:
        return ArchitectureViews.class_view(kb, python_class)
    return None
