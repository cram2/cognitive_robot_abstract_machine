"""
The executed-plan-tree drill-down/tab view.

Named ``plan_tree`` rather than ``plan`` to keep it distinct from coraplex's own
``Plan``/``PlanNode`` types: this module renders the serialized tree of plan nodes
recorded in a scene bundle, not a coraplex ``Plan`` itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from typing_extensions import Any, Dict, List, Optional, Tuple

from cram_viz.knowledge.enums import EdgeKind, NodeGroup
from cram_viz.knowledge.scene_bundle import load_scene
from cram_viz.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    LegendEntry,
    SubgraphAccumulator,
)

#: plan-node kind → node colour group of the graph panel
PLAN_GROUPS: Dict[str, NodeGroup] = {
    "ActionNode": NodeGroup.EVENT,
    "MotionNode": NodeGroup.ROBOT,
    "ConditionNode": NodeGroup.GOAL,
    "AttachNode": NodeGroup.OBJECT,
    "DetachNode": NodeGroup.OBJECT,
}

#: legend rows of the plan view
PLAN_LEGEND: Tuple[LegendEntry, ...] = (
    LegendEntry(NodeGroup.EVENT, "Action"),
    LegendEntry(NodeGroup.ROBOT, "Motion"),
    LegendEntry(NodeGroup.GOAL, "Condition"),
    LegendEntry(NodeGroup.OBJECT, "Attach / detach"),
    LegendEntry(NodeGroup.OTHER, "Other plan node"),
)


@dataclass
class PlanViewPayload:
    """
    The executed plan as a tree, one node per plan node the demo ran.
    """

    ok: bool
    """
    Always ``True`` — this view has no failure mode.
    """

    nodes: List[GraphNode]
    """
    Every node in this view.
    """

    edges: List[GraphEdge]
    """
    Every edge in this view.
    """

    details: Dict[str, DetailEntry]
    """
    Detail-panel entry per node id.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's graph panel expects.
        """
        return {
            "ok": self.ok,
            "crumb": "executed plan",
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "details": {
                node_id: asdict(entry) for node_id, entry in self.details.items()
            },
            "legend": [asdict(entry) for entry in PLAN_LEGEND],
            "layout": "hier",
            "live": "plan",
            "statusLegend": True,
            "empty": "No plan tree in this bundle — re-run cram-viz-onboard.",
        }


def shorten_action_label(label: str) -> str:
    """
    Drop the redundant ``Action`` suffix from a plan-node label.

    Only the suffix goes: a label that merely *contains* the word, such as
    ``ActionNode``, is left alone.

    :param label: The plan-node label to shorten.
    """
    return label.removesuffix("Action") or label


def _plan_view() -> PlanViewPayload:
    """
    The executed plan as a tree, one node per plan node the demo ran.

    The recorded statuses are thin on purpose: coraplex performs only the
    plan *root* (``Plan.perform`` → ``root.perform``), while
    ``ActionNode.notify`` merely expands its children into the merged motion
    statechart. So every inner node of a recorded tree reads ``CREATED``, and
    real per-step progress only shows up while the live bridge is attached
    (it derives it from the statechart life cycle).
    """
    scene = load_scene().scene
    trees = scene.get("planTrees") or []
    view = SubgraphAccumulator()
    counter = [0]

    def walk(tree: Dict[str, Any], parent: Optional[str]) -> None:
        """
        Add this plan node (with a freshly assigned id) and recurse into its children.

        :param tree: The serialized plan node to add.
        :param parent: Id of the node's parent entry, or None for the root.
        """
        node_id = "pn%d" % counter[0]
        counter[0] += 1
        status = tree.get("status") or "CREATED"
        lines = ["a " + tree.get("kind", "PlanNode"), "status: " + status]
        if tree.get("arm"):
            lines.append("arm: " + tree["arm"])
        if tree.get("target"):
            lines.append("target: " + tree["target"])
        label = shorten_action_label(tree.get("label", "?"))
        view.add(
            node_id,
            label,
            PLAN_GROUPS.get(tree.get("kind"), NodeGroup.OTHER),
            lines,
            status=status,
        )
        if parent:
            view.edges.append(GraphEdge(parent, node_id, EdgeKind.PROP, "has step"))
        for child in tree.get("children", []):
            walk(child, node_id)

    for tree in trees:
        walk(tree, None)
    return PlanViewPayload(True, view.nodes, view.edges, view.details)
