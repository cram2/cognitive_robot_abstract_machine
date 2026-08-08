"""
The scene robot's URDF kinematic-tree drill-down/tab view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from typing_extensions import Any, Dict, List, Optional, TYPE_CHECKING

from cram_viz.knowledge.enums import EdgeKind, NodeGroup
from cram_viz.knowledge.scene_bundle import UrdfJoint, load_scene, load_urdf
from cram_viz.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    LegendEntry,
    SubgraphAccumulator,
)

if TYPE_CHECKING:
    from cram_viz.knowledge.knowledge_base import EpisodeKnowledgeBase

#: the one URDF joint type that cannot move
FIXED_JOINT_TYPE = "fixed"


@dataclass
class UrdfViewPayload:
    """
    The scene robot's URDF as a kinematic tree.
    """

    ok: bool
    """
    Always ``True`` — this view has no failure mode.
    """

    crumb: str
    """
    Breadcrumb label shown above the subgraph.
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

    legend: Optional[List[LegendEntry]] = None
    """
    Colour legend rows, absent when the scene's URDF could not be found.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's graph panel expects.
        """
        payload = {
            "ok": self.ok,
            "crumb": self.crumb,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "details": {
                node_id: asdict(entry) for node_id, entry in self.details.items()
            },
        }
        if self.legend is not None:
            payload["legend"] = [asdict(entry) for entry in self.legend]
        return payload


def _is_movable(joint: UrdfJoint) -> bool:
    """
    Whether a URDF joint can move (every type except ``fixed``).

    :param joint: The URDF joint to check.
    """
    return joint.type != FIXED_JOINT_TYPE


def _urdf_view(knowledge_base: EpisodeKnowledgeBase) -> UrdfViewPayload:
    """
    The scene robot's URDF as a kinematic tree.

    Every link is a node, every joint an edge (parent → child); movable joints are solid
    edges, fixed ones dashed. Links are coloured by robot part from the recorded
    annotation.

    :param knowledge_base: The knowledge base whose robot's URDF is rendered.
    """
    parsed_urdf = load_urdf()
    links, joints = parsed_urdf.links, parsed_urdf.joints
    view = SubgraphAccumulator()
    if not links:
        return UrdfViewPayload(
            True, knowledge_base.robot.name + " · URDF (not found)", [], [], {}
        )

    scene = load_scene().scene
    parts = (scene.get("robot") or {}).get("parts") or {}
    link_to_part = {
        link: part for part, part_links in parts.items() for link in part_links
    }

    def chain_group(link_name: str) -> NodeGroup:
        """
        The visual group (colour) a kinematic-chain link is bucketed into.

        :param link_name: Name of the link to classify.
        """
        part = link_to_part.get(link_name, "").lower()
        if "gripper" in part or "hand" in part or "effector" in part:
            return NodeGroup.OBJECT  # grippers (teal)
        if "left" in part:
            return NodeGroup.ROBOT  # left arm (pink)
        if "right" in part:
            return NodeGroup.EVENT  # right arm (purple)
        lowered = link_name.lower()
        if any(
            keyword in lowered
            for keyword in ("head", "stereo", "sensor", "kinect", "camera", "laser")
        ):
            return NodeGroup.GOAL  # head / sensors (amber)
        return NodeGroup.CONCEPT  # base, torso, casters (green)

    # which joint drives each link (child link → its parent joint), for tooltips
    parent_joint = {joint.child: joint for joint in joints}
    for link in links:
        joint = parent_joint.get(link)
        lines = ["a URDF Link"]
        if joint:
            lines.append("joint: %s (%s)" % (joint.name, joint.type))
            lines.append("parent link: " + joint.parent)
        else:
            lines.append("root link")
        view.add("urdf:" + link, link, chain_group(link), lines)
    for joint in joints:
        if ("urdf:" + joint.parent) in view.details and (
            "urdf:" + joint.child
        ) in view.details:
            view.edges.append(
                GraphEdge(
                    "urdf:" + joint.parent,
                    "urdf:" + joint.child,
                    EdgeKind.PROP if _is_movable(joint) else EdgeKind.TYPE,
                    "%s (%s)" % (joint.name, joint.type),
                )
            )
    movable_count = sum(1 for joint in joints if _is_movable(joint))
    view.details["urdf:" + links[0]].lines.append(
        "%d links · %d joints (%d movable)" % (len(links), len(joints), movable_count)
    )
    legend = [
        LegendEntry(NodeGroup.CONCEPT, "Base / torso"),
        LegendEntry(NodeGroup.ROBOT, "Left arm"),
        LegendEntry(NodeGroup.EVENT, "Right arm"),
        LegendEntry(NodeGroup.OBJECT, "Grippers"),
        LegendEntry(NodeGroup.GOAL, "Head / sensors"),
    ]
    # force-directed, not hierarchical: the chains read better when the arms and
    # the sensor head spread out around the base than as one wide LR tree
    return UrdfViewPayload(
        True,
        knowledge_base.robot.name + " · URDF",
        view.nodes,
        view.edges,
        view.details,
        legend,
    )
