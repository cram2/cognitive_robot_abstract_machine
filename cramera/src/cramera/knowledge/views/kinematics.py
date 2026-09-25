"""
The scene robot's URDF kinematic-tree drill-down/tab view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from coraplex.datastructures.enums import Arms, JointType
from typing_extensions import Any, ClassVar, Dict, List, Optional, TYPE_CHECKING

from cramera.knowledge.enums import EdgeKind, KinematicChainGroup
from cramera.knowledge.scene_bundle import ParsedUrdf
from cramera.robot_parts import RobotPartAnnotation, RobotPartRole
from cramera.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    GraphPanelPayload,
    LegendEntry,
    SubgraphAccumulator,
)

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase


@dataclass(kw_only=True)
class UrdfViewPayload(GraphPanelPayload):
    """
    The scene robot's URDF as a kinematic tree.
    """

    TAB: ClassVar[Optional[str]] = "kinematics"

    breadcrumb: str
    """
    Breadcrumb label shown above the tree.
    """

    legend: Optional[List[LegendEntry]] = None
    """
    Robot-part colour legend, or None when the URDF could not be read.
    """

    def panel_options(self) -> Dict[str, Any]:
        """
        The breadcrumb, plus the part legend once the URDF has been read.
        """
        options: Dict[str, Any] = {"breadcrumb": self.breadcrumb}
        if self.legend is not None:
            options["legend"] = [asdict(entry) for entry in self.legend]
        return options

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> UrdfViewPayload:
        """
        The scene robot's URDF as a kinematic tree.

        Every link is a node, every joint an edge (parent → child); movable joints are
        solid edges, fixed ones dashed. Links are coloured by robot part from the
        recorded annotation.

        :param knowledge_base: The knowledge base whose robot's URDF is rendered.
        """
        parsed_urdf = ParsedUrdf.of_scene(knowledge_base.scene_name)
        links, joints = parsed_urdf.links, parsed_urdf.joints
        view = SubgraphAccumulator()
        if not links:
            return cls(breadcrumb=knowledge_base.robot.name + " · URDF (not found)")

        link_to_part = {
            link: part
            for part in sorted(
                knowledge_base.part_annotations,
                key=lambda part: (
                    part.role is not RobotPartRole.ARM,
                    part.role is RobotPartRole.END_EFFECTOR,
                ),
            )
            for link in part.links
        }

        # which joint drives each link (child link → its parent joint), for tooltips
        parent_joint = {joint.child: joint for joint in joints}
        for link in links:
            joint = parent_joint.get(link)
            lines = ["a URDF Link"]
            if joint:
                lines.append("joint: %s (%s)" % (joint.name, joint.type.name.lower()))
                lines.append("parent link: " + joint.parent)
            else:
                lines.append("root link")
            view.add(
                "urdf:" + link,
                link,
                cls._chain_group(link_to_part.get(link)),
                lines,
            )
        for joint in joints:
            if ("urdf:" + joint.parent) in view.details and (
                "urdf:" + joint.child
            ) in view.details:
                view.add_edge(
                    "urdf:" + joint.parent,
                    "urdf:" + joint.child,
                    (
                        EdgeKind.PROPERTY
                        if joint.type != JointType.FIXED
                        else EdgeKind.TYPE
                    ),
                    "%s (%s)" % (joint.name, joint.type.name.lower()),
                )
        movable_count = sum(1 for joint in joints if joint.type != JointType.FIXED)
        view.details["urdf:" + links[0]].lines.append(
            "%d links · %d joints (%d movable)"
            % (len(links), len(joints), movable_count)
        )
        legend = [
            LegendEntry(group, group.label) for group in KinematicChainGroup.legend()
        ]
        # force-directed, not hierarchical: the chains read better when the arms and
        # the sensor head spread out around the base than as one wide LR tree
        return cls(
            breadcrumb=knowledge_base.robot.name + " · URDF",
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
            legend=legend,
        )

    @staticmethod
    def _chain_group(part: RobotPartAnnotation | None) -> KinematicChainGroup:
        if part is None:
            return KinematicChainGroup.BASE
        if part.role is RobotPartRole.END_EFFECTOR:
            return KinematicChainGroup.GRIPPER
        if part.role is RobotPartRole.SENSOR:
            return KinematicChainGroup.SENSOR
        return {
            Arms.LEFT: KinematicChainGroup.LEFT_ARM,
            Arms.RIGHT: KinematicChainGroup.RIGHT_ARM,
        }.get(part.side, KinematicChainGroup.BASE)
