"""
The knowledge-graph overview: nodes, edges, details and presets for the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from typing_extensions import Any, ClassVar, Dict, List, Optional

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.spatial_types import Point3

from cramera.body_geometry import position_label
from cramera.knowledge.enums import EdgeKind, NodeGroup
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.presets import Preset
from cramera.knowledge.scene_bundle import SceneBundle
from cramera.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    GraphPanelPayload,
    SubgraphAccumulator,
)
from cramera.knowledge.views.plan_tree import PlanViewPayload


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
        """
        view = SubgraphAccumulator()

        robot_name = knowledge_base.robot.name
        view.add(
            robot_name,
            robot_name,
            NodeGroup.ROBOT,
            [
                "a Robot",
                "%d arm%s"
                % (
                    knowledge_base.robot.arm_count,
                    "" if knowledge_base.robot.arm_count == 1 else "s",
                ),
                "double-click: full URDF tree",
            ],
        )
        for arm in knowledge_base.arms:
            view.add(
                arm.name,
                arm.name.replace("_", " "),
                NodeGroup.ROBOT,
                [
                    "an Arm",
                    "side: " + cls._side_label(arm.side),
                    "gripper: " + arm.gripper.name,
                ],
            )
            view.add_edge(robot_name, arm.name, EdgeKind.PROPERTY, "has part")
            view.add(
                arm.gripper.name,
                arm.gripper.name.replace("_", " "),
                NodeGroup.ROBOT,
                ["a Gripper", "side: " + cls._side_label(arm.gripper.side)]
                + cls._measurement_line("opening", arm.gripper.opening_metres, "%.3f"),
            )
            view.add_edge(arm.name, arm.gripper.name, EdgeKind.PROPERTY, "has part")

        for bench_object in knowledge_base.objects:
            view.add(
                bench_object.name,
                bench_object.label,
                NodeGroup.OBJECT,
                [
                    "a BenchObject",
                    "kind: " + bench_object.kind,
                    "position: " + position_label(bench_object.position),
                ]
                + cls._measurement_line("height", bench_object.height_metres, "%.2f"),
            )

        previous = None
        for episode in knowledge_base.episodes:
            view.add(
                episode.name,
                episode.name,
                NodeGroup.EVENT,
                [
                    "an ActionEpisode",
                    "frames %d–%d" % (episode.start_frame, episode.end_frame),
                    "duration: %.1f s" % episode.duration_seconds,
                ]
                + (["picks: " + episode.picks.name] if episode.picks else [])
                + (
                    ["places at: " + episode.places_at.name]
                    if episode.places_at
                    else []
                ),
            )
            if previous:
                view.add_edge(previous, episode.name, EdgeKind.TYPE, "precedes")
            previous = episode.name
            # the robot performs the episode (with its arm); don't wire the episode
            # straight to the arm — the arm hangs off the robot, so the chain reads
            # transport_milk → pr2 → left_arm → left_gripper
            if episode.performed_by:
                view.add_edge(
                    episode.name,
                    episode.performed_by.robot,
                    EdgeKind.PROPERTY,
                    "performed by",
                )
            if episode.picks:
                view.add_edge(
                    episode.name, episode.picks.name, EdgeKind.PROPERTY, "picks"
                )
            if episode.places_at:
                view.add_edge(
                    episode.name, episode.places_at.name, EdgeKind.PROPERTY, "places at"
                )

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

            # ground the demo in the architecture at the SUBPACKAGE that actually
            # realises each part (only wire to a node that exists in this view)
            # anchor one representative manipulation episode (they share the stack)
            anchor = next(
                (episode.name for episode in knowledge_base.episodes if episode.picks),
                None,
            )
            if anchor:
                view.add_edge_to_existing(
                    anchor, "coraplex.plans", EdgeKind.TYPE, "planned by"
                )  # plan / designator layer
                view.add_edge_to_existing(  # motion execution
                    anchor,
                    "giskardpy.motion_statechart",
                    EdgeKind.TYPE,
                    "motion by",
                )
            # every physical thing in the scene is modelled in the semantic digital twin
            view.add_edge_to_existing(
                robot_name, "semantic_digital_twin", EdgeKind.TYPE, "modelled in"
            )
            for bench_object in knowledge_base.objects:
                view.add_edge_to_existing(
                    bench_object.name,
                    "semantic_digital_twin",
                    EdgeKind.TYPE,
                    "modelled in",
                )

        # the executed plan tree (captured from the real PlanNode graph)
        scene = SceneBundle.of_scene(knowledge_base.scene_name).scene
        if scene.get("planTrees"):
            node_count = PlanViewPayload.count_nodes(scene["planTrees"])
            view.add(
                "plan",
                "executed plan",
                NodeGroup.PLAN,
                [
                    "the plan tree the demo actually executed",
                    "%d nodes" % node_count,
                    "double-click to open",
                ],
            )
            view.add_edge("plan", robot_name, EdgeKind.PROPERTY, "executed by")
            for episode in knowledge_base.episodes:
                view.add_edge("plan", episode.name, EdgeKind.TYPE, "spans")

        status = "EQL ready · %d graph nodes · %d joints · %d CRAM classes" % (
            len(view.nodes),
            len(knowledge_base.joints),
            len(knowledge_base.classes),
        )
        return cls(
            status=status,
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
            presets=Preset.of_scene(knowledge_base.scene_name),
        )

    @staticmethod
    def _measurement_line(
        label: str, value: Optional[float], number_format: str
    ) -> List[str]:
        """
        A detail line for a measurement in metres, or nothing when it was not recorded.

        Showing a fabricated number would read as a fact about the scene.

        :param label: Label the measurement is shown under.
        :param value: The recorded measurement in metres, or None if it was not
            recorded.
        :param number_format:``%``-style format applied to ``value``.
        """
        if value is None:
            return []
        return ["%s: %s m" % (label, number_format % value)]

    @staticmethod
    def _side_label(side: Optional[Arms]) -> str:
        """
        Lower-case display name of an arm side, or ``unknown`` when it could not be
        inferred.

        :param side: The arm side to label.
        """
        return side.name.lower() if side is not None else "unknown"

