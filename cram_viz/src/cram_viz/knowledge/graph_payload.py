"""
The knowledge-graph overview: nodes, edges, details and presets for the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from typing_extensions import Any, Dict, List, Optional

from cram_viz.knowledge.enums import EdgeKind, NodeGroup
from cram_viz.knowledge.knowledge_base import get_knowledge_base
from cram_viz.knowledge.presets import Preset, get_presets
from cram_viz.knowledge.scene_bundle import load_scene
from cram_viz.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    SubgraphAccumulator,
)


@dataclass
class KnowledgeGraphPayload:
    """
    The knowledge-graph overview: nodes, edges, details and presets.
    """

    ok: bool
    """
    Always ``True`` — this view has no failure mode.
    """

    status: str
    """
    Human-readable summary line shown above the graph panel.
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

    presets: List[Preset]
    """
    Ready-made EQL queries for the EQL panel.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's graph panel expects.
        """
        return {
            "ok": self.ok,
            "status": self.status,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "details": {
                node_id: asdict(entry) for node_id, entry in self.details.items()
            },
            "presets": [asdict(preset) for preset in self.presets],
        }


def _measurement_line(
    label: str, value: Optional[float], number_format: str
) -> List[str]:
    """
    A detail line for a measurement in metres, or nothing when it was not recorded.

    Showing a fabricated number would read as a fact about the scene.

    :param label: Label the measurement is shown under.
    :param value: The recorded measurement in metres, or None if it was not recorded.
    :param number_format: ``%``-style format applied to ``value``.
    """
    if value is None:
        return []
    return ["%s: %s m" % (label, number_format % value)]


def _count_plan_nodes(tree: Dict[str, Any]) -> int:
    """
    Number of nodes in a serialized plan tree.

    :param tree: The serialized plan tree to count.
    """
    return 1 + sum(_count_plan_nodes(child) for child in tree.get("children", []))


def graph_payload() -> KnowledgeGraphPayload:
    """
    The knowledge-graph overview: nodes, edges, details and presets.
    """
    kb = get_knowledge_base()
    view = SubgraphAccumulator()

    rob = kb.robot.name
    view.add(
        rob,
        rob,
        NodeGroup.ROBOT,
        [
            "a Robot",
            "%d arm%s" % (kb.robot.arm_count, "" if kb.robot.arm_count == 1 else "s"),
            "double-click: full URDF tree",
        ],
    )
    for arm in kb.arms:
        view.add(
            arm.name,
            arm.name.replace("_", " "),
            NodeGroup.ROBOT,
            ["an Arm", "side: " + arm.side, "gripper: " + arm.gripper.name],
        )
        view.edges.append(GraphEdge(rob, arm.name, EdgeKind.PROP, "has part"))
        view.add(
            arm.gripper.name,
            arm.gripper.name.replace("_", " "),
            NodeGroup.ROBOT,
            ["a Gripper", "side: " + arm.gripper.side]
            + _measurement_line("opening", arm.gripper.opening_m, "%.3f"),
        )
        view.edges.append(
            GraphEdge(arm.name, arm.gripper.name, EdgeKind.PROP, "has part")
        )

    for bench_object in kb.objects:
        view.add(
            bench_object.name,
            bench_object.label,
            NodeGroup.OBJECT,
            [
                "a BenchObject",
                "kind: " + bench_object.kind,
                "position: " + repr(bench_object.position),
            ]
            + _measurement_line("height", bench_object.height_m, "%.2f"),
        )

    previous = None
    for episode in kb.episodes:
        view.add(
            episode.name,
            episode.name,
            NodeGroup.EVENT,
            [
                "an ActionEpisode",
                "frames %d–%d" % (episode.start_frame, episode.end_frame),
                "duration: %.1f s" % episode.duration_s,
            ]
            + (["picks: " + episode.picks.name] if episode.picks else [])
            + (["places at: " + episode.places_at.name] if episode.places_at else []),
        )
        if previous:
            view.edges.append(
                GraphEdge(previous, episode.name, EdgeKind.TYPE, "precedes")
            )
        previous = episode.name
        # the robot performs the episode (with its arm); don't wire the episode
        # straight to the arm — the arm hangs off the robot, so the chain reads
        # transport_milk → pr2 → left_arm → left_gripper
        if episode.performed_by:
            view.edges.append(
                GraphEdge(
                    episode.name,
                    episode.performed_by.robot,
                    EdgeKind.PROP,
                    "performed by",
                )
            )
        if episode.picks:
            view.edges.append(
                GraphEdge(episode.name, episode.picks.name, EdgeKind.PROP, "picks")
            )
        if episode.places_at:
            view.edges.append(
                GraphEdge(
                    episode.name, episode.places_at.name, EdgeKind.PROP, "places at"
                )
            )

    # the CRAM architecture cluster: repo root → packages, plus import edges
    if kb.packages:
        view.add(
            "cram",
            "CRAM architecture",
            NodeGroup.ROOT,
            [
                "~/cognitive_robot_abstract_machine",
                "%d packages · %d Python classes" % (len(kb.packages), len(kb.classes)),
            ],
        )
        for package in kb.packages:
            view.add(
                package.name,
                package.name,
                NodeGroup.CONCEPT,
                [
                    "a Package",
                    package.description,
                    "%d modules · %d classes"
                    % (package.module_count, package.class_count),
                    "double-click to open",
                ],
            )
            view.edges.append(
                GraphEdge("cram", package.name, EdgeKind.PROP, "contains")
            )
        for subpackage in kb.subpackages:
            view.add(
                subpackage.name,
                subpackage.name.split(".", 1)[1],
                NodeGroup.KLASS,
                [
                    "a SubPackage of " + subpackage.package,
                    "%d modules · %d classes"
                    % (subpackage.module_count, subpackage.class_count),
                    "double-click to open",
                ],
            )
            view.edges.append(
                GraphEdge(
                    subpackage.package, subpackage.name, EdgeKind.PROP, "contains"
                )
            )
        for dependency in kb.package_deps:
            view.edges.append(
                GraphEdge(
                    dependency.source, dependency.target, EdgeKind.TYPE, "imports"
                )
            )

        # ground the demo in the architecture at the SUBPACKAGE that actually
        # realises each part (only wire to a node that exists in this view)
        def link(source: str, target: str, label: str) -> None:
            """
            Add an edge, but only if target is actually a node in this view.

            :param source: Id of the edge's source node.
            :param target: Id of the edge's target node; the edge is dropped if this
                  node is not in the view.
            :param label: Label shown on the edge.
            """
            if any(node.id == target for node in view.nodes):
                view.edges.append(GraphEdge(source, target, EdgeKind.TYPE, label))

        # anchor one representative manipulation episode (they share the stack)
        anchor = next((episode.name for episode in kb.episodes if episode.picks), None)
        if anchor:
            link(anchor, "coraplex.plans", "planned by")  # plan / designator layer
            link(anchor, "giskardpy.motion_statechart", "motion by")  # motion execution
        # every physical thing in the scene is modelled in the semantic digital twin
        link(rob, "semantic_digital_twin", "modelled in")
        for bench_object in kb.objects:
            link(bench_object.name, "semantic_digital_twin", "modelled in")

    # the executed plan tree (captured from the real PlanNode graph)
    scene = load_scene().scene
    if scene.get("planTrees"):
        node_count = sum(_count_plan_nodes(tree) for tree in scene["planTrees"])
        view.add(
            "plan",
            "executed plan",
            NodeGroup.GOAL,
            [
                "the plan tree the demo actually executed",
                "%d nodes" % node_count,
                "double-click to open",
            ],
        )
        view.edges.append(GraphEdge("plan", rob, EdgeKind.PROP, "executed by"))
        for episode in kb.episodes:
            view.edges.append(GraphEdge("plan", episode.name, EdgeKind.TYPE, "spans"))

    status = "EQL ready · %d graph nodes · %d joints · %d CRAM classes" % (
        len(view.nodes),
        len(kb.joints),
        len(kb.classes),
    )
    return KnowledgeGraphPayload(
        True, status, view.nodes, view.edges, view.details, get_presets()
    )
