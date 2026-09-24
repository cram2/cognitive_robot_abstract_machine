"""
Live bridge snapshots of native plans, worlds and motion statecharts.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field

import pytest
from coraplex.datastructures.enums import Arms
from coraplex.language import SequentialNode
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.robot_plans.motions.base import BaseMotion
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Dict, List, Optional, Tuple

from cramera.knowledge.enums import PlanNodeGroup
from cramera.live.chart_structure import ChartEdgeEntry
from cramera.live.bridge import (
    Bridge,
    ROBOT_BASE_KEY,
)

from .dataset.plan_metadata import ArmSelectionAction, BodyTargetMotion
from .test_robot_parts import ArmPart, EndEffectorPart, NamedBody, OneArmedRobot


# %% world geometry fixtures
@dataclass
class ShapeSet:
    """
    A body's shape collection, of which the bridge reads only the shapes.
    """

    shapes: List[Any] = field(default_factory=list)


@dataclass
class PublishedBody:
    """
    A world body as the bridge publishes it: a prefixed name and its shapes.
    """

    name: str
    visual: ShapeSet = field(default_factory=ShapeSet)
    collision: ShapeSet = field(default_factory=ShapeSet)


# %% native plan fixtures
@pytest.fixture()
def plan_bridge() -> (
    tuple[Bridge, SequentialNode, ActionNode, ConditionNode, MotionNode]
):
    """
    Build a native plan with an action, condition, and body-targeting motion.

    :return: The observing bridge and the plan's root, action, condition and motion.
    """
    bridge = Bridge()
    target = Body(name=PrefixedName("milk.stl", prefix="world"))
    motion = MotionNode(designator=BodyTargetMotion(target_body=target))
    action = ActionNode(designator=ArmSelectionAction(arm=Arms.RIGHT))
    condition = ConditionNode(condition=True, pre_condition=True, action_node=action)
    root = SequentialNode()
    plan = Plan()
    plan.add_edge(root, action)
    plan.add_edge(action, condition)
    plan.add_edge(action, motion)
    bridge.publish_bodies({"milk.stl": target})
    bridge.begin_plan(plan)
    return bridge, root, action, condition, motion


def nodes_by_kind(bridge: Bridge) -> Dict[str, Dict[str, Any]]:
    """
    The published plan nodes, indexed by the class name they were built from.
    """
    return {node["kind"]: node for node in bridge.get_plan()["nodes"]}


# %% plan tree
class TestPlanSnapshot:
    """
    Each published node retains its native identity, metadata, and lifecycle.
    """

    def test_running_nodes_publish_their_own_status(self, plan_bridge) -> None:
        """
        Execution states are read from both the native action and its motion.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, _, motion = plan_bridge
        action.status = LifeCycleValues.RUNNING
        motion.status = LifeCycleValues.RUNNING
        bridge.snapshot_plan()
        nodes = nodes_by_kind(bridge)
        assert nodes["MotionNode"]["status"] == motion.status.name
        assert nodes["MotionNode"]["derived"] is False
        assert nodes["ActionNode"]["status"] == action.status.name

    def test_each_node_carries_the_colour_group_of_its_kind(self, plan_bridge):
        """
        The bridge publishes the native node kind's colour group.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, *_ = plan_bridge
        by_kind = {node["kind"]: node["group"] for node in bridge.get_plan()["nodes"]}
        assert by_kind["ActionNode"] == PlanNodeGroup.ACTION
        assert by_kind["MotionNode"] == PlanNodeGroup.MOTION
        assert by_kind["ConditionNode"] == PlanNodeGroup.CONDITION
        assert by_kind["SequentialNode"] == PlanNodeGroup.OTHER

    def test_the_plan_payload_carries_the_legend_of_every_group(self, plan_bridge):
        bridge, *_ = plan_bridge
        assert bridge.get_plan()["legend"] == [
            {"group": group.value, "label": group.label} for group in PlanNodeGroup
        ]

    def test_designator_metadata_is_serialized(self, plan_bridge) -> None:
        """
        Publish the target body and arm selected by native designators.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, _, motion = plan_bridge
        nodes = nodes_by_kind(bridge)
        assert nodes["MotionNode"]["target"] == motion.designator.target_body.name.name
        assert nodes["ActionNode"]["arm"] == str(action.designator.arm)

    def test_completed_parent_keeps_its_status_with_unstarted_children(
        self, plan_bridge
    ) -> None:
        """
        An unstarted child does not change its completed parent's lifecycle.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, root, *_ = plan_bridge
        root.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert nodes_by_kind(bridge)["SequentialNode"]["status"] == root.status.name
        assert nodes_by_kind(bridge)["SequentialNode"]["derived"] is False

    def test_running_parent_remains_running_after_a_motion_finishes(
        self, plan_bridge
    ) -> None:
        """
        Motion completion does not complete the native action's execution scope.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, _, motion = plan_bridge
        action.status = LifeCycleValues.RUNNING
        motion.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == action.status.name

    def test_completed_action_publishes_its_terminal_state(self, plan_bridge) -> None:
        """
        The action's terminal state is retained with completed descendants.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, condition, motion = plan_bridge
        action.status = condition.status = motion.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == action.status.name

    def test_failed_action_keeps_its_failure_with_succeeded_motion(
        self, plan_bridge
    ) -> None:
        """
        A successful motion does not clear an action-level failure.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, _, motion = plan_bridge
        action.status = LifeCycleValues.FAILED
        motion.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == action.status.name

    def test_signature_is_stable_across_status_changes(self, plan_bridge) -> None:
        """
        Changing lifecycle states leaves the plan structure signature unchanged.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, _, _, motion = plan_bridge
        motion.status = LifeCycleValues.RUNNING
        bridge.snapshot_plan()
        while_running = bridge.get_plan()["signature"]
        motion.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert bridge.get_plan()["signature"] == while_running

    def test_identical_designators_keep_separate_node_statuses(self) -> None:
        """
        Two steps with identical parameters retain their independent lifecycles.
        """
        bridge = Bridge()
        first = MotionNode(designator=BaseMotion(), status=LifeCycleValues.FAILED)
        second = MotionNode(designator=BaseMotion())
        root = SequentialNode()
        plan = Plan()
        plan.add_edge(root, first)
        plan.add_edge(root, second)
        bridge.begin_plan(plan)
        statuses = [
            node["status"]
            for node in bridge.get_plan()["nodes"]
            if node["kind"] == MotionNode.__name__
        ]
        assert statuses == [first.status.name, second.status.name]

    def test_a_new_plan_publishes_only_its_current_nodes(self, plan_bridge) -> None:
        """
        Starting a new plan replaces the previously published hierarchy.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, _, _, motion = plan_bridge
        motion.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        replacement = Plan()
        replacement.add_node(MotionNode(designator=BaseMotion()))
        bridge.begin_plan(replacement)
        [published] = bridge.get_plan()["nodes"]
        assert published["status"] == replacement.root.status.name
        assert published["id"] != "plan_node_%d" % id(motion)


# %% recording step labels
class TestRunningStep:
    """
    Recorded ticks name the deepest action whose native execution is running.
    """

    def test_nothing_is_reported_before_anything_runs(self, plan_bridge) -> None:
        """
        A plan that has not started does not label recording ticks.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, *_ = plan_bridge
        assert bridge.running_step() is None

    def test_the_running_action_is_reported(self, plan_bridge) -> None:
        """
        The running action provides the recording step label.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, _, _ = plan_bridge
        action.status = LifeCycleValues.RUNNING
        bridge.snapshot_plan()
        assert bridge.running_step() == type(action.designator).__name__

    def test_a_finished_action_is_no_longer_reported(self, plan_bridge) -> None:
        """
        A completed action no longer labels subsequent recording ticks.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, _, _ = plan_bridge
        action.status = LifeCycleValues.RUNNING
        bridge.snapshot_plan()
        action.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert bridge.running_step() is None

    def test_a_running_motion_is_not_reported_as_the_step(self):
        """
        A motion without a running action does not name a recording step.
        """
        bridge = Bridge()
        motion = MotionNode(designator=BaseMotion(), status=LifeCycleValues.RUNNING)
        plan = Plan()
        plan.add_node(motion)
        bridge.begin_plan(plan)
        assert bridge.running_step() is None


def make_hinged_door() -> Tuple[World, RevoluteConnection]:
    """
    A world with a fridge door on a hinge that swings between closed and a quarter turn.
    """
    world = World()
    fridge = Body(name=PrefixedName("fridge"))
    door = Body(name=PrefixedName("fridge_door"))
    hinge = DegreeOfFreedom(
        name=PrefixedName("fridge_door_joint", prefix="kitchen"),
        limits=DegreeOfFreedomLimits(
            lower=DerivativeMap(position=0.0), upper=DerivativeMap(position=1.57)
        ),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(fridge)
        world.add_kinematic_structure_entity(door)
        world.add_degree_of_freedom(hinge)
        connection = RevoluteConnection(
            parent=fridge,
            child=door,
            axis=Vector3.from_iterable([0, 0, 1]),
            raw_dof=hinge,
        )
        world.add_connection(connection)
    return world, connection


def bridge_attached_to(world: World) -> Bridge:
    """
    A bridge bound to ``world`` the way :meth:`Bridge.bind` binds the demo's world.
    """
    bridge = Bridge()
    bridge.world = world
    bridge._connections = bridge._actuated_connections(list(world.connections))
    return bridge


def make_free_floating_object() -> Tuple[World, Connection6DoF, Body]:
    """
    A world with one free-floating body, connected to the root by a Connection6DoF.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    obj = Body(name=PrefixedName("milk"))
    with world.modify_world():
        x, y, z, qx, qy, qz, qw = (
            DegreeOfFreedom(name=PrefixedName(component))
            for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
        )
        for dof in (x, y, z, qx, qy, qz, qw):
            world.add_degree_of_freedom(dof)
        connection = Connection6DoF(
            parent=root, child=obj, x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw
        )
        world.add_connection(connection)
        world.state[qw.id].position = 1.0
    return world, connection, obj


# %% what the Plan Builder captures


# %% what the HTTP layer reads
class TestViewerAccessors:
    """
    The viewer reads the bridge's published bodies and current session state.
    """

    def test_object_keys_exclude_the_robot_base(self):
        bridge = Bridge()
        bridge.publish_bodies(
            {
                ROBOT_BASE_KEY: PublishedBody(name="world/base_link"),
                "milk.stl": PublishedBody(name="world/milk.stl"),
            }
        )
        assert bridge.object_keys() == ["milk.stl"]

    def test_an_object_with_unscaled_shapes_falls_back_to_the_default_size(self):
        """
        A shapeless published body retains the shared placeholder dimensions.
        """
        bridge = Bridge()
        bridge.publish_bodies({"blob.stl": PublishedBody(name="world/blob.stl")})
        assert bridge.object_catalog()[0]["shapes"][0]["size"] == list(
            Bridge.DEFAULT_OBJECT_SIZE
        )

    def test_an_unserved_mesh_has_no_path(self):
        assert Bridge().mesh_path("milk.stl") is None

    def test_object_body_returns_the_published_body(self):
        bridge = Bridge()
        milk = PublishedBody(name="world/milk.stl")
        bridge.publish_bodies({"milk.stl": milk})

        assert bridge.object_body("milk.stl") is milk

    def test_object_body_is_none_for_an_unpublished_key(self):
        assert Bridge().object_body("milk.stl") is None

    def test_status_reports_no_demo_before_attaching(self):
        status = Bridge().status()
        assert status["running"] is False
        assert status["robot"] is None
        assert status["sequenceNumber"] == 0

    def test_the_robot_parts_are_read_off_the_live_annotations_when_asked(self):
        """
        The bridge keeps the robot's sem_dt annotations, not a snapshot of them, so a
        part attached after the last bind is still published.
        """
        arm = ArmPart(bodies=[NamedBody("pr2/l_upper_arm_link")])
        bridge = Bridge()
        bridge.robot = OneArmedRobot(arm=arm)
        assert bridge.status()["partAnnotations"] == [
            {
                "name": "ArmPart",
                "role": "arm",
                "side": None,
                "links": ["l_upper_arm_link"],
                "attachedTo": None,
            }
        ]

        arm.end_effector = EndEffectorPart(bodies=[NamedBody("pr2/l_gripper_link")])
        assert [
            annotation["name"] for annotation in bridge.status()["partAnnotations"]
        ] == ["ArmPart", "EndEffectorPart"]


# %% world-driven overlay discovery
def world_with(*bodies: Body) -> World:
    """
    A real world containing the given bodies, each fixed to a shared root.

    :param bodies: The bodies the world is built from.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    with world.modify_world():
        world.add_body(root)
        for body in bodies:
            world.add_connection(FixedConnection(parent=root, child=body))
    return world


def shaped_body(prefix: str, name: str) -> Body:
    """
    A body carrying one visual box shape, so the overlay publishes it.

    :param prefix: The body's namespace prefix.
    :param name: The body's local name.
    """
    return Body(
        name=PrefixedName(name, prefix=prefix),
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )


class TestWorldDrivenDiscovery:
    """
    ``bind`` publishes the demo's objects — the mesh-named bodies that spawn, get
    carried and disappear mid-run.

    Every other body is rendered by the scene bundle the viewer loads once, so the
    overlay must not duplicate it.
    """

    def test_a_scene_body_stays_out_of_the_overlay(self):
        """
        A body without a mesh-file name is part of the bundled scene the viewer loads
        once, however it was built.
        """
        bridge = Bridge()
        bridge.world = world_with(shaped_body("montessori", "board"))

        bridge.bind()

        assert bridge.object_keys() == []

    def test_a_mesh_named_body_is_published_even_without_shapes(self):
        bridge = Bridge()
        bridge.world = world_with(Body(name=PrefixedName("milk.stl", prefix="world")))

        bridge.bind()

        assert bridge.object_keys() == ["milk.stl"]

    def test_snapshot_streams_the_pose_of_every_published_body(self):
        bridge = Bridge()
        bridge.world = world_with(
            Body(
                name=PrefixedName("milk.stl", prefix="world"),
                visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
            )
        )

        bridge.bind()
        bridge.snapshot()

        objects = bridge.get_state()["objects"]
        assert objects["milk.stl"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class TestBundleSignature:
    def test_attaching_a_world_changes_the_signature(self):
        """
        The viewer compares the status signature against its loaded bundle's to notice
        that the demo switched worlds mid-run.
        """
        bridge = Bridge()
        before = bridge.status()["bundleSignature"]

        bridge.attach(world_with(shaped_body("montessori", "board")))

        after = bridge.status()["bundleSignature"]
        assert after != before
        assert after == bridge.bundle_signature()

    def test_a_new_scene_body_changes_the_signature(self):
        bridge = Bridge()
        bridge.attach(world_with(shaped_body("montessori", "board")))
        before = bridge.bundle_signature()

        world = bridge.world
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=world.root, child=shaped_body("montessori", "tray")
                )
            )
        bridge.observe_model_change()

        assert bridge.bundle_signature() != before

    def test_reparenting_an_overlay_object_keeps_the_signature(self):
        """
        A demo re-parents a grasped object on every pick and place; the object is
        rendered by the overlay, not the bundle, so the viewer must not reload the scene
        for it.
        """
        bridge = Bridge()
        board = shaped_body("montessori", "board")
        milk = Body(name=PrefixedName("milk.stl", prefix="world"))
        world = World()
        root = Body(name=PrefixedName("root", prefix="world"))
        with world.modify_world():
            world.add_body(root)
            world.add_connection(FixedConnection(parent=root, child=board))
            world.add_connection(FixedConnection(parent=root, child=milk))
        bridge.attach(world)
        before = bridge.bundle_signature()

        with world.modify_world():
            world.remove_connection(milk.parent_connection)
            world.add_connection(FixedConnection(parent=board, child=milk))
        bridge.observe_model_change()

        assert bridge.bundle_signature() == before

    def test_the_model_version_counts_attachments_and_model_changes(self):
        bridge = Bridge()
        assert bridge.status()["modelVersion"] == 0

        bridge.attach(world_with(shaped_body("montessori", "board")))
        bridge.observe_model_change()

        assert bridge.status()["modelVersion"] == 2


class TestShapeCatalogEntries:
    """
    A shape-published body's catalog entry carries every shape as the viewer builds it:

    kind, dimensions, colour and the shape's local pose within the body.
    """

    def test_primitive_shapes_carry_dimensions_colors_and_local_poses(self):
        """
        Each native primitive retains its appearance and body-relative pose.
        """
        body = Body(
            name=PrefixedName("tower", prefix="scene"),
            visual=ShapeCollection(
                shapes=[
                    Box(
                        scale=Scale(0.2, 0.3, 0.4),
                        color=Color(0.8, 0.2, 0.2),
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            0.1, 0.0, 0.05
                        ),
                    ),
                    Cylinder(width=0.1, height=0.3, color=Color(0.0, 0.5, 1.0)),
                    Sphere(radius=0.05),
                ]
            ),
        )
        bridge = Bridge()
        bridge.publish_bodies({"scene/tower": body})

        entry = bridge.object_catalog()[0]

        assert isinstance(bridge.object_metadata[0].shapes, ShapeCollection)
        assert entry["color"] == "#cc3333"
        box, cylinder, sphere = entry["shapes"]
        assert box["kind"] == "box"
        assert box["size"] == [0.2, 0.3, 0.4]
        assert box["color"] == "#cc3333"
        assert box["position"] == [0.1, 0.0, 0.05]
        assert box["quaternion"] == [0.0, 0.0, 0.0, 1.0]
        assert cylinder["kind"] == "cylinder"
        assert cylinder["radius"] == 0.05
        assert cylinder["height"] == 0.3
        assert cylinder["color"] == "#0080ff"
        assert sphere["kind"] == "sphere"
        assert sphere["radius"] == 0.05

    def test_a_mesh_shape_is_served_from_its_exported_file(self, tmp_path):
        mesh_file = tmp_path / "board.obj"
        mesh_file.write_text("o board\n")
        body = Body(
            name=PrefixedName("board", prefix="montessori"),
            visual=ShapeCollection(
                shapes=[Mesh(filename=str(mesh_file), scale=Scale(1.0, 2.0, 3.0))]
            ),
        )
        bridge = Bridge()
        bridge.publish_bodies({"montessori/board": body})

        shape = bridge.object_catalog()[0]["shapes"][0]

        serve_key = "montessori/board#0"
        assert shape["kind"] == "mesh"
        assert shape["format"] == "obj"
        assert shape["mesh"] == "/mesh?key=" + urllib.parse.quote(serve_key, safe="")
        assert shape["scale"] == [1.0, 2.0, 3.0]
        assert bridge.mesh_path(serve_key) == str(mesh_file)

    def test_a_mesh_shape_whose_file_vanished_becomes_a_default_box(self):
        body = Body(
            name=PrefixedName("board", prefix="montessori"),
            visual=ShapeCollection(shapes=[Mesh(filename="/gone/board.obj")]),
        )
        bridge = Bridge()
        bridge.publish_bodies({"montessori/board": body})

        shape = bridge.object_catalog()[0]["shapes"][0]

        assert shape["kind"] == "box"
        assert shape["size"] == list(Bridge.DEFAULT_OBJECT_SIZE)

    def test_collision_shapes_stand_in_when_a_body_has_no_visual_ones(self):
        """
        The catalog renders native collision geometry when visual geometry is absent.
        """
        body = Body(
            name=PrefixedName("guard", prefix="scene"),
            collision=ShapeCollection(shapes=[Box(scale=Scale(0.5, 0.5, 0.5))]),
        )
        bridge = Bridge()
        bridge.publish_bodies({"scene/guard": body})

        entry = bridge.object_catalog()[0]

        assert isinstance(bridge.object_metadata[0].shapes, ShapeCollection)
        assert entry["shapes"][0]["size"] == [0.5, 0.5, 0.5]


# %% motion statechart
@dataclass
class ChartNode:
    """
    A statechart node, of which the bridge reads index, name and parent index.
    """

    index: int
    name: str
    parent_node_index: Optional[int] = None


@dataclass
class ChartTransition:
    """
    An edge between statechart nodes, carrying the transition kind's name.
    """

    kind: Any


@dataclass
class TransitionKind:
    """
    The named kind of a statechart transition.
    """

    name: str


@dataclass
class TransitionGraph:
    """
    The rustworkx-shaped graph interface the chart serializer walks.
    """

    nodes: List[ChartNode]
    edges: List[tuple]

    def edge_index_map(self) -> Dict[int, tuple]:
        return dict(enumerate(self.edges))

    def get_node_data(self, index: int) -> ChartNode:
        return self.nodes[index]


@dataclass
class NodeStateVector:
    """
    A per-node state vector, indexed by node index.
    """

    data: List[float]


@dataclass
class ObservedStatechart:
    """
    A compiled motion statechart with its live life-cycle and observation vectors.
    """

    nodes: List[ChartNode]
    rx_graph: TransitionGraph
    life_cycle_state: NodeStateVector
    observation_state: NodeStateVector


def make_chart(life_cycle=(1, 1, 0), observation=(0.5, 0.5, 0.0)) -> ObservedStatechart:
    """
    A three-node statechart: a goal containing a motion, plus a monitor.
    """
    nodes = [
        ChartNode(0, "Goal"),
        ChartNode(1, "MoveJoints", 0),
        ChartNode(2, "JointGoalReached"),
    ]
    return ObservedStatechart(
        nodes=nodes,
        rx_graph=TransitionGraph(
            nodes=nodes,
            edges=[
                (0, 1, ChartTransition(kind=TransitionKind("START"))),
                (1, 2, ChartTransition(kind=TransitionKind("END"))),
            ],
        ),
        life_cycle_state=NodeStateVector(data=list(life_cycle)),
        observation_state=NodeStateVector(data=list(observation)),
    )


class TestChartEdgeEntry:
    def test_to_payload_renames_source_and_target_to_from_and_to(self):
        edge = ChartEdgeEntry(
            source="chart_node_0", target="chart_node_1", kind="START"
        )
        assert edge.to_payload() == {
            "from": "chart_node_0",
            "to": "chart_node_1",
            "kind": "START",
        }


class TestChartSnapshot:
    def test_structure_and_states(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        chart = bridge.get_chart()
        assert [node["life_cycle"] for node in chart["nodes"]] == [
            "RUNNING",
            "RUNNING",
            "NOT_STARTED",
        ]
        assert [node["observation"] for node in chart["nodes"]] == [
            "UNKNOWN",
            "UNKNOWN",
            "FALSE",
        ]
        assert chart["nodes"][1]["parent"] == "chart_node_0"
        assert chart["edges"] == [
            {"from": "chart_node_0", "to": "chart_node_1", "kind": "START"},
            {"from": "chart_node_1", "to": "chart_node_2", "kind": "END"},
        ]

    def test_lifecycle_update_keeps_signature(self):
        bridge = Bridge()
        chart = make_chart()
        bridge.observe_chart(chart)
        signature = bridge.get_chart()["signature"]
        chart.life_cycle_state.data = [3, 3, 3]
        bridge.observe_chart(chart)
        assert bridge.get_chart()["signature"] == signature
        assert [node["life_cycle"] for node in bridge.get_chart()["nodes"]] == [
            LifeCycleValues.SUCCEEDED.name
        ] * 3

    def test_new_chart_replaces_structure(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        signature = bridge.get_chart()["signature"]
        single_node = [ChartNode(0, "OtherGoal")]
        bridge.observe_chart(
            ObservedStatechart(
                nodes=single_node,
                rx_graph=TransitionGraph(nodes=single_node, edges=[]),
                life_cycle_state=NodeStateVector(data=[1]),
                observation_state=NodeStateVector(data=[1.0]),
            )
        )
        chart = bridge.get_chart()
        assert chart["signature"] != signature
        assert len(chart["nodes"]) == 1
        assert chart["nodes"][0]["observation"] == "TRUE"

    def test_trinary_observation_thresholds(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart(observation=(0.0, 0.5, 1.0)))
        assert [node["observation"] for node in bridge.get_chart()["nodes"]] == [
            "FALSE",
            "UNKNOWN",
            "TRUE",
        ]

    def test_observation_change_alone_is_published(self):
        """
        A monitor flipping its observation must reach the viewer even while every node's
        life cycle stays the same.
        """
        bridge = Bridge()
        chart = make_chart(life_cycle=(1, 1, 1), observation=(0.5, 0.5, 0.5))
        bridge.observe_chart(chart)
        chart.observation_state.data = [0.5, 0.5, 1.0]
        bridge.observe_chart(chart)
        assert [node["observation"] for node in bridge.get_chart()["nodes"]] == [
            "UNKNOWN",
            "UNKNOWN",
            "TRUE",
        ]
