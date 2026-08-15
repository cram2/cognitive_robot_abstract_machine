"""
Unit tests for the live bridge's serializers and its viewer-facing accessors.

The bridge is exercised against mimics of the duck-typed interfaces it reads, so no
coraplex import is needed. What is covered is the interesting logic: bottom-up status
aggregation in the plan tree, the pinning of each motion node's reported status,
statechart signatures that let the frontend distinguish "re-colour only" from "rebuild",
and the queue that carries viewer drags onto the simulation thread.
"""

from __future__ import annotations

import math
import threading
import urllib.parse
from dataclasses import dataclass, field

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
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
from cramera.live.bridge import (
    Bridge,
    ChartEdgeEntry,
    MalformedMoveRequest,
    MoveRequest,
    ROBOT_BASE_KEY,
    TaskStatusName,
)

from .test_robot_parts import ArmPart, EndEffectorPart, NamedBody, OneArmedRobot


# %% mimics of the interfaces the bridge reads
@dataclass
class ReportedStatus:
    """
    A plan node's own status, which the bridge reads as ``node.status.name``.
    """

    name: str


@dataclass
class ActionDescription:
    """
    A designator describing what an action acts on, as the bridge inspects it.

    Attributes are set only when present, mirroring real designators whose fields differ
    per action type.
    """

    def __init__(self, target: Optional[str] = None, arm: Optional[str] = None):
        if target is not None:
            self.acted_on = NamedWorldEntity(name="world/" + target)
        if arm is not None:
            self.arm = arm


@dataclass
class NamedWorldEntity:
    """
    Something in the world that carries a prefixed name.
    """

    name: str


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


def make_plan_node(
    kind: str,
    status: str = TaskStatusName.CREATED,
    designator: Optional[ActionDescription] = None,
    children: tuple = (),
) -> Any:
    """
    A plan-node mimic of the given class name, as the serializer walks it.
    """
    node_class = type(kind, (object,), {})
    node = node_class()
    node.status = ReportedStatus(name=status)
    node.designator = designator
    node.children = list(children)
    node.parent_action_node = None
    return node


@pytest.fixture()
def plan_bridge():
    """
    A bridge bound to a small plan: root -> action -> [condition, motion].
    """
    bridge = Bridge()
    motion = make_plan_node(
        "MotionNode", designator=ActionDescription(target="milk.stl")
    )
    condition = make_plan_node("ConditionNode")
    action = make_plan_node(
        "ActionNode",
        designator=ActionDescription(arm="LEFT"),
        children=[condition, motion],
    )
    root = make_plan_node(
        "SequentialNode", status=TaskStatusName.SUCCEEDED, children=[action]
    )
    bridge.publish_bodies(
        {
            "milk.stl": PublishedBody(name="world/milk.stl"),
            ROBOT_BASE_KEY: PublishedBody(name="world/base_link"),
        }
    )
    bridge.begin_plan(PlanWithRoot(root=root))
    return bridge, root, action, condition, motion


@dataclass
class PlanWithRoot:
    """
    A coraplex plan, of which the bridge only reads the root node.
    """

    root: Any


def nodes_by_kind(bridge: Bridge) -> Dict[str, Dict[str, Any]]:
    """
    The published plan nodes, indexed by the class name they were built from.
    """
    return {node["kind"]: node for node in bridge.get_plan()["nodes"]}


def end_motion(bridge: Bridge, node: Any, status: TaskStatusName) -> None:
    """
    Report a motion node's end with the given final status.

    The node's own status is restored to ``CREATED`` afterwards, so what the tests
    observe is the status the bridge pinned, not the mimic's attribute.

    :param bridge: The bridge observing the plan.
    :param node: The plan-node mimic that ended.
    :param status: The status the node reports while ending.
    """
    node.status = ReportedStatus(name=status)
    bridge.observe_motion_ended(node)
    node.status = ReportedStatus(name=TaskStatusName.CREATED)


# %% plan tree
class TestPlanSnapshot:
    def test_running_task_bubbles_up(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge.observe_motion_started(motion)
        bridge.snapshot_plan()
        nodes = nodes_by_kind(bridge)
        assert nodes["MotionNode"]["status"] == TaskStatusName.RUNNING
        assert nodes["MotionNode"]["derived"] is True
        assert nodes["ActionNode"]["status"] == TaskStatusName.RUNNING

    def test_each_node_carries_the_colour_group_of_its_kind(self, plan_bridge):
        """
        The bridge classifies plan nodes, so the viewer does not keep its own copy of
        the kind-to-group table.
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

    def test_designator_metadata_is_serialized(self, plan_bridge):
        bridge, *_ = plan_bridge
        nodes = nodes_by_kind(bridge)
        assert nodes["MotionNode"]["target"] == "milk.stl"
        assert nodes["ActionNode"]["arm"] == "LEFT"

    def test_real_status_wins_over_derivation(self, plan_bridge):
        bridge, *_ = plan_bridge
        assert nodes_by_kind(bridge)["SequentialNode"]["status"] == (
            TaskStatusName.SUCCEEDED
        )
        assert nodes_by_kind(bridge)["SequentialNode"]["derived"] is False

    def test_partially_done_parent_is_running_not_succeeded(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        end_motion(bridge, motion, TaskStatusName.SUCCEEDED)
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == TaskStatusName.RUNNING

    def test_fully_done_parent_is_succeeded(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        end_motion(bridge, motion, TaskStatusName.SUCCEEDED)
        end_motion(bridge, condition, TaskStatusName.SUCCEEDED)
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == TaskStatusName.SUCCEEDED

    def test_failure_outranks_done_sibling(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        end_motion(bridge, motion, TaskStatusName.SUCCEEDED)
        end_motion(bridge, condition, TaskStatusName.FAILED)
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == TaskStatusName.FAILED

    def test_signature_is_stable_across_status_changes(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge.observe_motion_started(motion)
        bridge.snapshot_plan()
        while_running = bridge.get_plan()["signature"]
        end_motion(bridge, motion, TaskStatusName.SUCCEEDED)
        assert bridge.get_plan()["signature"] == while_running

    def test_structurally_identical_nodes_keep_separate_statuses(self):
        """
        Two steps that compare equal must not share one status entry.

        coraplex's designator nodes compare by field value, so a status keyed by
        equality would leak from one step of a plan to an identical one.
        """
        bridge = Bridge()
        first = make_plan_node("MotionNode")
        second = make_plan_node("MotionNode")
        root = make_plan_node("SequentialNode", children=[first, second])
        bridge.begin_plan(PlanWithRoot(root=root))
        end_motion(bridge, first, TaskStatusName.FAILED)
        statuses = [
            node["status"]
            for node in bridge.get_plan()["nodes"]
            if node["kind"] == "MotionNode"
        ]
        assert statuses == [TaskStatusName.FAILED, TaskStatusName.CREATED]

    def test_a_new_plan_drops_the_previous_progress(self, plan_bridge):
        """
        A node's pinned status must not survive into the next plan.
        """
        bridge, root, action, condition, motion = plan_bridge
        end_motion(bridge, motion, TaskStatusName.SUCCEEDED)
        bridge.begin_plan(PlanWithRoot(root=motion))
        assert nodes_by_kind(bridge)["MotionNode"]["status"] == TaskStatusName.CREATED


# %% viewer -> world
class TestRunningStep:
    """
    What a recording labels each of its ticks with: the action the plan is performing
    right now (see cramera.live.recording_segments).
    """

    def test_nothing_is_reported_before_anything_runs(self, plan_bridge):
        bridge, _, _, _, _ = plan_bridge
        bridge.snapshot_plan()

        assert bridge.running_step() is None

    def test_the_running_action_is_reported(self, plan_bridge):
        bridge, _, _, _, motion = plan_bridge
        bridge.observe_motion_started(motion)
        bridge.snapshot_plan()

        assert bridge.running_step() == nodes_by_kind(bridge)["ActionNode"]["label"]

    def test_a_finished_action_is_no_longer_reported(self):
        motion = make_plan_node("MotionNode")
        action = make_plan_node("ActionNode", children=[motion])
        bridge = Bridge()
        bridge.begin_plan(PlanWithRoot(root=action))
        bridge.observe_motion_started(motion)
        bridge.snapshot_plan()
        end_motion(bridge, motion, TaskStatusName.SUCCEEDED)
        bridge.snapshot_plan()

        assert bridge.running_step() is None

    def test_a_running_motion_is_not_reported_as_the_step(self):
        """
        Only actions name a step; a motion node is one step's implementation, not a step
        of its own.
        """
        bridge = Bridge()
        motion = make_plan_node("MotionNode")
        bridge.begin_plan(PlanWithRoot(root=motion))
        bridge.observe_motion_started(motion)
        bridge.snapshot_plan()

        assert bridge.running_step() is None


class TestMoveRequestValidation:
    def test_a_complete_payload_is_accepted(self):
        move = MoveRequest.from_payload(
            {
                "object": "milk.stl",
                "position": [1.0, 2.0, 3.0],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
                "final": True,
            }
        )
        assert move.object_key == "milk.stl"
        assert move.position == [1.0, 2.0, 3.0]
        assert move.quaternion == [0.0, 0.0, 0.0, 1.0]
        assert move.is_final is True

    def test_orientation_is_optional(self):
        move = MoveRequest.from_payload({"object": "milk.stl", "position": [0, 0, 0]})
        assert move.quaternion is None
        assert move.is_final is False

    def test_integers_are_accepted_as_coordinates(self):
        move = MoveRequest.from_payload({"object": "milk.stl", "position": [1, 2, 3]})
        assert move.position == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize(
        "payload",
        [
            {"position": [0, 0, 0]},
            {"object": "", "position": [0, 0, 0]},
            {"object": 5, "position": [0, 0, 0]},
            {"object": "milk.stl"},
            {"object": "milk.stl", "position": [0, 0]},
            {"object": "milk.stl", "position": "here"},
            {"object": "milk.stl", "position": [0, 0, "up"]},
            {"object": "milk.stl", "position": [0, 0, True]},
            {"object": "milk.stl", "position": [0, 0, float("nan")]},
            {"object": "milk.stl", "position": [0, 0, float("inf")]},
            {"object": "milk.stl", "position": [0, 0, 0], "quaternion": [0, 0, 1]},
        ],
    )
    def test_unusable_payloads_are_rejected(self, payload):
        """
        Bad input must fail at the boundary, not inside the simulation tick.
        """
        with pytest.raises(MalformedMoveRequest):
            MoveRequest.from_payload(payload)


class TestQueuedMoves:
    def test_applying_without_a_world_is_harmless(self):
        """
        A drag that arrives before a demo attaches must not raise on the sim thread.
        """
        bridge = Bridge()
        bridge.queue_move(MoveRequest(object_key="milk.stl", position=[1.0, 2.0, 3.0]))
        bridge.apply_moves()

    def test_a_move_for_an_unknown_object_is_ignored(self):
        """
        An object the world does not have must be skipped, not raise.
        """
        bridge = Bridge()
        bridge.world = object()
        bridge.publish_bodies({})
        bridge.queue_move(MoveRequest(object_key="ghost.stl", position=[0.0, 0.0, 0.0]))
        bridge.apply_moves()


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


class TestApplyMove:
    """
    ``_apply_move`` writes a viewer drag into a free-floating object's connection.
    """

    def test_orientation_is_kept_exact_when_the_drag_omits_it(self):
        """
        A position-only drag must not round-trip the object's orientation through the
        5-decimal-place floats ``rounded_pose`` produces for the viewer feed; that would
        nudge the true orientation on every such drag.
        """
        world, connection, body = make_free_floating_object()
        half_angle = 0.123456789 / 2
        connection.origin = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(x=1.0, y=2.0, z=3.0),
            RotationMatrix.from_quaternion(
                Quaternion(z=math.sin(half_angle), w=math.cos(half_angle))
            ),
            reference_frame=world.root,
        )
        expected_orientation = body.global_pose.to_quaternion().to_np()

        bridge = Bridge()
        bridge.world = world
        bridge._apply_move(
            MoveRequest(object_key="milk.stl", position=[4.0, 5.0, 6.0]), body
        )

        assert np.allclose(
            body.global_pose.to_quaternion().to_np(),
            expected_orientation,
            rtol=0,
            atol=1e-8,
        )
        assert body.global_pose.to_position().to_np()[:3].tolist() == [4.0, 5.0, 6.0]

    def test_a_given_quaternion_is_applied(self):
        world, connection, body = make_free_floating_object()

        bridge = Bridge()
        bridge.world = world
        bridge._apply_move(
            MoveRequest(
                object_key="milk.stl",
                position=[1.0, 2.0, 3.0],
                quaternion=[0.0, 0.0, 1.0, 0.0],
            ),
            body,
        )

        pose = body.global_pose
        assert pose.to_position().to_np()[:3].tolist() == [1.0, 2.0, 3.0]
        assert np.allclose(pose.to_quaternion().to_np(), [0.0, 0.0, 1.0, 0.0])


# %% what the HTTP layer reads
class TestViewerAccessors:
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
        bridge = Bridge()
        bridge.publish_bodies({"blob.stl": PublishedBody(name="world/blob.stl")})
        assert bridge.object_catalog()[0]["size"] == list(Bridge.DEFAULT_OBJECT_SIZE)

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

        assert entry["kind"] == "shapes"
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
        body = Body(
            name=PrefixedName("guard", prefix="scene"),
            collision=ShapeCollection(shapes=[Box(scale=Scale(0.5, 0.5, 0.5))]),
        )
        bridge = Bridge()
        bridge.publish_bodies({"scene/guard": body})

        entry = bridge.object_catalog()[0]

        assert entry["kind"] == "shapes"
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
            "DONE"
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
