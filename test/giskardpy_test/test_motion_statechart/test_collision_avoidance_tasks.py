import json
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pytest

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
)
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    ExternalCollisionDistanceMonitor,
    SelfCollisionDistanceMonitor,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.collision_checking.collision_manager import CollisionConsumer
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
    AvoidExternalCollisions,
    AvoidAllCollisions,
    AllowAllCollisions,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import (
    Cylinder,
    Box,
    Scale,
    Color,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
)


def test_external_collision_avoidance(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    msc = MotionStatechart()
    msc.add_nodes(
        [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        violated_distance=0.0,
                        body_group_a=[tip],
                        body_group_b=[env1],
                    )
                ]
            ),
            distance_violated := ExternalCollisionDistanceMonitor(
                body=robot.root, threshold=0.049
            ),
            CartesianPose(
                root_link=cylinder_bot_world.root,
                tip_link=tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_world.root
                ),
            ),
            ExternalCollisionAvoidance(robot=robot),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))
    msc.add_node(CancelMotion.when_true(distance_violated))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_world)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(
        MotionStatechartContext(world=cylinder_bot_world),
        pacer=SimulationPacer(real_time_factor=2),
    )
    kin_sim.compile(motion_statechart=msc_copy)

    kin_sim.tick_until_end(500)
    collisions = kin_sim.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    assert collisions.contacts[0].distance > 0.049
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_external_collision_avoidance_battle():
    strong_robot_world = World()
    with strong_robot_world.modify_world():
        strong = Body(
            name=PrefixedName("strong"),
            collision=ShapeCollection(
                [Cylinder(width=0.1, height=0.1, color=Color(R=1, G=0, B=0, A=1))]
            ),
        )
        strong_robot_world.add_body(strong)
        strong_robot_sa = MinimalRobot.from_world(strong_robot_world)

    weak_robot_world = World()
    with weak_robot_world.modify_world():
        weak = Body(
            name=PrefixedName("weak"),
            collision=ShapeCollection(
                [Cylinder(width=0.1, height=0.1, color=Color(R=0, G=0, B=1, A=1))]
            ),
        )
        weak_robot_world.add_body(weak)
        weak_robot_sa = MinimalRobot.from_world(weak_robot_world)

    world = World()

    with world.modify_world():
        wall = Body(
            name=PrefixedName("wall"),
            collision=ShapeCollection([Box(scale=Scale(x=0.1, y=10, z=0.1))]),
        )
        map = Body(name=PrefixedName("map"))
        world.add_connection(
            FixedConnection(
                parent=map,
                child=wall,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5
                ),
            )
        )
        strong_odom = Body(name=PrefixedName("strong_odom"))
        world.add_connection(
            FixedConnection(
                parent=map,
                child=strong_odom,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.5
                ),
            )
        )
        weak_odom = Body(name=PrefixedName("weak_odom"))
        world.add_connection(
            FixedConnection(
                parent=map,
                child=weak_odom,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-0.5
                ),
            )
        )

        # %% attach robots
        world.merge_world(
            strong_robot_world,
            omni1 := OmniDrive.create_with_dofs(
                world=world,
                parent=strong_odom,
                child=strong_robot_world.root,
            ),
        )
        strong = world.get_kinematic_structure_entity_by_id(strong.id)
        strong_robot_sa = world.get_semantic_annotation_by_id(strong_robot_sa.id)
        world.merge_world(
            weak_robot_world,
            omni2 := OmniDrive.create_with_dofs(
                world=world,
                parent=weak_odom,
                child=weak_robot_world.root,
            ),
        )
        weak = world.get_kinematic_structure_entity_by_id(weak.id)
        weak_robot_sa = world.get_semantic_annotation_by_id(weak_robot_sa.id)

        omni1.has_hardware_interface = True
        omni2.has_hardware_interface = True

    msc = MotionStatechart()
    msc.add_nodes(
        [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        violated_distance=0.0,
                        body_group_a=[weak],
                        body_group_b=[wall],
                    ),
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        violated_distance=0.0,
                        body_group_a=[strong],
                        body_group_b=[wall],
                    ),
                ]
            ),
            CartesianPose(
                root_link=map,
                tip_link=strong,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=strong
                ),
                weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
            ),
            CartesianPose(
                root_link=map,
                tip_link=weak,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=weak
                ),
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            ),
            ExternalCollisionAvoidance(robot=weak_robot_sa),
            ExternalCollisionAvoidance(robot=strong_robot_sa),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))
    # msc.add_node(CancelMotion.when_true(distance_violated))

    kin_sim = Executor(
        MotionStatechartContext(world=world),
        pacer=SimulationPacer(real_time_factor=1),
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end(500)

    assert (
        0.05
        > world.collision_manager.collision_detector.check_collision_between_bodies(
            body_a=strong, body_b=wall, distance=1
        ).distance
        > -0.01
    )

    assert (
        world.collision_manager.collision_detector.check_collision_between_bodies(
            body_a=weak, body_b=wall, distance=1
        ).distance
        > 0.049
    )


def test_external_collision_avoidance_with_weight_above_ca(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    msc = MotionStatechart()
    msc.add_nodes(
        [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        violated_distance=0.0,
                        body_group_a=[tip],
                        body_group_b=[env1],
                    )
                ]
            ),
            distance_violated := ExternalCollisionDistanceMonitor(
                body=robot.root, threshold=0.0
            ),
            CartesianPose(
                root_link=cylinder_bot_world.root,
                tip_link=tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_world.root
                ),
                weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
            ),
            ExternalCollisionAvoidance(robot=robot),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))
    msc.add_node(CancelMotion.when_true(distance_violated))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_world)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(
        MotionStatechartContext(world=cylinder_bot_world),
        pacer=SimulationPacer(real_time_factor=1),
    )
    kin_sim.compile(motion_statechart=msc_copy)

    kin_sim.tick_until_end(500)
    collisions = kin_sim.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    assert collisions.contacts[0].distance < 0.049
    assert collisions.contacts[0].distance > 0.0
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_update_collision_matrix_later(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    msc = MotionStatechart()
    msc.add_nodes(
        [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        violated_distance=0.0,
                        body_group_a=[tip],
                        body_group_b=[env1],
                    )
                ]
            ),
            ExternalCollisionAvoidance(robot=robot),
            cart_goal_reached := CartesianPose(
                root_link=cylinder_bot_world.root,
                tip_link=tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_world.root
                ),
            ),
            Sequence(
                [
                    LocalMinimumReached(),
                    UpdateTemporaryCollisionRules(
                        temporary_rules=[AllowAllCollisions()]
                    ),
                ]
            ),
        ]
    )
    msc.add_node(EndMotion.when_true(cart_goal_reached))

    kin_sim = Executor(
        MotionStatechartContext(world=cylinder_bot_world),
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end(500)
    msc.draw("muh.pdf")
    cylinder_bot_world.collision_manager.clear_temporary_rules()
    cylinder_bot_world.collision_manager.add_temporary_rule(AvoidAllCollisions())
    cylinder_bot_world.collision_manager.update_collision_matrix()
    collisions = kin_sim.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    assert collisions.contacts[0].distance < 0.049
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_consumer_cleanup_after_cancel(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    msc = MotionStatechart()
    msc.add_nodes(
        [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        violated_distance=0.0,
                        body_group_a=[tip],
                        body_group_b=[env1],
                    ),
                ]
            ),
            CartesianPose(
                root_link=cylinder_bot_world.root,
                tip_link=tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_world.root
                ),
            ),
            ExternalCollisionAvoidance(robot=robot),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(
        Sequence(
            [
                CountControlCycles(control_cycles=5),
                CancelMotion(exception=Exception("muh")),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(local_min))

    kin_sim = Executor(MotionStatechartContext(world=cylinder_bot_world))
    kin_sim.compile(motion_statechart=msc)

    with pytest.raises(Exception, match="muh"):
        kin_sim.tick_until_end(500)
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_multiple_external_collision_avoidance_motions(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")

    def run_motion(goal_x):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                UpdateTemporaryCollisionRules(
                    temporary_rules=[
                        AvoidCollisionBetweenGroups(
                            buffer_zone_distance=0.05,
                            violated_distance=0.0,
                            body_group_a=[tip],
                            body_group_b=[env1],
                        ),
                    ]
                ),
                CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=goal_x, reference_frame=cylinder_bot_world.root
                    ),
                ),
                ExternalCollisionAvoidance(robot=robot),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))

        kin_sim = Executor(MotionStatechartContext(world=cylinder_bot_world))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end(500)
        return kin_sim

    # First motion
    run_motion(1.0)
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0

    # Second motion
    run_motion(0.5)
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_self_collision_avoidance(self_collision_bot_world: World):

    robot = self_collision_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    l_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_tip")
    r_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_tip")
    l_thumb = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_thumb")
    r_thumb = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_thumb")

    msc = MotionStatechart()
    msc.add_nodes(
        [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.25,
                        violated_distance=0.0,
                        body_group_a={l_thumb},
                        body_group_b={r_thumb},
                    ),
                ]
            ),
            SelfCollisionAvoidance(robot=robot),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(self_collision_bot_world)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(MotionStatechartContext(world=self_collision_bot_world))
    kin_sim.compile(motion_statechart=msc_copy)

    # 4 because of the base nodes + 20 that are added by self collision avoidance + 1 for CancelMotion
    assert len(msc_copy.nodes) == 4 + 20 + 1

    kin_sim.tick_until_end(500)
    collisions = kin_sim.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    for contact in collisions.contacts:
        assert contact.distance > 0.249
    assert len(self_collision_bot_world.collision_manager.collision_consumers) == 0


def test_avoid_collision_go_around_corner(pr2_with_box):
    r_tip = pr2_with_box.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    robot = pr2_with_box.get_semantic_annotations_by_type(AbstractRobot)[0]

    msc = MotionStatechart()
    msc.add_node(
        UpdateTemporaryCollisionRules(
            temporary_rules=[
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1, violated_distance=0.0, robot=robot
                )
            ]
        )
    )
    msc.add_node(
        Sequence(
            [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        {
                            PR2Joint.RIGHT_ELBOW_FLEX: -1.29610152504,
                            PR2Joint.RIGHT_FOREARM_ROLL: -0.0301682323805,
                            PR2Joint.RIGHT_SHOULDER_LIFT: 1.20324921318,
                            PR2Joint.RIGHT_SHOULDER_PAN: -0.73456435706,
                            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.70790051778,
                            PR2Joint.RIGHT_WRIST_FLEX: -0.10001,
                            PR2Joint.RIGHT_WRIST_ROLL: 0.258268529825,
                            PR2Joint.LEFT_ELBOW_FLEX: -1.29610152504,
                            PR2Joint.LEFT_FOREARM_ROLL: 0.0301682323805,
                            PR2Joint.LEFT_SHOULDER_LIFT: 1.20324921318,
                            PR2Joint.LEFT_SHOULDER_PAN: 0.73456435706,
                            PR2Joint.LEFT_UPPER_ARM_ROLL: 0.70790051778,
                            PR2Joint.LEFT_WRIST_FLEX: -0.1001,
                            PR2Joint.LEFT_WRIST_ROLL: -0.258268529825,
                            PR2Joint.TORSO_LIFT: 0.2,
                            PR2Joint.HEAD_PAN: 0,
                            PR2Joint.HEAD_TILT: 0,
                            PR2Joint.LEFT_GRIPPER_LEFT_FINGER: 0.55,
                            PR2Joint.RIGHT_GRIPPER_LEFT_FINGER: 0.55,
                        },
                        world=pr2_with_box,
                    )
                ),
                Parallel(
                    [
                        CartesianPose(
                            root_link=pr2_with_box.root,
                            tip_link=r_tip,
                            goal_pose=Pose.from_xyz_axis_angle(
                                x=0.8,
                                y=-0.38,
                                z=0.84,
                                axis=Vector3.Y(),
                                angle=np.pi / 2.0,
                                reference_frame=pr2_with_box.root,
                            ),
                            weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                        ),
                        ExternalCollisionAvoidance(robot=robot),
                    ]
                ),
            ]
        )
    )
    msc.add_node(
        distance_violated := ExternalCollisionDistanceMonitor(
            body=pr2_with_box.get_kinematic_structure_entity_by_name(
                "r_gripper_palm_link"
            ),
            threshold=0.0,
        ),
    )
    msc.add_node(local_min := LocalMinimumReached())
    msc.add_node(EndMotion.when_true(local_min))
    msc.add_node(CancelMotion.when_true(distance_violated))

    kin_sim = Executor(MotionStatechartContext(world=pr2_with_box))
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end(500)


def test_avoid_self_collision_with_l_arm(pr2_with_box, rclpy_node):
    VizMarkerPublisher(
        _world=pr2_with_box, node=rclpy_node
    ).with_tf_and_collision_visualization()
    r_tip = pr2_with_box.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    l_forearm_link = pr2_with_box.get_kinematic_structure_entity_by_name(
        "l_forearm_link"
    )
    r_palm_link = pr2_with_box.get_kinematic_structure_entity_by_name(
        "r_gripper_palm_link"
    )
    base_footprint = pr2_with_box.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )
    robot = pr2_with_box.get_semantic_annotations_by_type(AbstractRobot)[0]

    msc = MotionStatechart()
    msc.add_node(
        Sequence(
            [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        {
                            PR2Joint.RIGHT_ELBOW_FLEX: -1.43286344265,
                            PR2Joint.RIGHT_FOREARM_ROLL: -1.26465060073,
                            PR2Joint.RIGHT_SHOULDER_LIFT: 0.47990329056,
                            PR2Joint.RIGHT_SHOULDER_PAN: -0.281272240139,
                            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.528415402668,
                            PR2Joint.RIGHT_WRIST_FLEX: -1.18811419869,
                            PR2Joint.RIGHT_WRIST_ROLL: 2.26884630124,
                            PR2Joint.LEFT_ELBOW_FLEX: 0.0,
                            PR2Joint.LEFT_FOREARM_ROLL: 0.0,
                            PR2Joint.LEFT_SHOULDER_LIFT: 0.0,
                            PR2Joint.LEFT_SHOULDER_PAN: 0.0,
                            PR2Joint.LEFT_UPPER_ARM_ROLL: 0.0,
                            PR2Joint.LEFT_WRIST_FLEX: 0.0,
                            PR2Joint.LEFT_WRIST_ROLL: 0.0,
                        },
                        world=pr2_with_box,
                    )
                ),
                Parallel(
                    [
                        CartesianPose(
                            root_link=base_footprint,
                            tip_link=r_tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                0.2, reference_frame=r_tip
                            ),
                            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                        ),
                        SelfCollisionAvoidance(robot=robot),
                    ],
                ),
            ],
        ),
    )
    msc.add_node(
        contact := SelfCollisionDistanceMonitor(
            body_a=r_palm_link, body_b=l_forearm_link, threshold=0.01
        )
    )
    msc.add_node(local_min := LocalMinimumReached())
    msc.add_node(EndMotion.when_true(local_min))
    msc.add_node(CancelMotion.when_true(contact))

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_with_box,
            qp_controller_config=QPControllerConfig(
                target_frequency=100,
                prediction_horizon=30,
            ),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    assert len(msc.nodes) == 76

    kin_sim.tick_until_end(500)


# %% collision check counting


@dataclass
class CollisionCheckCountingObserver(CollisionConsumer):
    """
    Passive collision consumer that records how often collisions were computed.

    Being passive, it never causes collision checking itself, so its count reflects only
    the checks that other consumers required.
    """

    collision_check_count: int = field(init=False, default=0)
    """
    Number of collision computations observed while registered.
    """

    def on_compute_collisions(self, collision_results: CollisionCheckingResult):
        self.collision_check_count += 1

    def on_world_model_update(self, world: World):
        pass

    def on_collision_matrix_update(self):
        pass


def _build_arms_crossing_statechart(
    world: World, collision_avoidance: bool
) -> MotionStatechart:
    """
    Builds a statechart that drives the right gripper into the left arm.

    :param world: The PR2 world the goal refers to.
    :param collision_avoidance: Whether a self collision avoidance node is part of the
        goal.
    """
    r_tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    base_footprint = world.get_kinematic_structure_entity_by_name("base_footprint")

    msc = MotionStatechart()
    msc.add_node(
        goal := Sequence(
            [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        {
                            PR2Joint.RIGHT_ELBOW_FLEX: -1.43286344265,
                            PR2Joint.RIGHT_FOREARM_ROLL: -1.26465060073,
                            PR2Joint.RIGHT_SHOULDER_LIFT: 0.47990329056,
                            PR2Joint.RIGHT_SHOULDER_PAN: -0.281272240139,
                            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.528415402668,
                            PR2Joint.RIGHT_WRIST_FLEX: -1.18811419869,
                            PR2Joint.RIGHT_WRIST_ROLL: 2.26884630124,
                            PR2Joint.LEFT_ELBOW_FLEX: 0.0,
                            PR2Joint.LEFT_FOREARM_ROLL: 0.0,
                            PR2Joint.LEFT_SHOULDER_LIFT: 0.0,
                            PR2Joint.LEFT_SHOULDER_PAN: 0.0,
                            PR2Joint.LEFT_UPPER_ARM_ROLL: 0.0,
                            PR2Joint.LEFT_WRIST_FLEX: 0.0,
                            PR2Joint.LEFT_WRIST_ROLL: 0.0,
                        },
                        world=world,
                    )
                ),
                CartesianPose(
                    root_link=base_footprint,
                    tip_link=r_tip,
                    goal_pose=Pose.from_xyz_rpy(0.2, reference_frame=r_tip),
                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                ),
            ],
        ),
    )
    msc.add_node(EndMotion.when_true(goal))
    if collision_avoidance:
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
        msc.add_node(SelfCollisionAvoidance(robot=robot))
    return msc


def _run_and_count_collision_checks(
    world: World, msc: MotionStatechart
) -> CollisionCheckCountingObserver:
    """
    Executes the statechart and returns the observer that counted the collision checks.
    """
    observer = CollisionCheckCountingObserver()
    world.collision_manager.add_collision_consumer(observer)
    executor = Executor(
        MotionStatechartContext(
            world=world,
            qp_controller_config=QPControllerConfig(
                target_frequency=100,
                prediction_horizon=30,
            ),
        )
    )
    executor.compile(motion_statechart=msc)
    executor.tick_until_end(500)
    world.collision_manager.remove_collision_consumer(observer)
    return observer


def test_collisions_are_not_computed_without_collision_nodes(pr2_with_box):
    msc = _build_arms_crossing_statechart(pr2_with_box, collision_avoidance=False)

    for node in msc.nodes:
        assert not isinstance(node, SelfCollisionAvoidance)
        assert not isinstance(node, ExternalCollisionAvoidance)

    observer = _run_and_count_collision_checks(pr2_with_box, msc)
    assert observer.collision_check_count == 0


def test_collisions_are_computed_with_collision_avoidance_node(pr2_with_box):
    msc = _build_arms_crossing_statechart(pr2_with_box, collision_avoidance=True)

    observer = _run_and_count_collision_checks(pr2_with_box, msc)
    assert observer.collision_check_count > 0


def test_collisions_are_computed_for_a_distance_monitor_alone(pr2_with_box):
    msc = _build_arms_crossing_statechart(pr2_with_box, collision_avoidance=False)
    msc.add_node(
        ExternalCollisionDistanceMonitor(
            body=pr2_with_box.get_kinematic_structure_entity_by_name(
                "r_gripper_palm_link"
            ),
            threshold=0.0,
        )
    )

    observer = _run_and_count_collision_checks(pr2_with_box, msc)
    assert observer.collision_check_count > 0


def test_hard_constraints_violated(cylinder_bot_world: World):
    root = cylinder_bot_world.root
    with cylinder_bot_world.modify_world():
        env2 = Body(
            name=PrefixedName("environment2"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
        )
        env_connection = FixedConnection(
            parent=root,
            child=env2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.75
            ),
        )
        cylinder_bot_world.add_connection(env_connection)

        env3 = Body(
            name=PrefixedName("environment3"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
        )
        env_connection = FixedConnection(
            parent=root,
            child=env3,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                1.25
            ),
        )
        cylinder_bot_world.add_connection(env_connection)
        env4 = Body(
            name=PrefixedName("environment4"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
        )
        env_connection = FixedConnection(
            parent=root,
            child=env4,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=-0.25
            ),
        )
        cylinder_bot_world.add_connection(env_connection)
        env5 = Body(
            name=PrefixedName("environment5"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
        )
        env_connection = FixedConnection(
            parent=root,
            child=env5,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=0.25
            ),
        )
        cylinder_bot_world.add_connection(env_connection)

    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

    msc = MotionStatechart()
    msc.add_node(
        Sequence(
            [
                UpdateTemporaryCollisionRules(
                    temporary_rules=[
                        AvoidAllCollisions(
                            buffer_zone_distance=0.05,
                            violated_distance=0.0,
                        )
                    ]
                ),
                SetOdometry(
                    base_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_world.root
                    )
                ),
                Parallel(
                    [
                        CartesianPose(
                            root_link=cylinder_bot_world.root,
                            tip_link=tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                x=1, reference_frame=cylinder_bot_world.root
                            ),
                        ),
                        ExternalCollisionAvoidance(),
                    ]
                ),
            ]
        )
    )
    msc.add_node(local_min := LocalMinimumReached())
    msc.add_node(EndMotion.when_true(local_min))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_world)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(
        context=MotionStatechartContext(world=cylinder_bot_world),
    )
    kin_sim.compile(motion_statechart=msc_copy)

    with pytest.raises(CollisionViolatedError) as exc_info:
        kin_sim.tick_until_end()
    assert len(exc_info.value.violated_collisions) == 2


@pytest.mark.parametrize(
    "fix_name, tool_frame_id, robot_type",
    [
        ("tracy_world", "r_gripper_tool_frame", Tracy),
        ("daisy_world", "right_gripper_tool_frame", DAiSy),
    ],
)
def test_collision_for_robot_with_static_base(
    fix_name, tool_frame_id, robot_type, request, rclpy_node
):
    world = request.getfixturevalue(fix_name)
    robot = world.get_semantic_annotations_by_type(robot_type)[0]

    tool_frame = world.get_body_by_name(tool_frame_id)
    with world.modify_world():
        obstacle = Body(
            name=PrefixedName("obstacle"),
            collision=ShapeCollection([Box(scale=Scale(0.4, 0.4, 0.4))]),
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=obstacle,
                name=PrefixedName("obstacle_conn"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.5, 0.5, 1, reference_frame=world.root
                ),
            )
        )

    msc = MotionStatechart()
    msc.add_node(
        goal := Parallel(
            [
                CartesianPosition(
                    root_link=world.root,
                    tip_link=tool_frame,
                    goal_point=Point3(0.5, 0.5, 1, reference_frame=world.root),
                ),
                ExternalCollisionAvoidance(robot=robot),
                SelfCollisionAvoidance(robot=robot),
            ],
            minimum_success=1,
        )
    )
    msc.add_node(local_min := LocalMinimumReached())
    msc.add_node(CancelMotion.when_true(local_min))
    msc.add_node(EndMotion.when_true(goal))

    kin_sim = Executor(
        MotionStatechartContext(world=world),
        pacer=SimulationPacer(real_time_factor=2),
    )
    kin_sim.compile(motion_statechart=msc)
    with pytest.raises(Exception):
        # Either Timeout or CancelMotion Execption
        kin_sim.tick_until_end(500)

    # Verify no contact between the gripper and the obstacle
    collisions = kin_sim.context.world.collision_manager.compute_collisions()
    for contact in collisions.contacts:
        if obstacle in (contact.body_a, contact.body_b):
            assert contact.distance >= 0, (
                f"Gripper penetrated the obstacle (distance={contact.distance:.4f}m)."
                "ExternalCollisionAvoidance is not avoiding the obstacle."
            )


def test_repeated_collision_pr2_apartment_does_not_increase_execution_time(
    pr2_apartment_world,
):
    world = deepcopy(pr2_apartment_world)

    tool_frame = world.get_body_by_name("r_gripper_tool_frame")
    robot = world.get_semantic_annotations_by_type(PR2)[0]

    left_arm_park = robot.left_arm.get_joint_state_by_type(StaticJointState.PARK)
    right_arm_park = robot.right_arm.get_joint_state_by_type(StaticJointState.PARK)
    left_arm_park.apply_to(world)
    right_arm_park.apply_to(world)

    body = world.get_body_by_name("handle_cab11_t")

    execution_times = []
    for i in range(10):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                UpdateTemporaryCollisionRules(
                    temporary_rules=[
                        AvoidExternalCollisions(robot=robot),
                        AllowCollisionBetweenGroups(
                            body_group_a=[
                                b
                                for b in robot.right_arm.end_effector.bodies
                                if b.has_collision()
                            ],
                            body_group_b=[
                                b
                                for b in world.bodies
                                if "apartment" in str(b.name) and b.has_collision()
                            ],
                        ),
                    ]
                ),
                CartesianPose(
                    root_link=world.root,
                    tip_link=tool_frame,
                    goal_pose=body.global_pose,
                ),
                ExternalCollisionAvoidance(robot=robot),
                SelfCollisionAvoidance(robot=robot),
            ]
        )
        msc.add_node(EndMotion.when_true(msc.nodes[1]))

        kin_sim = Executor(
            MotionStatechartContext(world=world),
        )
        kin_sim.compile(motion_statechart=msc)
        with world.reset_state_context():
            start_time = time.time()
            kin_sim.tick_until_end(500)
            end_time = time.time()
            execution_times.append(end_time - start_time)

    # Split execution times into two halves
    half = len(execution_times) // 2
    first_half_median = np.median(execution_times[:half])
    second_half_median = np.median(execution_times[half:])

    # Assert that the second half is not significantly slower than the first half.
    # We allow a small margin (e.g., 20%) for natural noise.
    assert second_half_median <= first_half_median * 1.2, (
        f"Execution time is increasing: first half median {first_half_median:.4f}s, "
        f"second half median {second_half_median:.4f}s"
    )
