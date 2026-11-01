import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pytest

from cramph.executor import SimulationPacer
from giskardpy.motion_statechart.weights import (
    DefaultWeights,
)
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    _CancelBecauseExternalCollisionViolated,
    _CancelBecauseSelfCollisionViolated,
    _ExternalCollisionAvoidanceTask,
    _SelfCollisionAvoidanceTask,
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    ExternalCollisionDistanceMonitor,
    SelfCollisionDistanceMonitor,
    UpdateTemporaryCollisionRules,
)
from cramph.composites import Sequence, Parallel
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.nodes_for_testing import ConstFalseNode
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from cramph.node import CancelStatechart, StatechartNode
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from cramph.monitors import CountTicks
from cramph.statechart import (
    Statechart,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.collision_checking.collision_manager import CollisionConsumer
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
    AvoidExternalCollisions,
    AvoidAllCollisions,
    AllowAllCollisions,
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.exceptions import TemporaryCollisionRulesDependOnOrder
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    RotationMatrix,
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
    KinematicStructureEntity,
)
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from ..motion_control_context import (
    BuiltForTheExecutor,
    SentThroughJSON,
    StatechartOrigin,
)


@pytest.fixture(
    params=[BuiltForTheExecutor(), SentThroughJSON()],
    ids=lambda origin: type(origin).__name__,
)
def statechart_origin(request: pytest.FixtureRequest) -> StatechartOrigin:
    """
    :return: One of the ways a statechart reaches the executor running it.
    """
    return request.param


def test_external_collision_avoidance(
    cylinder_bot_world: World, statechart_origin: StatechartOrigin
) -> None:
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        pacer=SimulationPacer(real_time_factor=2),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=statechart_origin.create_building_context(executor))
    statechart.add_nodes(
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
    statechart.add_node(EndMotion.when_true(local_min))
    statechart.add_node(CancelStatechart.when_true(distance_violated))

    executor.compile(statechart=statechart_origin.hand_over(statechart, executor))

    executor.tick_until_end(500)
    collisions = executor.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    assert collisions.contacts[0].distance > 0.049
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_external_collision_avoidance_battle():
    strong_robot_world = World()
    with strong_robot_world.modify_world():
        strong = Body(
            name=PrefixedName("strong"),
            collision=ShapeCollection(
                [
                    Cylinder(
                        width=0.1,
                        height=0.1,
                        color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
                    )
                ]
            ),
        )
        strong_robot_world.add_body(strong)
        strong_robot_sa = MinimalRobot.from_world(strong_robot_world)

    weak_robot_world = World()
    with weak_robot_world.modify_world():
        weak = Body(
            name=PrefixedName("weak"),
            collision=ShapeCollection(
                [
                    Cylinder(
                        width=0.1,
                        height=0.1,
                        color=Color.from_rgba(red=0, green=0, blue=1, opacity=1),
                    )
                ]
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

    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        pacer=SimulationPacer(real_time_factor=1),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
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
    statechart.add_node(EndMotion.when_true(local_min))
    # statechart.add_node(CancelStatechart.when_true(distance_violated))

    executor.compile(statechart=statechart)

    executor.tick_until_end(500)

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


def test_external_collision_avoidance_with_weight_above_ca(
    cylinder_bot_world: World, statechart_origin: StatechartOrigin
) -> None:
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        pacer=SimulationPacer(real_time_factor=1),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=statechart_origin.create_building_context(executor))
    statechart.add_nodes(
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
    statechart.add_node(EndMotion.when_true(local_min))
    statechart.add_node(CancelStatechart.when_true(distance_violated))

    executor.compile(statechart=statechart_origin.hand_over(statechart, executor))

    executor.tick_until_end(500)
    collisions = executor.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    assert collisions.contacts[0].distance < 0.049
    assert collisions.contacts[0].distance > 0.0
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_update_collision_matrix_later(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            ExternalCollisionAvoidance(robot=robot),
            cart_goal_reached := CartesianPose(
                root_link=cylinder_bot_world.root,
                tip_link=tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_world.root
                ),
            ),
            # The avoidance rules are in force until the robot is stuck in front of the
            # obstacle; only then do the rules allowing every collision start.
            Sequence(
                nodes=[
                    Parallel(
                        nodes=[
                            LocalMinimumReached(),
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
                        ]
                    ),
                    Parallel(
                        nodes=[
                            ConstFalseNode(),
                            UpdateTemporaryCollisionRules(
                                temporary_rules=[AllowAllCollisions()]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
    statechart.add_node(EndMotion.when_true(cart_goal_reached))

    executor.compile(statechart=statechart)

    executor.tick_until_end(500)
    statechart.draw("muh.pdf")
    cylinder_bot_world.collision_manager.clear_temporary_rules()
    cylinder_bot_world.collision_manager.add_temporary_rule(AvoidAllCollisions())
    cylinder_bot_world.collision_manager.update_collision_matrix()
    collisions = executor.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    assert collisions.contacts[0].distance < 0.049
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_consumer_cleanup_after_cancel(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
    env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
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
    statechart.add_node(
        Sequence(
            nodes=[
                CountTicks(ticks=5),
                CancelStatechart(exception=Exception("muh")),
            ]
        )
    )
    statechart.add_node(EndMotion.when_true(local_min))

    executor.compile(statechart=statechart)

    with pytest.raises(Exception, match="muh"):
        executor.tick_until_end(500)
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_multiple_external_collision_avoidance_motions(cylinder_bot_world: World):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")

    def run_motion(goal_x):
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        statechart.add_nodes(
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
        statechart.add_node(EndMotion.when_true(local_min))

        executor.compile(statechart=statechart)
        executor.tick_until_end(500)
        return executor

    # First motion
    run_motion(1.0)
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0

    # Second motion
    run_motion(0.5)
    assert len(cylinder_bot_world.collision_manager.collision_consumers) == 0


def test_cancel_node_without_tasks_never_starts():
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(
        cancel := _CancelBecauseSelfCollisionViolated(name="cancel", tasks=[])
    )

    cancel.build()

    assert cancel.start_condition.is_constant_false()


def test_self_collision_avoidance_without_checked_body_combinations(
    cylinder_bot_world: World,
):
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            goal := SelfCollisionAvoidance(robot=robot),
            local_min := LocalMinimumReached(),
        ]
    )
    statechart.add_node(EndMotion.when_true(local_min))

    executor.compile(statechart=statechart)

    task, cancel = goal.nodes
    assert task.checked_combinations == []
    assert task.constraint_collection.inequality_constraints == []
    assert [cancel] == statechart.get_nodes_by_type(CancelStatechart)


def test_self_collision_avoidance(
    self_collision_bot_world: World, statechart_origin: StatechartOrigin
) -> None:

    robot = self_collision_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    l_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_tip")
    r_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_tip")
    l_thumb = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_thumb")
    r_thumb = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_thumb")

    executor = StatechartExecutor(
        context=StatechartContext(world=self_collision_bot_world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=statechart_origin.create_building_context(executor))
    statechart.add_nodes(
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
    statechart.add_node(EndMotion.when_true(local_min))

    executed_statechart = statechart_origin.hand_over(statechart, executor)
    executor.compile(statechart=executed_statechart)

    # 4 because of the base nodes + 1 task avoiding every self collision + 1 for CancelStatechart
    assert len(executed_statechart.nodes) == 4 + 1 + 1

    executor.tick_until_end(500)
    collisions = executor.context.world.collision_manager.compute_collisions()
    assert len(collisions.contacts) == 1
    for contact in collisions.contacts:
        assert contact.distance > 0.249
    assert len(self_collision_bot_world.collision_manager.collision_consumers) == 0


def test_avoid_collision_go_around_corner(pr2_with_box):
    r_tip = pr2_with_box.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    robot = pr2_with_box.get_semantic_annotations_by_type(AbstractRobot)[0]

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_with_box), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        UpdateTemporaryCollisionRules(
            temporary_rules=[
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1, violated_distance=0.0, robot=robot
                )
            ]
        )
    )
    statechart.add_node(
        Sequence(
            nodes=[
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
                    nodes=[
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
    statechart.add_node(
        distance_violated := ExternalCollisionDistanceMonitor(
            body=pr2_with_box.get_kinematic_structure_entity_by_name(
                "r_gripper_palm_link"
            ),
            threshold=0.0,
        ),
    )
    statechart.add_node(local_min := LocalMinimumReached())
    statechart.add_node(EndMotion.when_true(local_min))
    statechart.add_node(CancelStatechart.when_true(distance_violated))

    executor.compile(statechart=statechart)

    executor.tick_until_end(500)


def test_avoid_self_collision_with_l_arm(pr2_with_box, rclpy_node):
    VizMarkerPublisher(
        _world=pr2_with_box, node=rclpy_node
    ).with_collision_visualization()
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

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_with_box),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig(
                    target_frequency=100,
                    braking_time=timedelta(seconds=0.289),
                )
            )
        ],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        Sequence(
            nodes=[
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
                    nodes=[
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
    statechart.add_node(
        contact := SelfCollisionDistanceMonitor(
            body_a=r_palm_link, body_b=l_forearm_link, threshold=0.01
        )
    )
    statechart.add_node(local_min := LocalMinimumReached())
    statechart.add_node(EndMotion.when_true(local_min))
    statechart.add_node(CancelStatechart.when_true(contact))

    executor.compile(statechart=statechart)

    assert len(statechart.nodes) == 13

    executor.tick_until_end(500)


# %% one task avoiding every collision


def _compile_collision_avoidance(
    world: World, goal: StatechartNode, *nodes: StatechartNode
) -> StatechartExecutor:
    """
    :param world: The world the robot of `goal` stands in.
    :param goal: The collision avoidance goal to compile.
    :param nodes: Further nodes running next to it.
    :return: An executor that compiled `goal` and `nodes` into one statechart.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes([goal, *nodes, local_min := LocalMinimumReached()])
    statechart.add_node(EndMotion.when_true(local_min))
    executor.compile(statechart=statechart)
    return executor


def _weights(task: Task) -> list[float]:
    """
    :param task: A built task.
    :return: The current weight of every inequality constraint of `task`, in order.
    """
    return [
        float(constraint.quadratic_weight.evaluate()[0])
        for constraint in task.constraint_collection.inequality_constraints
    ]


def test_self_collision_avoidance_avoids_every_checked_combination_in_one_task(
    self_collision_bot_world: World,
) -> None:
    robot = self_collision_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    goal = SelfCollisionAvoidance(robot=robot)

    executor = _compile_collision_avoidance(self_collision_bot_world, goal)

    task, cancel = goal.nodes
    assert isinstance(task, _SelfCollisionAvoidanceTask)
    assert isinstance(cancel, _CancelBecauseSelfCollisionViolated)
    manager = executor.context.require_extension(
        MotionControlContext
    ).self_collision_manager
    assert task.checked_combinations == list(manager.registered_group_combinations)
    assert task.avoided_contacts == [
        contact
        for combination in manager.registered_group_combinations
        for contact in manager.avoided_contacts_of(combination)
    ]
    assert len(task.constraint_collection.inequality_constraints) == len(
        task.avoided_contacts
    )


def test_self_collision_avoidance_avoids_the_closest_contacts_of_a_combination(
    self_collision_bot_world: World,
) -> None:
    """
    Each hand holds a tip and a thumb, so the two hands touch in several places at once.

    Avoiding only the closest of them leaves the next closest free to close the gap
    within one control cycle.
    """
    world = self_collision_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    left_hand = {
        world.get_kinematic_structure_entity_by_name("l_tip"),
        world.get_kinematic_structure_entity_by_name("l_thumb"),
    }
    right_hand = {
        world.get_kinematic_structure_entity_by_name("r_tip"),
        world.get_kinematic_structure_entity_by_name("r_thumb"),
    }
    world.collision_manager.max_avoided_bodies_rules.append(
        MaxAvoidedCollisionsOverride(2, left_hand | right_hand)
    )
    goal = SelfCollisionAvoidance(robot=robot)
    executor = _compile_collision_avoidance(
        world,
        goal,
        UpdateTemporaryCollisionRules(
            temporary_rules=[
                AvoidCollisionBetweenGroups(
                    buffer_zone_distance=1.0,
                    violated_distance=0.0,
                    body_group_a=left_hand,
                    body_group_b=right_hand,
                ),
            ]
        ),
    )

    executor.tick()

    task = goal.nodes[0]
    manager = executor.context.require_extension(
        MotionControlContext
    ).self_collision_manager
    (hands,) = [
        combination
        for combination in task.checked_combinations
        if combination.group_a.root in left_hand | right_hand
        and combination.group_b.root in left_hand | right_hand
    ]
    contacts = manager.last_closest_contacts[hands]
    assert len(contacts) == 4
    avoided_distances = [
        manager.get_contact_distance_symbol(contact).evaluate()[0]
        for contact in manager.avoided_contacts_of(hands)
    ]
    assert avoided_distances == pytest.approx(
        sorted(contact.distance for contact in contacts)[:2]
    )


def test_external_collision_avoidance_avoids_every_closest_object_in_one_task(
    cylinder_bot_world: World,
) -> None:
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    goal = ExternalCollisionAvoidance(robot=robot)

    executor = _compile_collision_avoidance(cylinder_bot_world, goal)

    task, cancel = goal.nodes
    assert isinstance(task, _ExternalCollisionAvoidanceTask)
    assert isinstance(cancel, _CancelBecauseExternalCollisionViolated)
    manager = executor.context.require_extension(
        MotionControlContext
    ).external_collision_manager
    assert task.avoided_contacts == [
        contact
        for group in manager.registered_groups
        for contact in manager.avoided_contacts_of(group)
    ]
    assert len(task.constraint_collection.inequality_constraints) == len(
        task.avoided_contacts
    )


def test_only_self_collision_combinations_with_collision_data_weigh_anything(
    self_collision_bot_world: World,
) -> None:
    robot = self_collision_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    l_thumb = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_thumb")
    r_thumb = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_thumb")
    goal = SelfCollisionAvoidance(robot=robot)
    executor = _compile_collision_avoidance(
        self_collision_bot_world,
        goal,
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
    )

    executor.tick()

    task = goal.nodes[0]
    manager = executor.context.require_extension(
        MotionControlContext
    ).self_collision_manager
    with_data = [
        combination in manager.last_closest_contacts
        for combination in task.checked_combinations
    ]
    assert any(with_data) and not all(with_data)
    assert _weights(task) == [
        DefaultWeights.WEIGHT_COLLISION_AVOIDANCE if has_data else 0.0
        for has_data in with_data
    ]


def test_only_closest_objects_with_collision_data_weigh_anything(
    cylinder_bot_world: World,
) -> None:
    robot = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    environment = cylinder_bot_world.get_kinematic_structure_entity_by_name(
        "environment"
    )
    goal = ExternalCollisionAvoidance(robot=robot)
    executor = _compile_collision_avoidance(
        cylinder_bot_world,
        goal,
        UpdateTemporaryCollisionRules(
            temporary_rules=[
                AvoidCollisionBetweenGroups(
                    buffer_zone_distance=10.0,
                    violated_distance=0.0,
                    body_group_a=[tip],
                    body_group_b=[environment],
                )
            ]
        ),
    )

    executor.tick()

    task = goal.nodes[0]
    manager = executor.context.require_extension(
        MotionControlContext
    ).external_collision_manager
    with_data = [
        contact.rank < len(manager.last_closest_contacts[contact.group])
        for contact in task.avoided_contacts
    ]
    assert any(with_data) and not all(with_data)
    assert _weights(task) == [
        DefaultWeights.WEIGHT_COLLISION_AVOIDANCE if has_data else 0.0
        for has_data in with_data
    ]


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


def _create_counting_executor(world: World) -> StatechartExecutor:
    """
    :return: An executor with motion control acting in `world`, for counting collision
        checks.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig(
                    target_frequency=100,
                    braking_time=timedelta(seconds=0.289),
                )
            )
        ],
    )


def _build_arms_crossing_statechart(
    executor: StatechartExecutor, collision_avoidance: bool
) -> Statechart:
    """
    Builds a statechart that drives the right gripper into the left arm.

    :param executor: The executor whose world the goal refers to.
    :param collision_avoidance: Whether a self collision avoidance node is part of the
        goal.
    """
    world = executor.context.world
    r_tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    base_footprint = world.get_kinematic_structure_entity_by_name("base_footprint")

    statechart = Statechart(context=executor.context)
    statechart.add_node(
        goal := Sequence(
            nodes=[
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
    statechart.add_node(EndMotion.when_true(goal))
    if collision_avoidance:
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
        statechart.add_node(SelfCollisionAvoidance(robot=robot))
    return statechart


def _run_and_count_collision_checks(
    executor: StatechartExecutor, statechart: Statechart
) -> CollisionCheckCountingObserver:
    """
    Executes the statechart and returns the observer that counted the collision checks.
    """
    world = executor.context.world
    observer = CollisionCheckCountingObserver()
    world.collision_manager.add_collision_consumer(observer)
    executor.compile(statechart=statechart)
    executor.tick_until_end(500)
    world.collision_manager.remove_collision_consumer(observer)
    return observer


def test_collisions_are_not_computed_without_collision_nodes(pr2_with_box):
    executor = _create_counting_executor(pr2_with_box)
    statechart = _build_arms_crossing_statechart(executor, collision_avoidance=False)

    for node in statechart.nodes:
        assert not isinstance(node, SelfCollisionAvoidance)
        assert not isinstance(node, ExternalCollisionAvoidance)

    observer = _run_and_count_collision_checks(executor, statechart)
    assert observer.collision_check_count == 0


def test_collisions_are_computed_with_collision_avoidance_node(pr2_with_box):
    executor = _create_counting_executor(pr2_with_box)
    statechart = _build_arms_crossing_statechart(executor, collision_avoidance=True)

    observer = _run_and_count_collision_checks(executor, statechart)
    assert observer.collision_check_count > 0


def test_collisions_are_computed_for_a_distance_monitor_alone(pr2_with_box):
    executor = _create_counting_executor(pr2_with_box)
    statechart = _build_arms_crossing_statechart(executor, collision_avoidance=False)
    statechart.add_node(
        ExternalCollisionDistanceMonitor(
            body=pr2_with_box.get_kinematic_structure_entity_by_name(
                "r_gripper_palm_link"
            ),
            threshold=0.0,
        )
    )

    observer = _run_and_count_collision_checks(executor, statechart)
    assert observer.collision_check_count > 0


def test_hard_constraints_violated(
    cylinder_bot_world: World, statechart_origin: StatechartOrigin
) -> None:
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

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=statechart_origin.create_building_context(executor))
    # In force for the whole motion, next to the steps below.
    statechart.add_node(
        UpdateTemporaryCollisionRules(
            temporary_rules=[
                AvoidAllCollisions(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                )
            ]
        )
    )
    statechart.add_node(
        Sequence(
            nodes=[
                SetOdometry(
                    base_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_world.root
                    )
                ),
                Parallel(
                    nodes=[
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
    statechart.add_node(local_min := LocalMinimumReached())
    statechart.add_node(EndMotion.when_true(local_min))

    executor.compile(statechart=statechart_origin.hand_over(statechart, executor))

    with pytest.raises(CollisionViolatedError) as exc_info:
        executor.tick_until_end()
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

    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        pacer=SimulationPacer(real_time_factor=2),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        goal := Parallel(
            nodes=[
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
    statechart.add_node(local_min := LocalMinimumReached())
    statechart.add_node(CancelStatechart.when_true(local_min))
    statechart.add_node(EndMotion.when_true(goal))

    executor.compile(statechart=statechart)
    with pytest.raises(Exception):
        # Either Timeout or CancelStatechart Execption
        executor.tick_until_end(500)

    # Verify no contact between the gripper and the obstacle
    collisions = executor.context.world.collision_manager.compute_collisions()
    for contact in collisions.contacts:
        if obstacle in (contact.body_a, contact.body_b):
            assert contact.distance >= 0, (
                f"Gripper penetrated the obstacle (distance={contact.distance:.4f}m)."
                "ExternalCollisionAvoidance is not avoiding the obstacle."
            )


@pytest.mark.flaky(reruns=3)
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
        executor = StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )
        statechart = Statechart(context=executor.context)
        statechart.add_nodes(
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
        statechart.add_node(EndMotion.when_true(statechart.nodes[1]))

        executor.compile(statechart=statechart)
        with world.reset_state_context():
            start_time = time.time()
            executor.tick_until_end(500)
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


# %% rules in force while a rules node is built


def test_building_a_rules_node_keeps_the_temporary_rules_in_force(
    cylinder_bot_world: World,
) -> None:
    collision_manager = cylinder_bot_world.collision_manager
    rule_in_force = AllowAllCollisions()
    collision_manager.add_temporary_rule(rule_in_force)
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    rules_node = UpdateTemporaryCollisionRules(temporary_rules=[AvoidAllCollisions()])
    statechart.add_node(Sequence(nodes=[ConstFalseNode(), rules_node]))

    executor.compile(statechart)

    assert rules_node.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert collision_manager.temporary_rules == [rule_in_force]


# %% how long the rules of a rules node are in force


def _executor_of(world: World) -> StatechartExecutor:
    """
    :return: An executor running motion statecharts in `world`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )


def test_rules_nodes_running_at_once_keep_the_rules_of_each_other_in_force(
    pr2_world_copy: World,
) -> None:
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    left = UpdateTemporaryCollisionRules(
        temporary_rules=[
            AllowCollisionForEndEffector(end_effector=robot.left_arm.end_effector)
        ]
    )
    right = UpdateTemporaryCollisionRules(
        temporary_rules=[
            AllowCollisionForEndEffector(end_effector=robot.right_arm.end_effector)
        ]
    )
    executor = _executor_of(pr2_world_copy)
    statechart = Statechart(context=executor.context)
    statechart.add_nodes([left, right])

    executor.compile(statechart)
    executor.tick()

    assert pr2_world_copy.collision_manager.temporary_rules == [
        *left.temporary_rules,
        *right.temporary_rules,
    ]


def test_the_rules_of_a_rules_node_are_out_of_force_once_it_ended(
    cylinder_bot_world: World,
) -> None:
    collision_manager = cylinder_bot_world.collision_manager
    rule_in_force = AllowAllCollisions()
    collision_manager.add_temporary_rule(rule_in_force)
    executor = _executor_of(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    rules_node = UpdateTemporaryCollisionRules(temporary_rules=[AllowAllCollisions()])
    statechart.add_node(Sequence(nodes=[rules_node, ConstFalseNode()]))

    executor.compile(statechart)
    executor.tick()
    executor.tick()

    assert rules_node.life_cycle_state.is_terminal
    assert collision_manager.temporary_rules == [rule_in_force]


def test_rules_nodes_whose_order_changes_what_is_checked_do_not_run_at_once(
    cylinder_bot_world: World,
) -> None:
    executor = _executor_of(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            UpdateTemporaryCollisionRules(temporary_rules=[AvoidAllCollisions()]),
            UpdateTemporaryCollisionRules(temporary_rules=[AllowAllCollisions()]),
        ]
    )

    with pytest.raises(TemporaryCollisionRulesDependOnOrder):
        executor.compile(statechart)
        executor.tick()


# %% PR2 self collision with attached objects


def attach_box(
    world: World,
    name: str,
    size: tuple[float, float, float],
    parent: KinematicStructureEntity,
    pose: HomogeneousTransformationMatrix,
) -> Body:
    """
    Adds a box rigidly attached to ``parent`` at ``pose``.

    :return: The added box.
    """
    parent_T_box = world.transform(spatial_object=pose, target_frame=parent)
    with world.modify_world():
        box = Body(
            name=PrefixedName(name),
            collision=ShapeCollection([Box(scale=Scale(*size))]),
        )
        world.add_connection(
            FixedConnection(
                parent=parent, child=box, parent_T_connection_expression=parent_T_box
            )
        )
    return box


def assert_bodies_keep_distance(
    world: World, bodies: Iterable[Body], minimum_distance: float
) -> None:
    """
    Asserts that the collision matrix checks ``bodies`` and that every contact of one of
    them is at least ``minimum_distance`` away.

    A checked pair without a contact is farther apart than its checking range.
    """
    bodies = set(bodies)
    assert any(
        check.body_a in bodies or check.body_b in bodies
        for check in world.collision_manager.collision_matrix.collision_checks
    )
    contacts = world.collision_manager.compute_collisions().contacts
    for contact in contacts:
        if contact.body_a in bodies or contact.body_b in bodies:
            assert contact.distance >= minimum_distance, (
                f"{contact.distance} < {minimum_distance} "
                f"({contact.body_a.name} with {contact.body_b.name})"
            )


def run_to_end(
    world: World, nodes: list[StatechartNode], end_when: StatechartNode
) -> None:
    """
    Runs ``nodes`` in ``world`` until ``end_when`` observes true.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(nodes)
    statechart.add_node(EndMotion.when_true(end_when))
    executor.compile(statechart=statechart)
    executor.tick_until_end()


def test_cable_guide_collision(
    pr2_world_copy: World, default_pr2_pose: dict[str, float]
) -> None:
    """
    Turning the head into the cable guide is stopped by self collision avoidance.
    """
    JointState.from_str_dict(default_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    joint_goal = JointPositionList(
        goal_state=JointState.from_str_dict(
            {PR2Joint.HEAD_PAN: 2.84, PR2Joint.HEAD_TILT: 1.0}, world=pr2_world_copy
        )
    )
    local_min = LocalMinimumReached()

    run_to_end(
        pr2_world_copy,
        [joint_goal, SelfCollisionAvoidance(robot=robot), local_min],
        end_when=local_min,
    )

    assert joint_goal.observation_state == ObservationStateValues.FALSE


def test_attached_self_collision_avoid_stick(
    pr2_world_copy: World, default_pr2_pose: dict[str, float]
) -> None:
    """
    A stick held in the left gripper keeps its distance to the robot, too.
    """
    JointState.from_str_dict(default_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    JointState.from_str_dict(
        {
            PR2Joint.LEFT_ELBOW_FLEX: -1.1343683863086362,
            PR2Joint.LEFT_FOREARM_ROLL: 7.517553513504836,
            PR2Joint.LEFT_SHOULDER_LIFT: 0.5726770101613905,
            PR2Joint.LEFT_SHOULDER_PAN: 0.1592669164939349,
            PR2Joint.LEFT_UPPER_ARM_ROLL: 0.5532568387077381,
            PR2Joint.LEFT_WRIST_FLEX: -1.215660155912625,
            PR2Joint.LEFT_WRIST_ROLL: 4.249300323527076,
        },
        pr2_world_copy,
    ).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    left_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "l_gripper_tool_frame"
    )
    stick = attach_box(
        pr2_world_copy,
        name="stick",
        size=(0.16, 0.04, 0.04),
        parent=left_tip,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.04, reference_frame=left_tip
        ),
    )
    cart_goal = CartesianPose(
        root_link=pr2_world_copy.root,
        tip_link=left_tip,
        goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.2, reference_frame=left_tip
        ),
    )

    run_to_end(
        pr2_world_copy,
        [cart_goal, SelfCollisionAvoidance(robot=robot)],
        end_when=cart_goal,
    )

    assert_bodies_keep_distance(
        pr2_world_copy,
        robot.left_arm.end_effector.bodies_with_collision + [stick],
        0.048,
    )


def test_box_overlapping_with_gripper(
    pr2_world_copy: World, better_pr2_pose: dict[str, float]
) -> None:
    """
    A held box that overlaps the gripper holding it does not keep the gripper from
    moving.
    """
    JointState.from_str_dict(better_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    right_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    box = attach_box(
        pr2_world_copy,
        name="box",
        size=(0.2, 0.1, 0.1),
        parent=right_tip,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.05, reference_frame=right_tip
        ),
    )
    cart_goal = CartesianPose(
        root_link=pr2_world_copy.root,
        tip_link=box,
        goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-0.5, reference_frame=box
        ),
    )

    run_to_end(
        pr2_world_copy,
        [cart_goal, SelfCollisionAvoidance(robot=robot)],
        end_when=cart_goal,
    )

    assert cart_goal.observation_state == ObservationStateValues.TRUE


def test_avoid_self_collision_specific_link(
    pr2_world_copy: World, default_pr2_pose: dict[str, float]
) -> None:
    """
    A temporary rule makes the right gripper keep its distance to the left forearm.
    """
    JointState.from_str_dict(default_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    JointState.from_str_dict(
        {
            PR2Joint.RIGHT_SHOULDER_PAN: -0.0672581793019,
            PR2Joint.RIGHT_SHOULDER_LIFT: 0.429650469244,
            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.580889703636,
            PR2Joint.RIGHT_FOREARM_ROLL: -101.948215412,
            PR2Joint.RIGHT_ELBOW_FLEX: -1.35221928696,
            PR2Joint.RIGHT_WRIST_FLEX: -0.986144640142,
            PR2Joint.RIGHT_WRIST_ROLL: 0.7397216172451033,
        },
        pr2_world_copy,
    ).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    right_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    right_gripper = robot.right_arm.end_effector.bodies_with_collision
    cart_goal = CartesianPose(
        root_link=pr2_world_copy.get_kinematic_structure_entity_by_name(
            "base_footprint"
        ),
        tip_link=right_tip,
        goal_pose=Pose.from_xyz_rpy(0.2, reference_frame=right_tip),
    )

    run_to_end(
        pr2_world_copy,
        [
            cart_goal,
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=0.05,
                        body_group_a=right_gripper,
                        body_group_b=[
                            pr2_world_copy.get_kinematic_structure_entity_by_name(
                                "l_forearm_link"
                            )
                        ],
                    )
                ]
            ),
            SelfCollisionAvoidance(robot=robot),
        ],
        end_when=cart_goal,
    )

    assert_bodies_keep_distance(pr2_world_copy, right_gripper, 0.048)


def test_get_out_of_self_collision(
    pr2_world_copy: World, default_pr2_pose: dict[str, float]
) -> None:
    """
    Self collision avoidance gives up on a robot that already is in self collision.
    """
    JointState.from_str_dict(default_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    JointState.from_str_dict(
        {
            PR2Joint.LEFT_ELBOW_FLEX: -1.43286344265,
            PR2Joint.LEFT_FOREARM_ROLL: 1.26465060073,
            PR2Joint.LEFT_SHOULDER_LIFT: 0.47990329056,
            PR2Joint.LEFT_SHOULDER_PAN: 0.281272240139,
            PR2Joint.LEFT_UPPER_ARM_ROLL: 0.528415402668,
            PR2Joint.LEFT_WRIST_FLEX: -1.18811419869,
            PR2Joint.LEFT_WRIST_ROLL: 2.26884630124,
        },
        pr2_world_copy,
    ).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    left_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "l_gripper_tool_frame"
    )
    into_collision = CartesianPose(
        root_link=pr2_world_copy.get_kinematic_structure_entity_by_name(
            "base_footprint"
        ),
        tip_link=left_tip,
        goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.15, reference_frame=left_tip
        ),
    )
    run_to_end(pr2_world_copy, [into_collision], end_when=into_collision)
    local_min = LocalMinimumReached()

    with pytest.raises(CollisionViolatedError):
        run_to_end(
            pr2_world_copy,
            [SelfCollisionAvoidance(robot=robot), local_min],
            end_when=local_min,
        )


def test_attached_two_items(
    pr2_world_copy: World, default_pr2_pose: dict[str, float]
) -> None:
    """
    Boxes held in both grippers keep their distance to each other.
    """
    JointState.from_str_dict(default_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    JointState.from_str_dict(
        {
            PR2Joint.RIGHT_ELBOW_FLEX: -1.58118094489,
            PR2Joint.RIGHT_FOREARM_ROLL: -0.904933033043,
            PR2Joint.RIGHT_SHOULDER_LIFT: 0.822412440711,
            PR2Joint.RIGHT_SHOULDER_PAN: -1.07866800992,
            PR2Joint.RIGHT_UPPER_ARM_ROLL: -1.34905471854,
            PR2Joint.RIGHT_WRIST_FLEX: -1.20182042644,
            PR2Joint.RIGHT_WRIST_ROLL: 0.190433188769,
        },
        pr2_world_copy,
    ).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    right_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    left_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "l_gripper_tool_frame"
    )
    grippers_facing = CartesianPose(
        root_link=pr2_world_copy.get_kinematic_structure_entity_by_name(
            "torso_lift_link"
        ),
        tip_link=left_tip,
        # The boxes reach 0.2 m out of each gripper, so grippers 0.4 m apart would leave
        # them touching, which violates the collision before avoidance can act.
        goal_pose=HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=Point3(x=0.42),
            rotation_matrix=RotationMatrix(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]]
            ),
            reference_frame=right_tip,
        ),
    )
    run_to_end(pr2_world_copy, [grippers_facing], end_when=grippers_facing)
    boxes = [
        attach_box(
            pr2_world_copy,
            name=name,
            size=(0.2, 0.04, 0.04),
            parent=tip,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, reference_frame=tip
            ),
        )
        for name, tip in [("box1", right_tip), ("box2", left_tip)]
    ]
    local_min = LocalMinimumReached()

    run_to_end(
        pr2_world_copy,
        [SelfCollisionAvoidance(robot=robot), local_min],
        end_when=local_min,
    )

    assert_bodies_keep_distance(pr2_world_copy, boxes, 0.049)


# %% PR2 external collision avoidance


def test_avoid_collision_box_between_3_boxes(
    pr2_world_copy: World, pocky_pr2_pose: dict[str, float]
) -> None:
    """
    Aligning a held box with the world moves it between three boxes without touching
    them.
    """
    JointState.from_str_dict(pocky_pr2_pose, pr2_world_copy).apply_to(pr2_world_copy)
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    right_tip = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    box = attach_box(
        pr2_world_copy,
        name="box",
        size=(0.2, 0.05, 0.05),
        parent=right_tip,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.08, reference_frame=right_tip
        ),
    )
    for name, size, x, y in [
        ("b1", (0.01, 0.2, 0.2), 0.2, 0.0),
        ("bl", (0.1, 0.01, 0.2), 0.15, 0.04),
        ("br", (0.1, 0.01, 0.2), 0.15, -0.04),
    ]:
        attach_box(
            pr2_world_copy,
            name,
            size,
            parent=pr2_world_copy.root,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x, y=y, reference_frame=right_tip
            ),
        )
    local_min = LocalMinimumReached()

    run_to_end(
        pr2_world_copy,
        [
            AlignPlanes(
                root_link=pr2_world_copy.root,
                tip_link=box,
                tip_normal=Vector3.X(reference_frame=box),
                goal_normal=Vector3.X(reference_frame=pr2_world_copy.root),
            ),
            AlignPlanes(
                root_link=pr2_world_copy.root,
                tip_link=box,
                tip_normal=Vector3.Y(reference_frame=box),
                goal_normal=Vector3.Y(reference_frame=pr2_world_copy.root),
            ),
            ExternalCollisionAvoidance(robot=robot),
            local_min,
        ],
        end_when=local_min,
    )

    assert_bodies_keep_distance(
        pr2_world_copy, robot.right_arm.end_effector.bodies_with_collision, 0.04
    )


# %% HSR self collision


def test_hsr_self_collision_avoidance(
    hsr_world_copy: World, default_hsr_pose: dict[str, float]
) -> None:
    """
    The HSR reaches a goal for its gripper from a configuration close to self collision.
    """
    JointState.from_str_dict(default_hsr_pose, hsr_world_copy).apply_to(hsr_world_copy)
    JointState.from_str_dict(
        {
            HSRBJoint.ARM_FLEX: 0.0,
            HSRBJoint.ARM_LIFT: 0.0,
            HSRBJoint.ARM_ROLL: -1.52,
            HSRBJoint.HEAD_PAN: -0.09,
            HSRBJoint.HEAD_TILT: -0.62,
            HSRBJoint.WRIST_FLEX: -1.55,
            HSRBJoint.WRIST_ROLL: 0.11,
        },
        hsr_world_copy,
    ).apply_to(hsr_world_copy)
    robot = hsr_world_copy.get_semantic_annotations_by_type(HSRB)[0]
    cart_goal = CartesianPose(
        root_link=hsr_world_copy.root,
        tip_link=hsr_world_copy.get_kinematic_structure_entity_by_name(
            "hand_gripper_tool_frame"
        ),
        goal_pose=Pose.from_xyz_axis_angle(
            x=0.5,
            reference_frame=hsr_world_copy.get_kinematic_structure_entity_by_name(
                "hand_palm_link"
            ),
        ),
    )

    run_to_end(
        hsr_world_copy,
        [cart_goal, SelfCollisionAvoidance(robot=robot)],
        end_when=cart_goal,
    )

    assert cart_goal.observation_state == ObservationStateValues.TRUE
