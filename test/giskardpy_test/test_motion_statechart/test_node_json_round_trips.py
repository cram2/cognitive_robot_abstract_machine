"""
Every motion statechart node class sent through JSON inside a statechart.
"""

from __future__ import annotations

import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from giskardpy import motion_statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart import ros2_nodes
from giskardpy.motion_statechart.goals.base_pose_goals import (
    DifferentialDriveBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.goals.cartesian_goals import CartesianPoseStraight
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    ExternalCollisionDistanceMonitor,
    SelfCollisionAvoidance,
    SelfCollisionDistanceMonitor,
    SetInitialTemporaryCollisionRules,
    UpdateTemporaryCollisionRules,
    _CancelBecauseCollisionViolated,
    _CancelBecauseExternalCollisionViolated,
    _CancelBecauseSelfCollisionViolated,
    _CollisionAvoidanceTask,
    _ExternalCollisionAvoidanceTask,
    _SelfCollisionAvoidanceTask,
)
from giskardpy.motion_statechart.goals.gripper import (
    MoveFingersToJointState,
    MoveGripper,
)
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.goals.cylinder_insertion import (
    InsertCylinder,
    StayOnLineTask,
)
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    StructureCopyMotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.monitors.cartesian_monitors import (
    DistanceToLine,
    OrientationReached,
    PointingAt,
    PoseReached,
    PositionReached,
    VectorsAligned,
)
from giskardpy.motion_statechart.monitors.feature_monitors import (
    AngleMonitor,
    DistanceMonitor,
    FeatureFunctionMonitor,
    HeightMonitor,
    PerpendicularMonitor,
)
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.monitors.progress_monitors import (
    AnyMonitoredTaskShortOfItsGoal,
    NoMonitoredTaskApproachingItsGoal,
    Stalled,
    StillProgressing,
    _CancelBecauseNoProgress,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPosition,
    CartesianPositionStraight,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
    CartesianVelocityLimit,
    DistanceAlongAxis,
    HoldOrientation,
    HoldPose,
)
from giskardpy.motion_statechart.tasks.drive_tasks import (
    BaseHeading,
    BaseOrientation,
    BasePosition,
    BaseTurnTowardsGoalPosition,
    DrivenBaseTask,
)
from giskardpy.motion_statechart.tasks.feature_functions import (
    AlignPerpendicular,
    AngleGoal,
    DistanceGoal,
    HeightGoal,
    ReachPoint,
)
from giskardpy.motion_statechart.tasks.grasp_bar import GraspBar
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from giskardpy.motion_statechart.tasks.pointing import (
    Pointing,
    PointingCone,
)
from giskardpy.motion_statechart.tasks.weight_scaling_goals import MaxManipulability
from giskardpy.motion_statechart.tasks.wiggle_insert import WiggleInsert
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.world_entity import Body

from ..motion_control_context import create_context_with_motion_control
from ...statechart_json_round_trip import (
    NodeRoundTripCase,
    NodeRoundTripTests,
    WorldFixture,
    concrete_node_types,
)

# %% what the nodes of the cases refer to


def _root(world: World) -> Body:
    return world.root


def _tip(world: World) -> Body:
    """
    :return: The tool frame of the right arm of the PR2.
    """
    return world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")


def _connection(world: World) -> ActiveConnection1DOF:
    """
    :return: The torso joint of the PR2.
    """
    return world.get_connection_by_name("torso_lift_joint")


def _robot(world: World) -> AbstractRobot:
    return world.get_semantic_annotations_by_type(AbstractRobot)[0]


def _point(world: World) -> Point3:
    return Point3(0.1, 0.2, 0.3, reference_frame=_root(world))


def _tip_point(world: World) -> Point3:
    return Point3(0.0, 0.0, 0.1, reference_frame=_tip(world))


def _root_axis(world: World) -> Vector3:
    return Vector3.Z(reference_frame=_root(world))


def _tip_axis(world: World) -> Vector3:
    return Vector3.X(reference_frame=_tip(world))


def _pose(world: World) -> Pose:
    return Pose.from_xyz_rpy(x=0.1, yaw=0.2, reference_frame=_root(world))


def _transformation(world: World) -> HomogeneousTransformationMatrix:
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.1, yaw=0.2, reference_frame=_root(world)
    )


def _orientation(world: World) -> RotationMatrix:
    return RotationMatrix.from_rpy(yaw=0.2, reference_frame=_root(world))


def _joint_goal(world: World) -> JointPositionList:
    return JointPositionList(
        goal_state=JointState.from_mapping({_connection(world): 0.5})
    )


def _driven_base_node(node_type: type[DrivenBaseTask], world: World) -> DrivenBaseTask:
    """
    :return: A node of `node_type` moving the base of the differential drive in `world`
        to a pose given in the drive's parent.
    """
    return node_type(
        base=world.get_connections_by_type(DifferentialDrive)[0].child,
        goal_pose=_pose(world),
        parent_T_goal_reference_frame=HomogeneousTransformationMatrix(
            reference_frame=_root(world)
        ),
    )


def _collision_rule(world: World) -> AvoidCollisionBetweenGroups:
    return AvoidCollisionBetweenGroups(
        buffer_zone_distance=0.05,
        violated_distance=0.0,
        body_group_a=[_robot(world).root],
        body_group_b=[world.get_kinematic_structure_entity_by_name("environment")],
    )


# %% nodes of the cases


def _still_progressing_cancelling_the_motion(world: World) -> list[StatechartNode]:
    task = _joint_goal(world)
    still_progressing = StillProgressing(monitored_node=task)
    return [task, still_progressing, still_progressing.cancel_motion()]


def _external_collision_avoidance(world: World) -> list[StatechartNode]:
    return [ExternalCollisionAvoidance(robot=_robot(world))]


def _self_collision_avoidance(world: World) -> list[StatechartNode]:
    return [SelfCollisionAvoidance(robot=_robot(world))]


def _feature_fields(world: World) -> dict[str, Body]:
    return {"root_link": _root(world), "tip_link": _tip(world)}


def _cartesian_cases() -> list[NodeRoundTripCase]:
    return [
        NodeRoundTripCase(
            CartesianPose,
            lambda world: [
                CartesianPose(
                    root_link=_root(world), tip_link=_tip(world), goal_pose=_pose(world)
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            HolonomicBaseGoal,
            lambda world: [
                HolonomicBaseGoal(
                    mobile_base=world.get_semantic_annotations_by_type(MobileBase)[0],
                    goal_pose=_pose(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianPoseStraight,
            lambda world: [
                CartesianPoseStraight(
                    root_link=_root(world), tip_link=_tip(world), goal_pose=_pose(world)
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        *[
            NodeRoundTripCase(
                node_type,
                lambda world, node_type=node_type: [
                    node_type(root_link=_root(world), tip_link=_tip(world))
                ],
                WorldFixture.PR2_WORLD,
            )
            for node_type in [HoldOrientation, HoldPose]
        ],
        NodeRoundTripCase(
            CartesianPosition,
            lambda world: [
                CartesianPosition(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_point=_point(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            DistanceAlongAxis,
            lambda world: [
                DistanceAlongAxis(
                    root_link=_root(world),
                    reference_point=_point(world),
                    tip_point=_tip_point(world),
                    axis=_root_axis(world),
                    distance=0.1,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianPositionStraight,
            lambda world: [
                CartesianPositionStraight(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_point=_point(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianPositionTrajectory,
            lambda world: [
                CartesianPositionTrajectory(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_points=[_point(world), _point(world)],
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianOrientation,
            lambda world: [
                CartesianOrientation(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_orientation=_orientation(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianPositionVelocityLimit,
            lambda world: [
                CartesianPositionVelocityLimit(
                    root_link=_root(world), tip_link=_tip(world)
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianRotationVelocityLimit,
            lambda world: [
                CartesianRotationVelocityLimit(
                    root_link=_root(world), tip_link=_tip(world)
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            CartesianVelocityLimit,
            lambda world: [
                CartesianVelocityLimit(root_link=_root(world), tip_link=_tip(world))
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            DifferentialDriveBaseGoal,
            lambda world: [
                DifferentialDriveBaseGoal(
                    mobile_base=world.get_semantic_annotations_by_type(MobileBase)[0],
                    goal_pose=_pose(world),
                )
            ],
            WorldFixture.CYLINDER_BOT_DIFF_WORLD,
        ),
        *[
            NodeRoundTripCase(
                node_type,
                lambda world, node_type=node_type: [
                    _driven_base_node(node_type, world)
                ],
                WorldFixture.CYLINDER_BOT_DIFF_WORLD,
            )
            for node_type in [
                BasePosition,
                BaseOrientation,
                BaseHeading,
                BaseTurnTowardsGoalPosition,
            ]
        ],
        NodeRoundTripCase(
            AlignPlanes,
            lambda world: [
                AlignPlanes(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_normal=_root_axis(world),
                    tip_normal=_tip_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            Pointing,
            lambda world: [
                Pointing(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_point=_point(world),
                    pointing_axis=_tip_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            PointingCone,
            lambda world: [
                PointingCone(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_point=_point(world),
                    pointing_axis=_tip_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            GraspBar,
            lambda world: [
                GraspBar(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    tip_grasp_axis=_tip_axis(world),
                    bar_center=_point(world),
                    bar_axis=_root_axis(world),
                    bar_length=0.4,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            WiggleInsert,
            lambda world: [
                WiggleInsert(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    hole_point=_point(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            InsertCylinder,
            lambda world: [
                InsertCylinder(
                    tip_link=_tip(world),
                    tip_P_tool=_tip_point(world),
                    hole_point=_point(world),
                    tip_V_axis=_tip_axis(world),
                    up_axis=_root_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            StayOnLineTask,
            lambda world: [
                StayOnLineTask(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    tip_P_tool=_tip_point(world),
                    root_P_start=_point(world),
                    root_P_end=Point3(0.1, 0.2, 0.5, reference_frame=_root(world)),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            MaxManipulability,
            lambda world: [
                MaxManipulability(root_link=_root(world), tip_link=_tip(world))
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            ArticulateGraspedMechanism,
            lambda world: [
                ArticulateGraspedMechanism(
                    environment_link=world.get_body_by_name("robot"),
                    tip_link=world.get_body_by_name("robot2"),
                    goal_joint_state=0.0,
                )
            ],
            WorldFixture.PRISMATIC_BOT2,
        ),
    ]


def _feature_cases() -> list[NodeRoundTripCase]:
    return [
        NodeRoundTripCase(
            AlignPerpendicular,
            lambda world: [
                AlignPerpendicular(
                    **_feature_fields(world),
                    tip_normal=_tip_axis(world),
                    reference_normal=_root_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            AngleGoal,
            lambda world: [
                AngleGoal(
                    **_feature_fields(world),
                    tip_vector=_tip_axis(world),
                    reference_vector=_root_axis(world),
                    lower_angle=0.1,
                    upper_angle=0.2,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            DistanceGoal,
            lambda world: [
                DistanceGoal(
                    **_feature_fields(world),
                    tip_point=_tip_point(world),
                    reference_point=_point(world),
                    lower_limit=0.1,
                    upper_limit=0.2,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            HeightGoal,
            lambda world: [
                HeightGoal(
                    **_feature_fields(world),
                    tip_point=_tip_point(world),
                    reference_point=_point(world),
                    lower_limit=0.1,
                    upper_limit=0.2,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            ReachPoint,
            lambda world: [
                ReachPoint(
                    **_feature_fields(world),
                    tip_point=_tip_point(world),
                    reference_point=_point(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            PerpendicularMonitor,
            lambda world: [
                PerpendicularMonitor(
                    **_feature_fields(world),
                    tip_normal=_tip_axis(world),
                    reference_normal=_root_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            AngleMonitor,
            lambda world: [
                AngleMonitor(
                    **_feature_fields(world),
                    tip_vector=_tip_axis(world),
                    reference_vector=_root_axis(world),
                    lower_angle=0.1,
                    upper_angle=0.2,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            DistanceMonitor,
            lambda world: [
                DistanceMonitor(
                    **_feature_fields(world),
                    tip_point=_tip_point(world),
                    reference_point=_point(world),
                    lower_limit=0.1,
                    upper_limit=0.2,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            HeightMonitor,
            lambda world: [
                HeightMonitor(
                    **_feature_fields(world),
                    tip_point=_tip_point(world),
                    reference_point=_point(world),
                    lower_limit=0.1,
                    upper_limit=0.2,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
    ]


def _monitor_cases() -> list[NodeRoundTripCase]:
    return [
        NodeRoundTripCase(
            PoseReached,
            lambda world: [
                PoseReached(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_pose=_transformation(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            PositionReached,
            lambda world: [
                PositionReached(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_point=_point(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            OrientationReached,
            lambda world: [
                OrientationReached(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_orientation=_orientation(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            PointingAt,
            lambda world: [
                PointingAt(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_point=_point(world),
                    pointing_axis=_tip_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            VectorsAligned,
            lambda world: [
                VectorsAligned(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    goal_normal=_root_axis(world),
                    tip_normal=_tip_axis(world),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            DistanceToLine,
            lambda world: [
                DistanceToLine(
                    root_link=_root(world),
                    tip_link=_tip(world),
                    center_point=_point(world),
                    line_axis=_root_axis(world),
                    line_length=0.4,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            JointPositionReached,
            lambda world: [
                JointPositionReached(connection=_connection(world), position=0.5)
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            LocalMinimumReached,
            lambda world: [LocalMinimumReached()],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            SetSeedConfiguration,
            lambda world: [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_mapping(
                        {_connection(world): 0.5}
                    )
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            SetOdometry,
            lambda world: [SetOdometry(base_pose=_transformation(world))],
            WorldFixture.CYLINDER_BOT_WORLD,
        ),
        NodeRoundTripCase(
            NoMonitoredTaskApproachingItsGoal,
            lambda world: [
                task := _joint_goal(world),
                NoMonitoredTaskApproachingItsGoal(monitored_tasks=[task]),
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            AnyMonitoredTaskShortOfItsGoal,
            lambda world: [
                task := _joint_goal(world),
                AnyMonitoredTaskShortOfItsGoal(monitored_tasks=[task]),
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            Stalled,
            lambda world: [task := _joint_goal(world), Stalled(monitored_node=task)],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            StillProgressing,
            _still_progressing_cancelling_the_motion,
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            _CancelBecauseNoProgress,
            _still_progressing_cancelling_the_motion,
            WorldFixture.PR2_WORLD,
        ),
    ]


def _joint_and_base_cases() -> list[NodeRoundTripCase]:
    return [
        NodeRoundTripCase(
            EndMotion, lambda world: [EndMotion()], WorldFixture.PR2_WORLD
        ),
        NodeRoundTripCase(Task, lambda world: [Task()], WorldFixture.PR2_WORLD),
        NodeRoundTripCase(
            StructureCopyMotionStatechartNode,
            lambda world: [StructureCopyMotionStatechartNode()],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            JointPositionList,
            lambda world: [_joint_goal(world)],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            JointVelocityLimit,
            lambda world: [JointVelocityLimit(connections=[_connection(world)])],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            MoveGripper,
            lambda world: [
                MoveGripper(
                    gripper=world.get_semantic_annotations_by_type(EndEffector)[0],
                    width=0.04,
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
        NodeRoundTripCase(
            MoveFingersToJointState,
            lambda world: [
                MoveFingersToJointState(
                    goal_state=world.get_semantic_annotations_by_type(EndEffector)[
                        0
                    ].get_joint_state_by_type(GripperState.CLOSE),
                )
            ],
            WorldFixture.PR2_WORLD,
        ),
    ]


def _collision_cases() -> list[NodeRoundTripCase]:
    return [
        NodeRoundTripCase(
            UpdateTemporaryCollisionRules,
            lambda world: [
                UpdateTemporaryCollisionRules(temporary_rules=[_collision_rule(world)])
            ],
            WorldFixture.CYLINDER_BOT_WORLD,
        ),
        NodeRoundTripCase(
            SetInitialTemporaryCollisionRules,
            lambda world: [
                SetInitialTemporaryCollisionRules(
                    temporary_rules=[_collision_rule(world)]
                )
            ],
            WorldFixture.CYLINDER_BOT_WORLD,
        ),
        NodeRoundTripCase(
            ExternalCollisionDistanceMonitor,
            lambda world: [
                ExternalCollisionDistanceMonitor(
                    body=_robot(world).root, threshold=0.05
                )
            ],
            WorldFixture.CYLINDER_BOT_WORLD,
        ),
        NodeRoundTripCase(
            SelfCollisionDistanceMonitor,
            lambda world: [
                SelfCollisionDistanceMonitor(
                    body_a=world.get_kinematic_structure_entity_by_name("l_thumb"),
                    body_b=world.get_kinematic_structure_entity_by_name("r_thumb"),
                    threshold=0.05,
                )
            ],
            WorldFixture.SELF_COLLISION_BOT_WORLD,
        ),
        *[
            NodeRoundTripCase(
                node_type,
                _external_collision_avoidance,
                WorldFixture.CYLINDER_BOT_WORLD,
            )
            for node_type in (
                ExternalCollisionAvoidance,
                _ExternalCollisionAvoidanceTask,
                _CancelBecauseExternalCollisionViolated,
            )
        ],
        *[
            NodeRoundTripCase(
                node_type,
                _self_collision_avoidance,
                WorldFixture.SELF_COLLISION_BOT_WORLD,
            )
            for node_type in (
                SelfCollisionAvoidance,
                _SelfCollisionAvoidanceTask,
                _CancelBecauseSelfCollisionViolated,
            )
        ],
    ]


def _cases() -> list[NodeRoundTripCase]:
    """
    :return: One case per motion statechart node class that needs no ROS.
    """
    return [
        *_joint_and_base_cases(),
        *_cartesian_cases(),
        *_feature_cases(),
        *_monitor_cases(),
        *_collision_cases(),
    ]


def _node_types_never_added_to_a_statechart() -> set[type[StatechartNode]]:
    """
    :return: The node classes that only hold what their subclasses share.
    """
    return {
        FeatureFunctionMonitor,
        _CollisionAvoidanceTask,
        _CancelBecauseCollisionViolated,
    }


# %% sender and receiver


@pytest.fixture(params=_cases(), ids=lambda case: case.node_type.__name__)
def case(request: pytest.FixtureRequest) -> NodeRoundTripCase:
    return request.param


@pytest.fixture()
def receiving_executor(
    case: NodeRoundTripCase, request: pytest.FixtureRequest
) -> StatechartExecutor:
    """
    :return: An executor with simulated motion control in the world of `case`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=request.getfixturevalue(case.world_fixture)),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            )
        ],
    )


@pytest.fixture()
def sending_context(
    case: NodeRoundTripCase, request: pytest.FixtureRequest
) -> StatechartContext:
    return create_context_with_motion_control(
        request.getfixturevalue(case.world_fixture)
    )


@pytest.fixture()
def cases() -> list[NodeRoundTripCase]:
    return _cases()


@pytest.fixture()
def node_types_needing_a_case() -> set[type[StatechartNode]]:
    """
    :return: The motion statechart node classes that need no ROS and are added to
        statecharts.
    """
    return (
        concrete_node_types(motion_statechart)
        - concrete_node_types(ros2_nodes)
        - _node_types_never_added_to_a_statechart()
    )


# %% round trips


class TestMotionStatechartNodeRoundTrips(NodeRoundTripTests):
    pass
