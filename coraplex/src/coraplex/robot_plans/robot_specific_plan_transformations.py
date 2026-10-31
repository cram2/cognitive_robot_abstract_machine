"""
Plan transformations that let particular robots carry out an action with a motion of
their own.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.exceptions import ClosingGoalNotFound, ToolPathNotFound
from coraplex.plans.plan_transformation import (
    ConfigurationTransformation,
    ParallelInsertion,
    PlanTransformation,
)
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.tool_based import ToolMotionAction
from coraplex.robot_plans.actions.core.container import (
    CloseAction,
    MovesContainerByItsHandle,
)
from coraplex.robot_plans.actions.core.navigation import DrivesBase
from coraplex.robot_plans.mixins import MovesToolCenterPoint
from cramph.composites import ControlStructureNode, Parallel
from cramph.node import StatechartNode
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.goals.base_pose_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.cartesian_goals import CartesianPoseStraight
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPositionTrajectory,
    HoldOrientation,
)
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3
from semantic_digital_twin.world_description.connections import DifferentialDrive

# %% keeping the torso upright while using a tool


@dataclass
class KeepTheTorsoUprightWhileUsingATool(ParallelInsertion[ToolMotionAction]):
    """
    Keeps Justin's torso upright while it moves a tool, which no other robot needs: its
    torso would otherwise lean into the motion.
    """

    def is_applicable(self, plan_node: ToolMotionAction) -> bool:
        return isinstance(plan_node.robot, Justin)

    def anchor(self, plan_node: ToolMotionAction) -> StatechartNode:
        """
        :return: The goal moving the tool along its path, alongside which the torso is
            held.
        :raises ToolPathNotFound: If the action holds no goal moving the tool along a
            path.
        """
        for node in plan_node.descendants:
            if isinstance(node, CartesianPositionTrajectory):
                return node
        raise ToolPathNotFound(plan_node)

    def nodes_to_insert(self, plan_node: ToolMotionAction) -> list[StatechartNode]:
        root = plan_node.controlled_root
        torso_tip = plan_node.robot.mobile_base.torso.tip
        return [
            AlignPlanes(
                tip_link=torso_tip,
                root_link=root,
                tip_normal=Vector3.X(torso_tip),
                goal_normal=Vector3.Z(root),
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value,
            )
        ]


# %% driving a differential drive


@dataclass
class DriveWithTheDifferentialDrive(ConfigurationTransformation[DrivesBase]):
    """
    Moves the base of a robot driven by a differential drive with a goal for that drive,
    which turns towards a pose, drives there and turns again, instead of a goal that
    asks the base to move sideways.

    Applies simulated or not.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return isinstance(plan_node.robot.drive, DifferentialDrive)

    def apply(self, plan_node: DrivesBase) -> None:
        plan_node.base_goal_type = DifferentialDriveBaseGoal


# %% keeping the base from turning while moving the tool


@dataclass
class KeepStretchFromTurningWhileMovingItsTool(ParallelInsertion[MovesToolCenterPoint]):
    """
    Holds the orientation of a Stretch's base while its tool moves, so that the base
    only drives along its heading, as the tool's sixth degree of freedom.

    The base does not turn to bring a target into reach, which would sweep it through
    wide arcs: a target out of reach fails, and where the robot stands is up to the
    plan. Left as they are: actions driving the base, actions moving a container by its
    handle, and actions below one that holds the base already.
    """

    def is_applicable(self, plan_node: MovesToolCenterPoint) -> bool:
        robot = plan_node.robot
        return (
            isinstance(robot, Stretch)
            and robot.mobile_base.full_body_controlled
            and not any(
                isinstance(node, MovesContainerByItsHandle)
                for node in [plan_node, *plan_node.path]
            )
            and not any(isinstance(node, DrivesBase) for node in plan_node.descendants)
            and not any(
                self._holds_the_base(ancestor, robot) for ancestor in plan_node.path
            )
        )

    def anchor(self, plan_node: MovesToolCenterPoint) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: MovesToolCenterPoint) -> list[StatechartNode]:
        return [
            HoldOrientation(
                root_link=plan_node.world.root, tip_link=plan_node.robot.root
            )
        ]

    @staticmethod
    def _holds_the_base(node: StatechartNode, robot: Stretch) -> bool:
        """
        :return: Whether `node` holds the orientation of the base of `robot` beside its
            other children, as this transformation makes an action do. A hold further
            down, such as that of an earlier action, does not cover the others.
        """
        return isinstance(node, Parallel) and any(
            type(child) is HoldOrientation and child.tip_link is robot.root
            for child in node.nodes
        )


# %% moving the tool center point in a straight line


@dataclass
class MoveStretchToolCenterPointInAStraightLine(PlanTransformation[CartesianPose]):
    """
    Moves the tool center point of a Stretch to each of its targets along a straight
    line, by giving every pose goal that an action moving it sets relative to its
    controlled root the tasks of a
    :class:`~giskardpy.motion_statechart.goals.cartesian_goals.CartesianPoseStraight`
    reaching the same pose. Pose goals inside another goal are left as they are.
    """

    def is_applicable(self, plan_node: CartesianPose) -> bool:
        if type(plan_node) is not CartesianPose:
            return False
        owner = next(
            (
                node
                for node in plan_node.path
                if isinstance(node, Action)
                or not isinstance(node, ControlStructureNode)
            ),
            None,
        )
        return (
            isinstance(owner, MovesToolCenterPoint)
            and isinstance(owner.robot, Stretch)
            and plan_node.root_link is owner.controlled_root
        )

    def apply(self, plan_node: CartesianPose) -> None:
        straight = CartesianPoseStraight(
            name=plan_node.name,
            root_link=plan_node.root_link,
            tip_link=plan_node.tip_link,
            goal_pose=plan_node.goal_pose,
            translation_threshold=plan_node.translation_threshold,
            orientation_threshold=plan_node.orientation_threshold,
            binding_policy=plan_node.binding_policy,
        )
        # the tasks of a pose goal come before the nodes accompanying it
        for task, straight_task in zip(list(plan_node.nodes), straight.create_tasks()):
            plan_node.statechart.replace(task, straight_task)


# %% closing a container


@dataclass
class KeepStretchInFrontOfTheContainerItCloses(PlanTransformation[CloseAction]):
    """
    Keeps a kinematically simulated Stretch standing in front of the container it closes
    and holding on to the handle, so that it closes the container by pushing its arm
    forward.
    """

    def is_applicable(self, plan_node: CloseAction) -> bool:
        return isinstance(plan_node.robot, Stretch)

    @staticmethod
    def closing_goal(plan_node: CloseAction) -> ArticulateGraspedMechanism:
        """
        :param plan_node: The close action.
        :return: The goal closing the container.
        :raises ClosingGoalNotFound: If the action holds no goal closing a container,
            which it has none of before it is expanded.
        """
        for node in plan_node.descendants:
            if isinstance(node, ArticulateGraspedMechanism):
                return node
        raise ClosingGoalNotFound(plan_node)

    def apply(self, plan_node: CloseAction) -> None:
        closing_goal = self.closing_goal(plan_node)
        handle = closing_goal.environment_link
        tool_frame = closing_goal.tip_link
        robot_root = plan_node.robot.root
        plan_node.statechart.replace(
            closing_goal,
            Parallel(
                nodes=[
                    CartesianPose(
                        name="Keep holding handle",
                        root_link=handle,
                        tip_link=tool_frame,
                        goal_pose=Pose(reference_frame=tool_frame),
                    ),
                    AlignPlanes(
                        root_link=plan_node.world.root,
                        tip_link=robot_root,
                        goal_normal=Vector3.X(handle),
                        tip_normal=Vector3.NEGATIVE_Y(robot_root),
                    ),
                    ArticulateGraspedMechanism(
                        tip_link=tool_frame,
                        environment_link=handle,
                        goal_joint_state=closing_goal.goal_joint_state,
                    ),
                ]
            ),
        )
