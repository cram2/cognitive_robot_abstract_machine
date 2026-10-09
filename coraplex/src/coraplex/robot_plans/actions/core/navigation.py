from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Any, Dict

from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import NoFloorBelowRobot, NotOnASingleLevelException
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import execute_single, pause_until, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import (
    CameraTargetParameters,
    NavigationTargetParameter,
    LookTargetParameter,
)
from coraplex.robot_plans.motions.navigation import MoveMotion, TurnMotion
from coraplex.robot_plans.motions.robot_body import LookingMotion
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.joint_monitors import (
    JointPositionReached,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable_from, and_, ConditionType
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.reasoning.predicates import allclose, InsideOf
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Level,
    Elevator,
    Floor,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Pose2D,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox


@dataclass
class NavigateAction(ActionDescription, NavigationTargetParameter):
    """
    Navigates the Robot to a position.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveMotion(
                target_location=Pose2D.from_pose(
                    self.robot.mobile_base.pose_facing(self.target_location)
                )
            )
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The robot needs to have a drive and the target location needs to be free from
        obstacles.
        """
        drive_variable = variable_from(context.robot.drive is not None)
        return and_(
            is_pose_free_for_robot(context.robot, variables["target_location"]),
            drive_variable,
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The robot needs to be within 3 cm of where the heading puts its base.
        """
        return allclose(
            variable_from(context.robot.root).global_pose,
            context.robot.mobile_base.pose_facing(kwargs["target_location"]),
            atol=0.03,
        )


@dataclass
class LookAtAction(ActionDescription, CameraTargetParameters):
    """
    Lets the robot look at a position.
    """

    @property
    def _action_plan(self) -> PlanNode:
        camera = self.camera or self.robot.get_default_camera()
        return execute_single(LookingMotion(target=self.target, camera=camera))


@dataclass
class FaceAtAction(ActionDescription, LookTargetParameter):
    """
    Turns the robot's base on the spot until its front faces a target.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(TurnMotion(self.target))


@dataclass
class PathPlanningNavigateAction(ActionDescription, NavigationTargetParameter):
    """
    Navigates the robot to a pose along a path through the environment's free space.

    The free space is decomposed into a graph of convex sets, so the robot drives around
    the furniture and walls between it and the target instead of straight at them.

    This works for obstacles which are known in the environment beforehand, not for
    those added during navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [MoveMotion(target_location=waypoint) for waypoint in self._path()]
        )

    @property
    def _floor(self) -> Floor:
        """
        The floor the robot stands on, whose free space the path is laid out in.

        A world with several storeys puts more than one floor below the robot; the one
        it stands on is the topmost of those it stands within the footprint of.

        :raises NoFloorBelowRobot: If the robot stands over no annotated floor.
        :return: The floor the robot drives on.
        """
        floors_below = [
            floor
            for floor in self.world.get_semantic_annotations_by_type(Floor)
            if self._stands_on(floor)
        ]
        if not floors_below:
            raise NoFloorBelowRobot(self.robot)
        return max(floors_below, key=lambda floor: self._extent_of(floor).max_z)

    def _extent_of(self, floor: Floor) -> VolumetricBoundingBox:
        """
        :param floor: The floor to measure.
        :return: The floor's bounding box in the world's root frame.
        """
        return floor.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=self.world.root)
        ).bounding_box()

    def _stands_on(self, floor: Floor) -> bool:
        """
        :param floor: The floor to test.
        :return: Whether the robot's base rests within this floor's footprint and no
            lower than its top.
        """
        extent = self._extent_of(floor)
        base_pose = self.robot.root.global_pose
        return (
            extent.min_x <= float(base_pose.x) <= extent.max_x
            and extent.min_y <= float(base_pose.y) <= extent.max_y
            and extent.max_z <= float(base_pose.z)
        )

    def _path(self) -> list[Pose2D]:
        """
        The poses the robot drives to, one per leg of the path.

        Each pose faces the waypoint after it, so the leg leaving a waypoint no longer
        has to begin by turning. The waypoint the robot already stands on is left out,
        and the last pose is the requested target.

        .. note::
            The orientation aims the base's x-axis, which is the axis a drive travels
            along, rather than the base's
            :attr:`~semantic_digital_twin.robots.robot_parts.MobileBase.forward_axis`.
            The two differ on a base whose front is not its direction of travel, and it
            is travel that these orientations exist to line up.

        :return: The poses to drive to, in order.
        """
        waypoints = self._waypoints()
        poses = [
            Pose2D.from_position_and_yaw(
                waypoint,
                yaw=np.arctan2(
                    float(next_waypoint.y - waypoint.y),
                    float(next_waypoint.x - waypoint.x),
                ),
            )
            for waypoint, next_waypoint in zip(waypoints[1:], waypoints[2:])
        ]
        return poses + [self.target_location]

    def _waypoints(self) -> list[Point2]:
        """
        The points the robot travels through to get from where it stands to the target.

        :return: The path, beginning at the robot's own position and ending at the
            target's.
        """
        base_pose = self.robot.root.global_pose
        free_space = self._floor.planar_free_space(
            max_height=self.robot.as_bounding_box_collection_in_frame(self.robot.root)
            .bounding_box()
            .scale.z,
            bloat_obstacles=self.robot.mobile_base.base_radius,
        )
        return free_space.path_from_to(
            Point2.from_pose(base_pose), self.target_location.position
        )


@dataclass
class ElevatorNavigation(ActionDescription):
    """
    Navigates a robot to another level of a building using an elevator, the robot drives
    in the elevator and waits there until the doors open again and the elevator is at
    the right level.
    """

    elevator: Elevator
    """
    Elevator the robot rides.
    """

    target_floor: Level
    """
    Level of the building the robot should end up on.
    """

    exit_clearance: float = field(default=0.5, kw_only=True)
    """
    Distance the robot keeps from the elevator's opening after driving out, on top of
    half the cabin's depth.
    """

    arrival_threshold: float = field(default=0.01, kw_only=True)
    """
    Position error within which the elevator's drive and doors count as having arrived.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(target_location=self._pose_infront_of_elevator),
                pause_until(
                    [
                        NavigateAction(
                            target_location=self._on_cabin_floor(
                                Pose2D(reference_frame=self.elevator.root)
                            )
                        )
                    ],
                    monitor=self._elevator_open_at_floor(self._current_floor),
                ),
                ReAttachNode(body=self.robot.root, new_parent=self.elevator.root),
                pause_until(
                    [NavigateAction(target_location=self._pose_infront_of_elevator)],
                    monitor=self._elevator_open_at_floor(self.target_floor),
                ),
                ReAttachNode(body=self.robot.root, new_parent=self.world.root),
            ]
        )

    @property
    def _current_floor(self) -> Level:
        """
        Finds the floor the robot is currently on, based on its position in the world.

        Raises :class:`WrongLevelException` if the robot is not on any floor or on
        multiple floors at once.
        :return: The semantic annotation for the floor
        """
        current_floor = [
            floor
            for floor in self.world.get_semantic_annotations_by_type(Level)
            if InsideOf(self.robot.bodies_with_collision[0], floor.root)() > 0.9
        ]
        if len(current_floor) == 0:
            raise NotOnASingleLevelException("Robot is not on any recognized floor.")
        if len(current_floor) > 1:
            raise NotOnASingleLevelException("Robot is on multiple floors at once.")
        return current_floor[0]

    @property
    def _pose_infront_of_elevator(self) -> Pose2D:
        """
        Where the robot stands in front of the cabin's opening.
        """
        return self._on_cabin_floor(
            Pose2D(
                x=self.elevator.hole_direction[0]
                * (self.elevator.scale.x / 2 + self.exit_clearance),
                reference_frame=self.elevator.root,
            )
        )

    def _on_cabin_floor(self, spot: Pose2D) -> Pose2D:
        """
        :param spot: A spot given relative to the cabin's centre.
        :return: The same spot, expressed on the cabin's floor, which is flush with the
            level the cabin stops at.
        """
        return self.world.transform(spot, self.elevator.require_supporting_surface())

    def _elevator_open_at_floor(self, target_floor: Level) -> Parallel:
        """
        Observes True once the cabin serves :attr:`target_floor` with its doors open.
        """
        nodes = []
        for door in self.elevator.doors:
            if door.movable_joint is None:
                raise MissingMovableJointError(door)
            nodes.append(
                JointPositionReached(
                    connection=door.movable_joint,
                    position=door.movable_joint.dof.limits.upper.position,
                    threshold=self.arrival_threshold,
                    name=f"{door.name}Open",
                )
            )
        if self.elevator.movable_joint is None:
            raise MissingMovableJointError(self.elevator)
        nodes.append(
            JointPositionReached(
                connection=self.elevator.movable_joint,
                position=self.elevator.drive_position_for_floor(target_floor),
                threshold=self.arrival_threshold,
                name="ElevatorAtTargetFloor",
            )
        )
        return Parallel(
            nodes,
            name="ElevatorOpenAtTargetFloor",
        )
