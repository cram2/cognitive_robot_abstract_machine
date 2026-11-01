from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field


from coraplex.exceptions import NoFloorBelowRobot, NotOnASingleLevelException
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import (
    Action,
    GoalAction,
    ParallelAction,
    SequentialAction,
)
from coraplex.robot_plans.mixins import ReattachesBranches
from cramph.composites import Parallel, PausedUntilTrue
from giskardpy.motion_statechart.goals.base_pose_goals import (
    MobileBasePoseGoal,
    DifferentialDriveBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.monitors.joint_monitors import (
    JointPositionReached,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Level,
    Elevator,
    Floor,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    HomogeneousTransformationMatrix,
    Point2,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox

# %% driving the base


@dataclass(eq=False, repr=False, kw_only=True)
class DrivesBase(Action, ABC):
    """
    Base class for the actions that move the robot's base to a pose.
    """

    base_pose_threshold: float = DifferentialDriveBaseGoal.threshold
    """
    How far the base may be from a pose it drives to, in meters and in radians, to count
    as arrived.
    """

    base_goal_type: type[MobileBasePoseGoal] = field(
        default=HolonomicBaseGoal, init=False
    )
    """
    The kind of node putting the base at the poses this action drives it to, as set by
    the plan transformations configuring the action.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NavigateAction(DrivesBase, GoalAction):
    """
    Navigates the Robot to a position.
    """

    target_location: Pose
    """
    Where the robot should stand, and which way it should face given as the pose's
    x-axis.
    """

    def create_goal(self) -> StatechartNode:
        return self.base_goal_type.moving_base_of(
            self.robot,
            self.robot.mobile_base.pose_facing(self.target_location),
            self.base_pose_threshold,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class LookAtAction(ParallelAction):
    """
    Lets the robot look at a position, following it while the look runs when it is given
    relative to a body that moves.
    """

    target: Pose
    """
    Position at which the robot should look, given as 6D pose.
    """

    camera: Camera | None = None
    """
    Camera that should be looking at the target.
    """

    def create_steps(self) -> list[StatechartNode]:
        camera = self.camera or self.robot.get_default_camera()
        pointing = Pointing(
            root_link=self.robot.get_torso().root,
            tip_link=camera.root,
            goal_point=self._world_P_target(),
            pointing_axis=camera.viewing_direction,
        )
        return [pointing, *self.keep_base_still()]

    def _world_P_target(self) -> Point3:
        """
        :return: Where :attr:`target` is in the world frame, as an expression of where
            the body it is given relative to stands while the look runs.
        """
        target_frame = self.target.reference_frame or self.world.root
        world_T_target_frame = self.world.compose_forward_kinematics_expression(
            self.world.root, target_frame
        )
        world_T_target_frame.reference_frame = self.world.root
        world_T_target_frame.child_frame = target_frame
        return world_T_target_frame @ self.target.position


@dataclass(eq=False, repr=False, kw_only=True)
class FaceAtAction(ParallelAction):
    """
    Turns the robot's base on the spot until its front faces a target.

    The base keeps the position it has when the action starts, so the turn is towards
    the target from wherever an earlier action left it.
    """

    target: Pose
    """
    What to face; only its horizontal position matters.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            Pointing(
                root_link=self.world.root,
                tip_link=self.robot.root,
                goal_point=self._target_at_base_height(),
                pointing_axis=Vector3(
                    *self.robot.mobile_base.forward_axis.to_np()[:3],
                    reference_frame=self.robot.root,
                ),
            ),
            CartesianPosition(
                root_link=self.world.root,
                tip_link=self.robot.root,
                goal_point=Point3(reference_frame=self.robot.root),
            ),
        ]

    def _target_at_base_height(self) -> Point3:
        """
        :return: :attr:`target` moved vertically to the height of the base, which can
            only turn about the vertical and so can only point level.
        """
        root_P_target = self.world.transform(self.target, self.world.root).position
        root_P_target.z = self.robot.root.global_pose.z
        return root_P_target


@dataclass(eq=False, repr=False, kw_only=True)
class PathPlanningNavigateAction(DrivesBase, SequentialAction):
    """
    Navigates the robot to a pose along a path through the environment's free space.

    The free space is decomposed into a graph of convex sets, so the robot drives around
    the furniture and walls between it and the target instead of straight at them.

    This works for obstacles which are known in the environment beforehand, not for
    those added during navigation.
    """

    target: Pose
    """
    Where the robot should stand at the end of the path, with its base.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            self.base_goal_type.moving_base_of(
                self.robot, waypoint, self.base_pose_threshold
            )
            for waypoint in self._path()
        ]

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

    def _path(self) -> list[Pose]:
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
        base_height = self.world.transform(
            self.robot.root.global_transform, waypoints[0].reference_frame
        ).z
        poses = [
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(waypoint.x, waypoint.y, base_height, waypoint.reference_frame),
                RotationMatrix.from_vectors(
                    x=Vector3(
                        next_waypoint.x - waypoint.x,
                        next_waypoint.y - waypoint.y,
                        0,
                        reference_frame=waypoint.reference_frame,
                    ),
                    z=Vector3.Z(),
                    reference_frame=waypoint.reference_frame,
                ),
                reference_frame=waypoint.reference_frame,
            ).pose
            for waypoint, next_waypoint in zip(waypoints[1:], waypoints[2:])
        ]
        return poses + [self.target]

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
            Point2.from_pose(base_pose), Point2.from_pose(self.target)
        )


@dataclass(eq=False, repr=False, kw_only=True)
class ElevatorNavigation(ReattachesBranches, SequentialAction):
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

    exit_clearance: float = 0.5
    """
    Distance the robot keeps from the elevator's opening after driving out, on top of
    half the cabin's depth.
    """

    arrival_threshold: float = 0.01
    """
    Position error within which the elevator's drive and doors count as having arrived.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            NavigateAction(target_location=self._pose_infront_of_elevator),
            PausedUntilTrue(
                monitor=self._elevator_open_at_floor(self._current_floor),
                monitored_node=NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        z=self._height_in_cabin,
                        reference_frame=self.elevator.root,
                    )
                ),
            ),
            *self.reattach(self.robot.root, self.elevator.root),
            PausedUntilTrue(
                monitor=self._elevator_open_at_floor(self.target_floor),
                monitored_node=NavigateAction(
                    target_location=self._pose_infront_of_elevator
                ),
            ),
            *self.reattach(self.robot.root, self.world.root),
        ]

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
    def _pose_infront_of_elevator(self):
        return Pose.from_xyz_rpy(
            x=self.elevator.hole_direction[0]
            * (self.elevator.scale.x / 2 + self.exit_clearance),
            z=self._height_in_cabin,
            reference_frame=self.elevator.root,
        )

    @property
    def _height_in_cabin(self) -> float:
        """
        The robot's height in the cabin's frame.

        Taken from where the robot stands now, because it is the same throughout the
        ride and the robot's drive cannot change it anyway.
        """
        return float(
            self.world.transform(self.robot.root.global_transform, self.elevator.root).z
        )

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
            nodes=nodes,
            name="ElevatorOpenAtTargetFloor",
        )
