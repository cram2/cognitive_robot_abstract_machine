from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, Any, Dict

from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import NotOnASingleLevelException
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import execute_single, pause_until, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.navigation import MoveMotion
from coraplex.robot_plans.motions.robot_body import LookingMotion
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.joint_monitors import (
    JointPositionReached,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable_from, and_, ConditionType
from semantic_digital_twin.reasoning.predicates import allclose, InsideOf
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Level,
    Elevator,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
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
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)


@dataclass
class NavigateAction(ActionDescription):
    """
    Navigates the Robot to a position.
    """

    target_location: Pose
    """
    Where the robot should stand, and which way it should face given as the pose's
    x-axis.
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveMotion(
                self.robot.mobile_base.pose_facing(self.target_location),
                self.keep_joint_states,
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
class LookAtAction(ActionDescription):
    """
    Lets the robot look at a position.
    """

    target: Pose
    """
    Position at which the robot should look, given as 6D pose.
    """

    camera: Optional[Camera] = None
    """
    Camera that should be looking at the target.
    """

    @property
    def _action_plan(self) -> PlanNode:
        camera = self.camera or self.robot.get_default_camera()
        return execute_single(LookingMotion(target=self.target, camera=camera))


@dataclass
class GCSNavigateAction(ActionDescription):
    """
    Navigates the robot to a pose along a path through the environment's free space.

    The free space is decomposed into a graph of convex sets, so the robot drives around
    the furniture and walls between it and the target instead of straight at them.
    """

    target: Pose
    """
    Where the robot should stand at the end of the path, with its base.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([MoveMotion(waypoint) for waypoint in self._path()])

    def _navigation_map(self, floor_level: float) -> PlanarGraphOfBoundingBoxes:
        """
        The floor plan of everything the robot can drive on.

        :param floor_level: The height the robot's base stands at.
        :return: The navigation map covering the whole environment.
        """
        origin = HomogeneousTransformationMatrix(reference_frame=self.world.root)
        environment = SemanticEnvironmentAnnotation(
            root=self.world.root, _world=self.world
        )
        extent = environment.as_bounding_box_collection_at_origin(origin).bounding_box()
        search_space = BoundingBoxCollection(
            [
                VolumetricBoundingBox(
                    min_x=extent.min_x,
                    min_y=extent.min_y,
                    min_z=floor_level,
                    max_x=extent.max_x,
                    max_y=extent.max_y,
                    max_z=floor_level + self.robot.as_bounding_box_collection_in_frame(self.robot.root).bounding_box().scale.z,
                    origin=origin,
                )
            ],
            self.world.root,
        )
        return PlanarGraphOfBoundingBoxes.navigation_map_from_world(
            self.world,
            search_space=search_space,
            bloat_obstacles=self.robot.mobile_base.base_radius,
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
        poses = [
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(waypoint.x, waypoint.y, self.robot.root.global_pose.z, waypoint.reference_frame),
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
            ).to_pose()
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
        return self._navigation_map(float(base_pose.z)).path_from_to(
            Point2.from_pose(base_pose), Point2.from_pose(self.target)
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
                NavigateAction(self._pose_infront_of_elevator),
                pause_until(
                    [
                        NavigateAction(
                            Pose.from_xyz_rpy(
                                z=self._height_in_cabin,
                                reference_frame=self.elevator.root,
                            )
                        )
                    ],
                    monitor=self._elevator_open_at_floor(self._current_floor),
                ),
                ReAttachNode(body=self.robot.root, new_parent=self.elevator.root),
                pause_until(
                    [NavigateAction(self._pose_infront_of_elevator)],
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
            self.world.transform(self.robot.root.global_transform, self.elevator.root)
            .z
        )

    def _elevator_open_at_floor(self, target_floor: Level) -> Parallel:
        """
        Observes True once the cabin serves :attr:`target_floor` with its doors open.
        """
        nodes = []
        for door in self.elevator.doors:
            connection = door.mechanical_joint.root.parent_connection
            nodes.append(
                JointPositionReached(
                    connection=connection,
                    position=connection.dof.limits.upper.position,
                    threshold=self.arrival_threshold,
                    name=f"{door.name}Open",
                )
            )
        nodes.append(
            JointPositionReached(
                connection=self.elevator.mechanical_joint.root.parent_connection,
                position=self.elevator.drive_position_for_floor(target_floor),
                threshold=self.arrival_threshold,
                name="ElevatorAtTargetFloor",
            )
        )
        return Parallel(
            nodes,
            name="ElevatorOpenAtTargetFloor",
        )
