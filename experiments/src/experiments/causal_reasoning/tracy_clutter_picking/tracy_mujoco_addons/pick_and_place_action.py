"""
:class:`PickUpActionMujoco`/:class:`PlaceActionMujoco`: MuJoCo-driven siblings of
:class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`/
:class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`, matching their own
field interface (``object_designator``, ``arm``, ``grasp_description``/
``target_location``) so a caller can compose them into a
:func:`~coraplex.plans.factories.sequential` plan the same way, but with each own leaf
motion running plain Python (wrapped via :func:`~coraplex.plans.factories.code`) rather
than a Giskard motion mapping.

The real ``PickUpAction``/``PlaceAction`` build their own plan entirely from motion
designators, each of which ticks Giskard's own closed loop live against the world
model. That races
:class:`~semantic_digital_twin.adapters.multi_sim.MujocoSynchronizer`'s own
physics-thread state sync for a physically simulated robot -- see
:mod:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.equipment`'s own module docstring. These two
actions instead plan each reach by Giskard against an isolated scratch copy of the
world and play the resulting trajectory back by commanding the real MuJoCo actuators
(see :class:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.trajectory_planning.TrajectoryPlanner`).

Unlike ``PickUpAction``/``PlaceAction``, neither action here kinematically attaches or
detaches the object: the object is held only by real MuJoCo contact friction between
the fingers throughout, so a poor grasp visibly fails instead of being rescued by a
weld. Both actions are generic over any body and arm.

Both actions approach top-down (``grasp_description`` is accepted for interface parity
with ``PickUpAction``, but its own approach direction and vertical alignment are not yet
read); the gripper may be turned about that vertical axis via ``grasp_yaw`` so the
fingers meet an object across a chosen width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy
from typing_extensions import Dict, Optional

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import code
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation import (
    RealTimeSimulation,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.trajectory_planning import (
    RobotiqGripper,
    TrajectoryPlanner,
    arm_of,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Actuator, Body


def bounding_box_center_world(world: World, body: Body) -> numpy.ndarray:
    """
    A body's own collision bounding box centre, in the world root frame.

    :param world: The world ``body`` belongs to.
    :param body: The body to measure.
    :return: The centre.
    """
    bounding_box = body.collision[0].local_frame_bounding_box
    center_local = numpy.array(
        [
            (bounding_box.min_x + bounding_box.max_x) / 2,
            (bounding_box.min_y + bounding_box.max_y) / 2,
            (bounding_box.min_z + bounding_box.max_z) / 2,
        ]
    )
    root_transform_body = world.compute_forward_kinematics_np(world.root, body)
    return root_transform_body[:3, :3] @ center_local + root_transform_body[:3, 3]


@dataclass
class TopDownGraspGeometry:
    """
    Where to put an arm's tool frame so that the gripper's own finger midpoint (not the
    tool frame) ends up at a given point, approaching top-down.

    Cartesian planning places the tool frame itself at a goal, not where the fingers
    actually meet -- confirmed directly, targeting a shape's own centre that way put the
    tool frame there but left the fingers several centimetres away, closing on open air.
    Each fingertip's own collision bounding box centre (not its link origin) stands in
    for where that finger actually is, since the link origin sits at one edge of the
    fingertip mesh, not its geometric centre.
    """

    world: World
    """
    The world the poses are expressed in.
    """

    robot: Tracy
    """
    The robot whose gripper geometry corrects the target.
    """

    arm_side: Arms
    """
    Which arm's gripper geometry to use.
    """

    grasp_yaw: float = 0.0
    """
    Rotation of the gripper about the vertical approach axis, in radians.

    ``0`` closes the fingers along the world x-axis; a non-zero value turns the closing
    axis in the horizontal plane so the pads can meet an object across a chosen width.
    """

    def finger_midpoint_offset(self) -> numpy.ndarray:
        """
        :return: The fixed offset from the arm's own tool frame to its gripper's own
            finger-tip midpoint, in the tool frame's own local axes.
        """
        gripper = RobotiqGripper(self.arm_side)
        tool_frame = arm_of(self.robot, self.arm_side).end_effector.tool_frame
        root_transform_tool = self.world.compute_forward_kinematics_np(
            self.world.root, tool_frame
        )
        left_center = bounding_box_center_world(
            self.world, self.world.get_body_by_name(gripper.left_fingertip_name)
        )
        right_center = bounding_box_center_world(
            self.world, self.world.get_body_by_name(gripper.right_fingertip_name)
        )
        finger_midpoint = (left_center + right_center) / 2
        offset_in_root_frame = finger_midpoint - root_transform_tool[:3, 3]
        return root_transform_tool[:3, :3].T @ offset_in_root_frame

    def tool_frame_pose(self, x: float, y: float, z: float) -> Pose:
        """
        :param x: Where the finger midpoint should be along the world root's x-axis.
        :param y: Where the finger midpoint should be along the world root's y-axis.
        :param z: Where the finger midpoint should be along the world root's z-axis.
        :return: The tool frame pose that puts the finger midpoint there, top-down.
        """
        orientation = Pose.from_xyz_rpy(
            0, 0, 0, pitch=math.pi, yaw=self.grasp_yaw, reference_frame=self.world.root
        )
        tool_frame_rotation = orientation.to_rotation_matrix().evaluate()[:3, :3]
        finger_target = numpy.array([x, y, z])
        tool_frame_target = (
            finger_target - tool_frame_rotation @ self.finger_midpoint_offset()
        )
        return Pose.from_xyz_rpy(
            *tool_frame_target,
            pitch=math.pi,
            yaw=self.grasp_yaw,
            reference_frame=self.world.root,
        )


@dataclass
class MujocoArmAction(ActionDescription):
    """
    What the two actions share: the simulation they drive, the actuators they command,
    and the planner every reach goes through.
    """

    simulation: RealTimeSimulation
    """
    The running real-time simulation to drive.
    """

    actuators: Dict[str, Actuator]
    """
    Every joint's own actuator, keyed by joint name.
    """

    planner: TrajectoryPlanner = field(default_factory=TrajectoryPlanner)
    """
    Plans every motion and plays it back.
    """

    hover_clearance: float = 0.3
    """
    Height, in metres, above a body's own top face the tool centre point moves to before
    descending onto it -- clears obstacles during the horizontal part of each approach.

    ``0.3`` clears the Montessori board plus its three drawers with a wide margin; a
    smaller hover height left Giskard's own collision-avoidance solver too little
    vertical room to route the arm above the board at all.
    """

    grasp_yaw: float = 0.0
    """
    Rotation of the gripper about the vertical approach axis, in radians; see
    :attr:`TopDownGraspGeometry.grasp_yaw`.
    """

    def _reach(self, arm_side: Arms, goal_pose: Pose) -> None:
        """
        Plan a Cartesian reach against an isolated scratch copy of the world and play it
        back on the real, physically simulated one.

        Collision avoidance is off: a much more crowded scene than an open table (e.g.
        the Montessori board plus its three drawers) can make Giskard's own collision-
        avoidance solver repeatedly raise ``CollisionViolatedError`` even after widening
        clearances -- confirmed directly, still colliding with the board, a drawer, and
        even the target shape itself. Orientation is still constrained: every pose here
        shares the same fixed top-down orientation, and leaving it unconstrained let
        Giskard's own IK redundancy resolution drift the achieved orientation slightly
        on each leg.

        :param arm_side: Which arm's tool centre point should reach ``goal_pose``.
        :param goal_pose: Target pose for the arm's tool frame.
        """
        trajectory = self.planner.plan_cartesian_trajectory(
            self.world,
            arm_side,
            goal_pose,
            translation_only=False,
            avoid_collisions=False,
        )
        self.planner.follow_joint_trajectory(
            self.simulation, self.actuators, trajectory
        )

    def _geometry(self, arm_side: Arms) -> TopDownGraspGeometry:
        """
        :param arm_side: Which arm's gripper geometry to use.
        :return: The top-down grasp geometry of this action's grasp yaw.
        """
        return TopDownGraspGeometry(self.world, self.robot, arm_side, self.grasp_yaw)


@dataclass
class PickUpActionMujoco(MujocoArmAction):
    """
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`'s own field
    interface, but driven by direct MuJoCo actuator control; see this module's own
    docstring.
    """

    object_designator: Body = field(kw_only=True)
    """
    The body to pick up.
    """

    arm: Arms = field(kw_only=True)
    """
    Which arm picks it up.
    """

    grasp_description: GraspDescription = field(kw_only=True)
    """
    Accepted for interface parity with ``PickUpAction``; not yet read (see this module's
    own docstring) -- every grasp is currently a fixed top-down approach.
    """

    grasp_close_swing_clearance: float = 0.0435
    """
    Extra height, in metres, added on top of the object's own vertical centre when
    reaching the grasp pose, so the fingertip pads still clear the resting surface after
    closing.

    The pads aren't fixed relative to the tool frame: as the Robotiq-85 knuckle closes,
    each pad's own position travels ~1.35cm further out along the gripper's reach axis
    (measured directly: pad centre sits at 0.1208m from the gripper mount when open,
    0.1343m when closed) -- a pad that clears the table by that same ~1.35cm while open
    ends up flush with the table once closed.
    """

    squeeze_margin: Optional[float] = None
    """
    How far past the object's own half-width the fingers close, in metres; the planner's
    own
    :attr:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.trajectory_planning.TrajectoryPlanner.squeeze_margin`
    if not given.

    Raise it for an object held at a point or edge rather than a flat face.
    """

    grasp_half_width: Optional[float] = None
    """
    Half the width the pads are to meet the object across, in metres, for an object
    grasped across a known pair of faces; see
    :meth:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.trajectory_planning.TrajectoryPlanner.close_gripper_around`.

    Defaults to reading it off the object's bounding box in the gripper frame.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return code(self._run)

    def _run(self) -> None:
        geometry = self._geometry(self.arm)
        body_center = bounding_box_center_world(self.world, self.object_designator)
        pick_hover = geometry.tool_frame_pose(
            body_center[0], body_center[1], body_center[2] + self.hover_clearance
        )
        pick_grasp = geometry.tool_frame_pose(
            body_center[0],
            body_center[1],
            body_center[2] + self.grasp_close_swing_clearance,
        )

        self._reach(self.arm, pick_hover)
        self._reach(self.arm, pick_grasp)
        self.planner.close_gripper_around(
            self.simulation,
            self.actuators,
            self.robot,
            self.arm,
            self.object_designator,
            squeeze_margin=self.squeeze_margin,
            half_width=self.grasp_half_width,
        )
        self._reach(self.arm, pick_hover)


@dataclass
class PlaceActionMujoco(MujocoArmAction):
    """
    :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`'s own field
    interface, but driven by direct MuJoCo actuator control; see this module's own
    docstring.
    """

    object_designator: Body = field(kw_only=True)
    """
    The body to place; only used to release it, since this action does not kinematically
    attach it in the first place (see this module's own docstring).
    """

    target_location: Pose = field(kw_only=True)
    """
    Where to place :attr:`object_designator`.
    """

    arm: Arms = field(kw_only=True)
    """
    Which arm places it.
    """

    place_hover_clearance: float = 0.05
    """
    Height, in metres, above :attr:`target_location` the body is released at, rather
    than descending onto it exactly.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return code(self._run)

    def _run(self) -> None:
        geometry = self._geometry(self.arm)
        target_position = self.target_location.to_position()
        place_hover = geometry.tool_frame_pose(
            float(target_position.x),
            float(target_position.y),
            float(target_position.z) + self.hover_clearance,
        )
        place_pose = geometry.tool_frame_pose(
            float(target_position.x),
            float(target_position.y),
            float(target_position.z) + self.place_hover_clearance,
        )

        self._reach(self.arm, place_hover)
        self._reach(self.arm, place_pose)
        self.planner.set_gripper(
            self.simulation, self.actuators, self.robot, self.arm, GripperState.OPEN
        )
        self._reach(self.arm, place_hover)
