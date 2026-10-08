"""
:class:`PickUpActionMujoco`/:class:`PlaceActionMujoco`: siblings of
:class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`/
:class:`~coraplex.robot_plans.actions.core.placing.PlaceAction` for a physically
simulated robot, matching their own field interface (``object_designator``, ``arm``,
``grasp_description``/``target_location``) so a caller can compose them into a
:func:`~coraplex.plans.factories.sequential` plan the same way, with each leaf motion
run live by Giskard against the MuJoCo-mirrored world (see :class:`~experiments.causal_r
easoning.tracy_clutter_picking.tracy_mujoco_addons.live_motion.MotionRunner`).

Unlike ``PickUpAction``/``PlaceAction``, neither action here kinematically attaches or
detaches the object: the object is held only by real MuJoCo contact friction between the
fingers throughout, so a poor grasp visibly fails instead of being rescued by a weld.
Both actions are generic over any body and arm.

Both actions approach top-down (``grasp_description`` is accepted for interface parity
with ``PickUpAction``, but its own approach direction and vertical alignment are not yet
read); the gripper may be turned about that vertical axis via ``grasp_yaw`` so the
fingers meet an object across a chosen width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy
from typing_extensions import Optional

from coraplex.plans.factories import code
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.live_motion import (
    MotionRunner,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def centre_in_root(world: World, body: Body) -> numpy.ndarray:
    """
    :param world: The world ``body`` belongs to.
    :param body: The body to measure.
    :return: The centre of the body's own collision bounding box, in the world root
        frame.
    """
    return (
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .center.to_np()[:3]
        .ravel()
    )


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

    arm: Arm
    """
    The arm whose gripper geometry corrects the target.
    """

    grasp_yaw: float = 0.0
    """
    Rotation of the gripper about the vertical approach axis, in radians.

    ``0`` closes the fingers along the world x-axis; a non-zero value turns the closing
    axis in the horizontal plane so the pads can meet an object across a chosen width.
    """

    def _centre_in_root(self, body: Body) -> numpy.ndarray:
        """
        :param body: The body to measure.
        :return: Its centre in the world root frame.
        """
        return centre_in_root(self.world, body)

    def finger_midpoint_offset(self) -> numpy.ndarray:
        """
        :return: The fixed offset from the arm's own tool frame to its gripper's own
            finger-tip midpoint, in the tool frame's own local axes.
        """
        gripper = self.arm.end_effector
        root_transform_tool = self.world.compute_forward_kinematics_np(
            self.world.root, gripper.tool_frame
        )
        thumb_center = self._centre_in_root(gripper.thumb.tip)
        finger_center = self._centre_in_root(gripper.finger.tip)
        finger_midpoint = (thumb_center + finger_center) / 2
        offset_in_root_frame = finger_midpoint - root_transform_tool[:3, 3]
        return root_transform_tool[:3, :3].T @ offset_in_root_frame

    def grasp_frame(self, x: float, y: float, z: float) -> Pose:
        """
        The grasp frame that meets an object at a point from straight above.

        A grasp frame points its x-axis the way the gripper travels and its y-axis along
        the axis the fingers close along (see
        :class:`~semantic_digital_twin.grasping.grasp_candidates.GraspCandidate`), so
        approaching top-down is an x-axis pointing down and a closing axis turned by
        :attr:`grasp_yaw` in the horizontal plane.

        :param x: Where the fingers should meet along the world root's x-axis.
        :param y: Where the fingers should meet along the world root's y-axis.
        :param z: Where the fingers should meet along the world root's z-axis.
        :return: The grasp frame at that point.
        """
        root = self.world.root
        return Pose(
            position=Point3(x, y, z, reference_frame=root),
            orientation=RotationMatrix.from_vectors(
                x=Vector3(0.0, 0.0, -1.0, reference_frame=root),
                y=Vector3(
                    -math.cos(self.grasp_yaw),
                    -math.sin(self.grasp_yaw),
                    0.0,
                    reference_frame=root,
                ),
                reference_frame=root,
            ).quaternion,
            reference_frame=root,
        )

    def tool_frame_pose(self, x: float, y: float, z: float) -> Pose:
        """
        The gripper states where its own approach and closing axes point, so the grasp
        frame becomes a tool frame goal without this module knowing which way the tool
        frame faces. Only the offset between the tool frame and where the fingers
        actually meet is left to correct here.

        :param x: Where the finger midpoint should be along the world root's x-axis.
        :param y: Where the finger midpoint should be along the world root's y-axis.
        :param z: Where the finger midpoint should be along the world root's z-axis.
        :return: The tool frame pose that puts the finger midpoint there, top-down.
        """
        oriented = self.arm.end_effector.tool_frame_goal(self.grasp_frame(x, y, z))
        rotation = numpy.array(oriented.rotation_matrix.evaluate()[:3, :3], dtype=float)
        tool_frame_target = numpy.array([x, y, z]) - rotation @ (
            self.finger_midpoint_offset()
        )
        return Pose(
            position=Point3(*tool_frame_target, reference_frame=self.world.root),
            orientation=oriented.quaternion,
            reference_frame=self.world.root,
        )


@dataclass
class MujocoArmAction(ActionDescription):
    """
    What the two actions share: the runner every motion goes through.
    """

    runner: MotionRunner
    """
    Runs every motion live against the physically simulated world.
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

    def _reach(self, arm: Arm, goal_pose: Pose) -> None:
        """
        Move an arm's tool frame to a Cartesian pose.

        Collision avoidance is off: in a crowded scene Giskard's own collision-avoidance
        solver treats the very object being approached as an obstacle and raises rather
        than reaching it. Orientation is still constrained: every pose here shares the
        same fixed top-down orientation, and leaving it unconstrained lets Giskard's own
        redundancy resolution drift the achieved orientation slightly on each leg.

        :param arm: The arm whose tool centre point should reach ``goal_pose``.
        :param goal_pose: Target pose for the arm's tool frame.
        """
        self.runner.reach(
            self.robot,
            arm,
            goal_pose,
            translation_only=False,
            avoid_collisions=False,
        )

    def _geometry(self, arm: Arm) -> TopDownGraspGeometry:
        """
        :param arm: The arm whose gripper geometry to use.
        :return: The top-down grasp geometry of this action's grasp yaw.
        """
        return TopDownGraspGeometry(self.world, arm, self.grasp_yaw)


@dataclass
class PickUpActionMujoco(MujocoArmAction):
    """
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`'s own field
    interface, run live against a physically simulated robot; see this module's own
    docstring.
    """

    object_designator: Body = field(kw_only=True)
    """
    The body to pick up.
    """

    arm: Arm = field(kw_only=True)
    """
    The arm that picks it up.
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
    How far past the object's own half-width the fingers close, in metres; the runner's
    own
    :attr:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.live_motion.MotionRunner.squeeze_margin`
    if not given.

    Raise it for an object held at a point or edge rather than a flat face.
    """

    grasp_half_width: Optional[float] = None
    """
    Half the width the pads are to meet the object across, in metres, for an object
    grasped across a known pair of faces; see :meth:`~experiments.causal_reasoning.tracy
    _clutter_picking.tracy_mujoco_addons.live_motion.MotionRunner.close_gripper_around`.

    Defaults to reading it off the object's bounding box in the gripper frame.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return code(self._run)

    def _run(self) -> None:
        geometry = self._geometry(self.arm)
        body_center = centre_in_root(self.world, self.object_designator)
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
        self.runner.close_gripper_around(
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
    interface, run live against a physically simulated robot; see this module's own
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

    arm: Arm = field(kw_only=True)
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
        target_position = self.target_location.position
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
        self.runner.set_gripper(self.robot, self.arm, GripperState.OPEN)
        self._reach(self.arm, place_hover)
