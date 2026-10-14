"""
Picking one carton up, top-down, as an ordinary coraplex plan.

Every motion is a coraplex motion designator, so a plan performed inside
:class:`~coraplex.execution_environment.PhysicallySimulatedRobot` is carried out by the
robot's own servos with the physics stepping between control cycles. What this action
adds is only what is particular to this grasp: aiming the pads rather than the tool
frame at the carton, closing them to its own width, and waiting for the servos to catch
up before the arm moves on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from coraplex.plans.factories import code, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.gripper import (
    MoveFingersToWidthMotion,
    MoveToolCenterPointMotion,
)
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Optional

from experiments.causal_reasoning.tracy_clutter_picking.grasping import (
    SimulatedServos,
    TopDownGraspGeometry,
    centre_in_root,
)


@dataclass
class PickUpActionMujoco(ActionDescription):
    """
    Pick a carton up by closing the pads onto it, with nothing attached to the gripper,
    so whether it comes along is decided by the grasp.
    """

    object_designator: Body = field(kw_only=True)
    """
    The body to pick up.
    """

    arm: Arm = field(kw_only=True)
    """
    The arm that picks it up.
    """

    hover_clearance: float = 0.3
    """
    Height, in metres, above a body's own top face the tool centre point moves to before
    descending onto it -- clears obstacles during the horizontal part of the approach.

    ``0.3`` leaves a wide margin over the tallest obstacle a scene has stood on the
    table; a smaller hover height left Giskard's own collision-avoidance solver too
    little vertical room to route the arm over it at all.
    """

    grasp_yaw: float = 0.0
    """
    Rotation of the gripper about the vertical approach axis, in radians; see :attr:`~ex
    periments.causal_reasoning.tracy_clutter_picking.grasping.TopDownGraspGeometry.grasp
    _yaw`.
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

    squeeze_margin: float = 0.001
    """
    How far, in metres, past the object's own half-width the pads are commanded to
    close, so they press into it firmly rather than merely touching.

    Kept small, since it is commanded penetration into a rigid object.
    """

    grasp_half_width: Optional[float] = None
    """
    Half the width the pads are to meet the object across, in metres, for an object
    grasped across a known pair of faces.

    Defaults to half the object's own bounding-box width along the closing axis, which
    over-reads the width of a box standing a little turned relative to the gripper.
    """

    grasp_settle_time: timedelta = field(default=timedelta(milliseconds=500))
    """
    Simulated time the pads are held at their closing target before the arm moves, so
    the servos build up their holding force against the object.
    """

    @property
    def _action_plan(self) -> PlanNode:
        geometry = TopDownGraspGeometry(self.world, self.arm, self.grasp_yaw)
        body_center = centre_in_root(self.world, self.object_designator)
        hover = geometry.tool_frame_pose(
            body_center[0], body_center[1], body_center[2] + self.hover_clearance
        )
        grasp = geometry.tool_frame_pose(
            body_center[0],
            body_center[1],
            body_center[2] + self.grasp_close_swing_clearance,
        )
        return sequential(
            [
                self._reach(hover),
                self._reach(grasp),
                MoveFingersToWidthMotion(
                    gripper=self.arm.end_effector,
                    width=self._grasp_width(),
                    squeeze=self.squeeze_margin,
                ),
                code(self._hold_the_grip),
                self._reach(hover),
            ]
        )

    def _reach(self, goal_pose: Pose) -> PlanNode:
        """
        Move the arm's tool frame to a Cartesian pose and wait for its servos.

        Collision avoidance stays off, as a physically simulated robot runs it: in a
        crowded scene it otherwise treats the very object being approached as an
        obstacle and leaves the goal unreachable. Orientation is still constrained,
        since every pose here shares the same fixed top-down orientation and leaving it
        free lets Giskard's own redundancy resolution drift it on each leg.

        :param goal_pose: Target pose for the arm's tool frame.
        :return: The reach, followed by the wait for the servos.
        """
        return sequential(
            [
                MoveToolCenterPointMotion(goal_pose, self.arm),
                code(self._settle),
            ]
        )

    def _settle(self) -> None:
        """
        Advance the physics until the arm's servos have caught up with the command.
        """
        SimulatedServos.of_the_running_simulation().settle(self.arm)

    def _grasp_width(self) -> float:
        """
        :return: How wide the object stands where the fingers meet it, in metres.
        """
        if self.grasp_half_width is not None:
            return 2 * self.grasp_half_width
        bounding_box = (
            self.object_designator.collision.as_bounding_box_collection_in_frame(
                self.arm.end_effector.root
            ).bounding_box()
        )
        return bounding_box.max_x - bounding_box.min_x

    def _hold_the_grip(self) -> None:
        """
        Hold the pads where they were sent, since the fingers stop on the object short
        of their set point and the servos need the time to press rather than to arrive.
        """
        SimulatedServos.of_the_running_simulation().hold(self.grasp_settle_time)
