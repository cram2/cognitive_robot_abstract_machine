"""
Plan Cartesian- or joint-space goals against an isolated, physics-free scratch copy of
the world via Giskard, then play the resulting trajectory back onto the real, physically
simulated MuJoCo world by driving each joint's own position-servo actuator directly.

Giskard never touches the live, physically simulated world under this module: it only
ever ticks against a :func:`copy.deepcopy` of it, so it cannot race
:class:`~semantic_digital_twin.adapters.multi_sim.MujocoSynchronizer`'s own
physics-thread state sync (see :mod:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.equipment`'s own
module docstring). Execution then drives real MuJoCo actuators the same way, so tracking
is genuinely closed-loop against measured physics, not open-loop trajectory replay.

Every motion here, including park and gripper open/close, is planned this way rather
than commanded straight to the target: commanding a target directly, with no velocity
profile at all, lets a joint move far faster than its real hardware ever could.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from typing_extensions import Dict, List, Optional

from coraplex.datastructures.enums import Arms
from coraplex.exceptions import MotionDidNotFinish
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.equipment import (
    joint_state_of_type,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation import (
    RealTimeSimulation,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Actuator, Body

logger = logging.getLogger(__name__)

Trajectory = List[Dict[str, float]]
"""
A planned motion: one mapping from joint name to position per planning tick, in order.
"""


def arm_of(robot: Tracy, arm_side: Arms) -> Arm:
    """
    :param robot: The robot to look the arm up on.
    :param arm_side: Which arm to return; must be :attr:`Arms.LEFT` or
        :attr:`Arms.RIGHT`.
    :return: ``robot``'s own left or right arm.
    """
    return robot.left_arm if arm_side == Arms.LEFT else robot.right_arm


# %% the gripper's own geometry


@dataclass(frozen=True)
class RobotiqGripper:
    """
    One of Tracy's Robotiq 2F-85 grippers, by the arm it hangs off, and the bodies and
    joints of its own description that a grasp has to know.
    """

    arm_side: Arms
    """
    Which arm the gripper hangs off.
    """

    @property
    def prefix(self) -> str:
        """
        What the description prefixes the gripper's own bodies and joints with.
        """
        return "right_" if self.arm_side == Arms.RIGHT else "left_"

    @property
    def left_fingertip_name(self) -> str:
        """
        Name of the left fingertip pad's body.
        """
        return f"{self.prefix}robotiq_85_left_finger_tip_link"

    @property
    def right_fingertip_name(self) -> str:
        """
        Name of the right fingertip pad's body.
        """
        return f"{self.prefix}robotiq_85_right_finger_tip_link"

    @property
    def knuckle_joint_name(self) -> str:
        """
        Name of the joint that actually drives the gripper; every other finger joint in
        the mimic linkage follows it.
        """
        return f"{self.prefix}robotiq_85_left_knuckle_joint"

    def knuckle_raw_dof(self, robot: Tracy) -> DegreeOfFreedom:
        """
        :param robot: The robot the gripper belongs to.
        :return: The raw degree of freedom driving the knuckle.
        """
        return next(
            connection.raw_dof
            for connection in arm_of(
                robot, self.arm_side
            ).end_effector.active_connections
            if connection.raw_dof.name.name == self.knuckle_joint_name
        )

    def closing_raw_angle_for_half_width(
        self,
        world: World,
        target_half_width: float,
        iterations: int = 30,
    ) -> float:
        """
        The knuckle's raw angle at which the fingertip pads' own inner faces first reach
        ``target_half_width`` out from the gripper's own centreline.

        The pad's inner position decreases monotonically as the knuckle closes
        (confirmed directly, sampled across its own full range), so bisection against an
        isolated scratch copy of ``world`` converges reliably.

        :param world: The live world to clone for the search; never itself modified.
        :param target_half_width: The half-width, in metres, to close to.
        :param iterations: Bisection steps; 30 narrows the joint's own ~0.8 rad range to
            well under a micro-radian.
        :return: The raw angle.
        """
        scratch_world = deepcopy(world)
        [scratch_robot] = scratch_world.get_semantic_annotations_by_type(Tracy)
        gripper_root = arm_of(scratch_robot, self.arm_side).end_effector.root
        left_fingertip = scratch_world.get_body_by_name(self.left_fingertip_name)
        raw_dof = self.knuckle_raw_dof(scratch_robot)

        def inner_x(raw_angle: float) -> float:
            """
            The left pad's innermost point along the closing axis at one raw angle,
            moving the scratch world's own state directly.
            """
            scratch_world.state[raw_dof.id].position = raw_angle
            scratch_world.notify_state_change()
            scratch_world.update_forward_kinematics()
            return (
                left_fingertip.collision.as_bounding_box_collection_in_frame(
                    gripper_root
                )
                .bounding_box()
                .min_x
            )

        lower, upper = raw_dof.limits.lower.position, raw_dof.limits.upper.position
        if target_half_width >= inner_x(lower):
            return lower
        if target_half_width <= inner_x(upper):
            return upper
        for _ in range(iterations):
            midpoint = (lower + upper) / 2
            if inner_x(midpoint) > target_half_width:
                lower = midpoint
            else:
                upper = midpoint
        return upper


# %% planning and playing back


@dataclass
class TrajectoryPlanner:
    """
    Plans every motion kinematically against a scratch copy of the world and plays it
    back on the real, physically simulated one.
    """

    target_frequency: int = 50
    """
    Giskard tick rate used both to plan (scratch world) and to pace trajectory playback
    (real world), so one recorded waypoint corresponds to one real physics-advance step.
    """

    prediction_horizon: int = 4
    """
    Giskard QP prediction horizon used to plan, matching the one
    :class:`~coraplex.plans.executables.GiskardExecutable` builds its own controller
    with, so a trajectory planned here converges the way a live Giskard motion would.
    """

    max_ticks: int = 2000
    """
    Tick budget a single plan gets before giving up, matching
    :attr:`~coraplex.datastructures.dataclasses.Context.ticks_per_motion`'s own
    kinematic-motion default -- planning always runs kinematically, regardless of
    whether the executing arm is physically simulated.
    """

    convergence_threshold: float = 0.01
    """
    Maximum per-joint error, in radians, to count a trajectory's final waypoint as
    reached.
    """

    settle_timeout: float = 10.0
    """
    Simulated seconds to wait for convergence after a trajectory's last waypoint, on top
    of the trajectory's own duration.
    """

    squeeze_margin: float = 0.001
    """
    How far, in metres, past a grasped object's own half-width the fingers are commanded
    to close, so they press into it firmly rather than merely touching.

    Mirrors the proven-working Franka Montessori demo's own
    ``MoveGripperMotion.squeeze_margin``: kept small, since it is *commanded*
    penetration into a rigid object, and the whole point of sizing the close to the
    object is to keep that penetration bounded and deliberate.
    """

    grasp_settle_time: float = 0.5
    """
    Simulated seconds the fingers are held at their closing target, after the planned
    close finishes, before the caller moves the arm.

    The position servos need this long to build up their holding force against the
    object; moving the arm the instant the last close waypoint is issued lets a barely-
    seated grip peel off during the first reach.
    """

    @property
    def tick_period(self) -> float:
        """
        Simulated seconds one planning tick stands for.
        """
        return 1.0 / self.target_frequency

    def _plan(
        self,
        scratch_world: World,
        task: Task,
        joint_connections: List[ActiveConnection1DOF],
        avoid_collisions: bool,
    ) -> Trajectory:
        """
        Tick ``task`` (already built against ``scratch_world``) until it converges,
        recording ``joint_connections``' own positions after every tick.

        :param scratch_world: The isolated world ``task`` was built against.
        :param task: The Giskard task to converge.
        :param joint_connections: Which connections' positions to record each tick.
        :param avoid_collisions: Whether Giskard's own ``ExternalCollisionAvoidance`` is
            enabled for this plan. Set False for a goal whose own point is to approach
            (and end up touching) an object -- e.g. descending onto something to grasp
            or place it, or closing a gripper around it -- since collision avoidance
            would otherwise treat that same object as an obstacle to stay away from and
            make the goal unreachable.
        :return: The trajectory.
        :raises MotionDidNotFinish: If the goal was not reached within
            :attr:`max_ticks`.
        """
        motion_state_chart = MotionStatechart()
        motion_state_chart.add_node(task)
        if avoid_collisions:
            motion_state_chart.add_node(ExternalCollisionAvoidance())
        end_motion = EndMotion()
        end_motion.start_condition = task.observation_variable
        motion_state_chart.add_node(end_motion)

        executor = Executor(
            context=MotionStatechartContext(
                world=scratch_world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=self.target_frequency,
                    prediction_horizon=self.prediction_horizon,
                    verbose=False,
                ),
            )
        )
        executor.compile(motion_state_chart)

        trajectory: Trajectory = []
        try:
            for _ in range(self.max_ticks):
                executor.tick()
                trajectory.append(
                    {
                        connection.raw_dof.name.name: scratch_world.state[
                            connection.raw_dof.id
                        ].position
                        for connection in joint_connections
                    }
                )
                if executor.motion_statechart.is_end_motion():
                    return trajectory
        finally:
            executor.set_velocity_acceleration_jerk_to_zero()
            executor.motion_statechart.cleanup_nodes(context=executor.context)
            executor.context.cleanup()

        raise MotionDidNotFinish(failed_motions=[task])

    def plan_cartesian_trajectory(
        self,
        world: World,
        arm_side: Arms,
        goal_pose: Pose,
        translation_only: bool,
        avoid_collisions: bool = True,
    ) -> Trajectory:
        """
        Solve a Cartesian goal against an isolated clone of ``world``, returning the
        resulting joint-space trajectory for the given arm's own joints.

        :param world: The live world to clone; never itself modified.
        :param arm_side: Which arm's tool centre point should reach ``goal_pose``.
        :param goal_pose: Target pose for the arm's tool centre point.
        :param translation_only: If True, only the tip's position is constrained;
            otherwise both position and orientation are.
        :param avoid_collisions: See :meth:`_plan`.
        :return: The trajectory.
        :raises MotionDidNotFinish: If the goal was not reached within
            :attr:`max_ticks`.
        """
        scratch_world = deepcopy(world)
        [scratch_robot] = scratch_world.get_semantic_annotations_by_type(Tracy)
        scratch_arm = arm_of(scratch_robot, arm_side)
        joint_connections = [
            connection
            for connection in scratch_arm.active_connections
            if isinstance(connection, ActiveConnection1DOF)
        ]

        if translation_only:
            task = CartesianPosition(
                root_link=scratch_robot.root,
                tip_link=scratch_arm.end_effector.tool_frame,
                goal_point=goal_pose.to_position(),
            )
        else:
            task = CartesianPose(
                root_link=scratch_robot.root,
                tip_link=scratch_arm.end_effector.tool_frame,
                goal_pose=goal_pose,
            )

        return self._plan(scratch_world, task, joint_connections, avoid_collisions)

    def plan_joint_trajectory(
        self, world: World, targets: Dict[str, float], avoid_collisions: bool = True
    ) -> Trajectory:
        """
        Solve a joint-space goal against an isolated clone of ``world``.

        Uses Giskard's own
        :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointPositionList` task,
        which clamps its own reference velocity to each joint's own real
        ``dof.limits.upper.velocity``, so every joint stays within the speed the real
        robot could actually achieve, unlike commanding a target straight to a
        position-servo actuator with no velocity profile at all.

        :param world: The live world to clone; never itself modified.
        :param targets: Target position by joint name.
        :param avoid_collisions: See :meth:`_plan`.
        :return: The trajectory.
        :raises MotionDidNotFinish: If the goal was not reached within
            :attr:`max_ticks`.
        """
        scratch_world = deepcopy(world)
        joint_connections = [
            scratch_world.get_connection_by_name(name) for name in targets
        ]
        goal_state = JointState.from_mapping(
            dict(zip(joint_connections, targets.values()))
        )
        task = JointPositionList(goal_state=goal_state)
        return self._plan(scratch_world, task, joint_connections, avoid_collisions)

    def follow_joint_trajectory(
        self,
        simulation: RealTimeSimulation,
        actuators: Dict[str, Actuator],
        trajectory: Trajectory,
        settle_timeout: Optional[float] = None,
    ) -> None:
        """
        Drive ``actuators`` through ``trajectory`` on the real, physically simulated
        world, one recorded waypoint per
        :meth:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation.RealTimeSimulation.advance`
        step, then hold the final waypoint until every joint settles.

        :param simulation: The running real-time simulation to drive.
        :param actuators: Every joint's own actuator, keyed by joint name.
        :param trajectory: The waypoints to follow, in order.
        :param settle_timeout: Simulated seconds to wait for convergence after the last
            waypoint; :attr:`settle_timeout` if not given.
        """
        if settle_timeout is None:
            settle_timeout = self.settle_timeout
        for waypoint in trajectory:
            for joint_name, target in waypoint.items():
                simulation.command(actuators[joint_name], target)
            simulation.advance(self.tick_period)

        targets = trajectory[-1]
        simulated_time = 0.0
        errors: Dict[str, float] = {}
        while simulated_time < settle_timeout:
            simulation.advance(self.tick_period)
            simulated_time += self.tick_period
            errors = {
                joint_name: abs(
                    simulation.mirror.simulator.get_joint_value(joint_name).result
                    - target
                )
                for joint_name, target in targets.items()
            }
            if max(errors.values()) < self.convergence_threshold:
                return
        worst_joint = max(errors, key=errors.get)
        logger.warning(
            "Trajectory did not settle within %.0fs; worst joint %s is %.3f rad off.",
            settle_timeout,
            worst_joint,
            errors[worst_joint],
        )

    def park_arms(
        self,
        simulation: RealTimeSimulation,
        actuators: Dict[str, Actuator],
        robot: Tracy,
        arm_sides: List[Arms],
        avoid_collisions: bool = True,
    ) -> None:
        """
        Plan a velocity-limited joint trajectory to park ``arm_sides`` and play it back
        on the real, physically simulated world.

        :param simulation: The running real-time simulation to drive.
        :param actuators: Every joint's own actuator, keyed by joint name.
        :param robot: The robot whose arms are parked.
        :param arm_sides: Which arms to park, e.g. ``[Arms.LEFT, Arms.RIGHT]``.
        :param avoid_collisions: See :meth:`_plan`.
        """
        targets: Dict[str, float] = {}
        for arm_side in arm_sides:
            park_state = joint_state_of_type(
                arm_of(robot, arm_side), StaticJointState.PARK
            )
            for connection, target in zip(
                park_state.connections, park_state.target_values
            ):
                targets[connection.raw_dof.name.name] = target

        trajectory = self.plan_joint_trajectory(
            robot._world, targets, avoid_collisions=avoid_collisions
        )
        self.follow_joint_trajectory(simulation, actuators, trajectory)

    def set_gripper(
        self,
        simulation: RealTimeSimulation,
        actuators: Dict[str, Actuator],
        robot: Tracy,
        arm_side: Arms,
        state: GripperState,
        settle_timeout: float = 3.0,
    ) -> None:
        """
        Plan a velocity-limited joint trajectory to close or open an arm's gripper and
        play it back on the real, physically simulated world.

        Collision avoidance is always off for this plan (unlike :meth:`park_arms`'s own
        default): the goal is to close the fingers around whatever object is between
        them, which collision avoidance would otherwise treat as an obstacle.

        :param simulation: The running real-time simulation to drive.
        :param actuators: Every joint's own actuator, keyed by joint name.
        :param robot: The robot whose gripper is driven.
        :param arm_side: Which arm's gripper to drive.
        :param state: The gripper state to command, e.g. :attr:`GripperState.CLOSE`.
        :param settle_timeout: Simulated seconds to wait for convergence after the
            trajectory's own last waypoint.
        """
        goal_state = joint_state_of_type(arm_of(robot, arm_side).end_effector, state)
        # The gripper's own connections are a mimic linkage: every one of them shares
        # the same raw_dof, so a naive target per connection would disagree with itself
        # -- a mimic connection's own target is expressed in its own, not the raw_dof's,
        # sign and offset. Convert each connection's own target back to its raw_dof's
        # own value first, so every connection agrees on one target per degree of freedom.
        raw_targets: Dict[str, float] = {}
        for connection, target in zip(goal_state.connections, goal_state.target_values):
            raw_targets[connection.raw_dof.name.name] = (
                target - connection.offset
            ) / connection.multiplier

        trajectory = self.plan_joint_trajectory(
            robot._world, raw_targets, avoid_collisions=False
        )
        self.follow_joint_trajectory(
            simulation, actuators, trajectory, settle_timeout=settle_timeout
        )

    def close_gripper_around(
        self,
        simulation: RealTimeSimulation,
        actuators: Dict[str, Actuator],
        robot: Tracy,
        arm_side: Arms,
        target_body: Body,
        squeeze_margin: Optional[float] = None,
        half_width: Optional[float] = None,
    ) -> None:
        """
        Close an arm's gripper around ``target_body``, sized to the object's own width
        instead of always driving to the gripper's fully closed position.

        Closing all the way to zero opening on an object that is not perfectly centred
        between the fingers wedges it sideways rather than gripping it -- confirmed
        directly on Tracy: the fully-closed target let the fingers close *past* a
        shape's own width, shoving it out from between them before both sides ever
        made contact. So the target's own half-width along the gripper's closing axis
        is measured in the gripper's own root frame, and the fingers close to that
        instead, minus the squeeze margin so they press in rather than merely touch.
        The whole planned close is played out even once both fingertip pads register
        MuJoCo contact, so the squeeze is actually applied -- a face contact holds on
        friction alone, but a point or edge contact slips straight back out without it.

        :param simulation: The running real-time simulation to drive.
        :param actuators: Every joint's own actuator, keyed by joint name.
        :param robot: The robot whose gripper is driven.
        :param arm_side: Which arm's gripper to drive.
        :param target_body: The body to close around.
        :param squeeze_margin: How far past the half-width the fingers close;
            :attr:`squeeze_margin` if not given.
        :param half_width: Half the width, in metres, the pads are to meet the object
            across, for an object grasped across a known pair of faces. Defaults to
            half the object's own bounding-box width along the closing axis, which
            over-reads the width of a box standing a little turned relative to the
            gripper, since the box's corners then reach further out than its faces do.
        """
        if squeeze_margin is None:
            squeeze_margin = self.squeeze_margin
        world = robot._world
        gripper = RobotiqGripper(arm_side)
        if half_width is None:
            bounding_box = target_body.collision.as_bounding_box_collection_in_frame(
                arm_of(robot, arm_side).end_effector.root
            ).bounding_box()
            half_width = (bounding_box.max_x - bounding_box.min_x) / 2
        target_inner_x = max(0.0, half_width - squeeze_margin)
        raw_angle = gripper.closing_raw_angle_for_half_width(world, target_inner_x)

        raw_dof = gripper.knuckle_raw_dof(robot)
        trajectory = self.plan_joint_trajectory(
            world, {raw_dof.name.name: raw_angle}, avoid_collisions=False
        )

        simulator = simulation.mirror.simulator
        target_name = target_body.name.name

        def both_fingertips_touching() -> bool:
            """
            Whether both pads are in contact with the target right now.
            """
            left_contacts = simulator.get_contact_bodies(
                body_name=gripper.left_fingertip_name, including_children=False
            ).result
            right_contacts = simulator.get_contact_bodies(
                body_name=gripper.right_fingertip_name, including_children=False
            ).result
            return target_name in left_contacts and target_name in right_contacts

        made_contact = False
        for waypoint in trajectory:
            for joint_name, target in waypoint.items():
                simulation.command(actuators[joint_name], target)
            simulation.advance(self.tick_period)
            made_contact = made_contact or both_fingertips_touching()

        for _ in range(round(self.grasp_settle_time / self.tick_period)):
            for joint_name, target in trajectory[-1].items():
                simulation.command(actuators[joint_name], target)
            simulation.advance(self.tick_period)
            made_contact = made_contact or both_fingertips_touching()

        logger.info(
            "%s: gripper closed to its own half-width-sized target (%.4fm); "
            "both fingertips %s the object.",
            target_body.name,
            target_inner_x,
            "reached" if made_contact else "never reached",
        )
