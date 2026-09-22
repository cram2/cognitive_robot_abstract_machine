"""
Runs Giskard's own control loop against the live, physically simulated world.

Every control cycle Giskard writes its command into the world state, the MuJoCo
synchronizer hands that to the joints' servos as their set point, and the executor's
pacer steps the physics before the next cycle. A motion is therefore reached in the
physics, as fast and as hard as the servos allow, rather than planned kinematically
first and played back afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from giskardpy.executor import Executor, SteppedSimulationPacer
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
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.robotiq_85_gripper import Robotiq85Gripper
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


def arm_of(robot: Tracy, arm_side: Arms) -> Arm:
    """
    :param robot: The robot to look the arm up on.
    :param arm_side: Which arm to return; must be :attr:`Arms.LEFT` or
        :attr:`Arms.RIGHT`.
    :return: ``robot``'s own left or right arm.
    """
    return robot.left_arm if arm_side == Arms.LEFT else robot.right_arm


# %% running motions live


@dataclass
class MotionRunner:
    """
    Runs one Giskard motion after another against the live, physically simulated world,
    ticking Giskard's own control loop in lockstep with the physics.
    """

    simulation: MujocoSim
    """
    The simulation of the world the motions are run in; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`
    already.
    """

    target_frequency: int = 50
    """
    Giskard's control rate, in cycles per simulated second; the physics advances one
    control period between cycles.
    """

    prediction_horizon: int = 4
    """
    Giskard's QP prediction horizon, matching the one
    :class:`~coraplex.plans.executables.GiskardExecutable` builds its own controller
    with.
    """

    max_ticks: int = 2000
    """
    Control cycles a single motion gets before giving up, matching
    :attr:`~coraplex.datastructures.dataclasses.Context.ticks_per_motion`'s own default.
    """

    convergence_threshold: float = 0.01
    """
    Largest per-joint distance, in radians, between a joint's simulated position and its
    set point for the motion to count as settled.
    """

    settle_timeout: timedelta = timedelta(seconds=10)
    """
    Simulated time to wait for the servos to settle once Giskard's own motion has ended.
    """

    squeeze_margin: float = 0.001
    """
    How far, in metres, past a grasped object's own half-width the fingers are commanded
    to close, so they press into it firmly rather than merely touching.

    Kept small, since it is commanded penetration into a rigid object.
    """

    closing_threshold: float = 0.0005
    """
    How close, in metres, the distance between the fingertip frames has to come to its
    goal for a closing to count as done: well under the squeeze margin, so the squeeze
    is not lost to the tolerance.
    """

    grasp_settle_time: timedelta = timedelta(milliseconds=500)
    """
    Simulated time the fingers are held at their closing target before the caller moves
    the arm, so the servos build up their holding force against the object.
    """

    @property
    def world(self) -> World:
        """
        The live world the motions are run in.
        """
        return self.simulation.world

    @property
    def tick_period(self) -> timedelta:
        """
        Simulated time one control cycle stands for.
        """
        return timedelta(seconds=1 / self.target_frequency)

    def run(self, task: Task, avoid_collisions: bool) -> None:
        """
        Run ``task`` against the live world until Giskard's own motion ends, with the
        executor's pacer stepping the physics one control period per tick.

        :param task: The Giskard task to run.
        :param avoid_collisions: Whether Giskard's own ``ExternalCollisionAvoidance``
            runs alongside. Set False for a goal whose own point is to approach and
            touch an object, since avoidance would otherwise treat that object as an
            obstacle and make the goal unreachable.
        :raises MotionDidNotFinish: If the goal was not reached within
            :attr:`max_ticks`.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(task)
        if avoid_collisions:
            motion_statechart.add_node(ExternalCollisionAvoidance())
        end_motion = EndMotion()
        end_motion.start_condition = task.observation_variable
        motion_statechart.add_node(end_motion)

        executor = Executor(
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=self.target_frequency,
                    prediction_horizon=self.prediction_horizon,
                    verbose=False,
                ),
            ),
            pacer=SteppedSimulationPacer(self.simulation),
        )
        executor.compile(motion_statechart)
        try:
            executor.tick_until_end(timeout=self.max_ticks)
        except TimeoutError as error:
            raise MotionDidNotFinish(failed_motions=[task]) from error

    def settle(
        self, joint_names: List[str], timeout: Optional[timedelta] = None
    ) -> None:
        """
        Advance the physics until every one of ``joint_names`` has reached its set
        point, or the timeout passes.

        :param joint_names: The joints to wait for.
        :param timeout: Simulated time to wait; :attr:`settle_timeout` if not given.
        """
        if timeout is None:
            timeout = self.settle_timeout
        simulator = self.simulation.simulator
        set_points = {
            joint_name: self.world.state[
                self.world.get_connection_by_name(joint_name).raw_dof.id
            ].position
            for joint_name in joint_names
        }
        simulated_time = timedelta()
        errors: Dict[str, float] = {}
        while simulated_time < timeout:
            self.simulation.step_simulation(self.tick_period)
            simulated_time += self.tick_period
            errors = {
                joint_name: abs(
                    simulator.get_joint_value(joint_name).result - set_point
                )
                for joint_name, set_point in set_points.items()
            }
            if max(errors.values()) < self.convergence_threshold:
                return
        worst_joint = max(errors, key=errors.get)
        logger.warning(
            "Motion did not settle within %s; worst joint %s is %.3f rad off.",
            timeout,
            worst_joint,
            errors[worst_joint],
        )

    def hold(self, duration: timedelta) -> None:
        """
        Advance the physics with every set point held where it is.

        :param duration: Simulated time to hold.
        """
        self.simulation.step_simulation(duration)

    def reach(
        self,
        robot: Tracy,
        arm_side: Arms,
        goal_pose: Pose,
        translation_only: bool,
        avoid_collisions: bool = True,
    ) -> None:
        """
        Move an arm's tool frame to a Cartesian goal and wait for the arm to settle.

        :param robot: The robot whose arm moves.
        :param arm_side: Which arm's tool frame should reach ``goal_pose``.
        :param goal_pose: Target pose for the arm's tool frame.
        :param translation_only: If True, only the tool frame's position is constrained;
            otherwise both position and orientation are.
        :param avoid_collisions: See :meth:`run`.
        """
        arm = arm_of(robot, arm_side)
        if translation_only:
            task = CartesianPosition(
                root_link=robot.root,
                tip_link=arm.end_effector.tool_frame,
                goal_point=goal_pose.to_position(),
            )
        else:
            task = CartesianPose(
                root_link=robot.root,
                tip_link=arm.end_effector.tool_frame,
                goal_pose=goal_pose,
            )
        self.run(task, avoid_collisions)
        self.settle(
            [
                connection.raw_dof.name.name
                for connection in arm.active_connections
                if isinstance(connection, ActiveConnection1DOF)
            ]
        )

    def move_joints(
        self,
        targets: Dict[str, float],
        avoid_collisions: bool = True,
        settle_timeout: Optional[timedelta] = None,
    ) -> None:
        """
        Move joints to target positions with Giskard's own
        :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointPositionList`, which
        keeps every joint within its own velocity limit, and wait for them to settle.

        :param targets: Target position by joint name.
        :param avoid_collisions: See :meth:`run`.
        :param settle_timeout: See :meth:`settle`.
        """
        joint_connections = [
            self.world.get_connection_by_name(name) for name in targets
        ]
        goal_state = JointState.from_mapping(
            dict(zip(joint_connections, targets.values()))
        )
        self.run(JointPositionList(goal_state=goal_state), avoid_collisions)
        self.settle(list(targets), settle_timeout)

    def park_arms(
        self, robot: Tracy, arm_sides: List[Arms], avoid_collisions: bool = True
    ) -> None:
        """
        Move ``arm_sides`` to their park configuration.

        :param robot: The robot whose arms are parked.
        :param arm_sides: Which arms to park, e.g. ``[Arms.LEFT, Arms.RIGHT]``.
        :param avoid_collisions: See :meth:`run`.
        """
        targets: Dict[str, float] = {}
        for arm_side in arm_sides:
            park_state = arm_of(robot, arm_side).get_joint_state_by_type(
                StaticJointState.PARK
            )
            for connection, target in zip(
                park_state.connections, park_state.target_values
            ):
                targets[connection.raw_dof.name.name] = target
        self.move_joints(targets, avoid_collisions=avoid_collisions)

    def set_gripper(
        self,
        robot: Tracy,
        arm_side: Arms,
        state: GripperState,
        settle_timeout: timedelta = timedelta(seconds=3),
    ) -> None:
        """
        Open or close an arm's gripper.

        Collision avoidance is always off: the goal is to close the fingers around
        whatever object is between them, which avoidance would treat as an obstacle.

        :param robot: The robot whose gripper is driven.
        :param arm_side: Which arm's gripper to drive.
        :param state: The gripper state to command, e.g. :attr:`GripperState.CLOSE`.
        :param settle_timeout: Simulated time to wait for the fingers to settle.
        """
        goal_state = arm_of(robot, arm_side).end_effector.get_joint_state_by_type(state)
        # every connection of the mimic linkage shares one raw degree of freedom, so
        # each connection's own target is converted back to that degree of freedom's
        raw_targets: Dict[str, float] = {}
        for connection, target in zip(goal_state.connections, goal_state.target_values):
            raw_targets[connection.raw_dof.name.name] = (
                target - connection.offset
            ) / connection.multiplier
        self.move_joints(
            raw_targets, avoid_collisions=False, settle_timeout=settle_timeout
        )

    @staticmethod
    def _pad_depth(gripper: Robotiq85Gripper) -> float:
        """
        How far a fingertip pad's inner face sits inside its own frame along the
        closing axis: the distance between the fingertip frames is the pad-to-pad
        width plus twice this.

        :param gripper: The gripper whose pads are measured, at any opening.
        :return: The depth, in metres.
        """
        world = gripper._world
        frame_x = world.compute_forward_kinematics_np(gripper.root, gripper.thumb.tip)[
            0, 3
        ]
        inner_face_x = (
            gripper.thumb.tip.collision.as_bounding_box_collection_in_frame(
                gripper.root
            )
            .bounding_box()
            .min_x
        )
        return frame_x - inner_face_x

    def close_gripper_around(
        self,
        robot: Tracy,
        arm_side: Arms,
        target_body: Body,
        squeeze_margin: Optional[float] = None,
        half_width: Optional[float] = None,
    ) -> None:
        """
        Close an arm's gripper around ``target_body``, sized to the object's own width
        instead of driving to the fully closed position.

        Closing all the way on an object that is not perfectly centred between the
        fingers wedges it sideways rather than gripping it, so the fingers close to the
        object's half-width along the closing axis minus the squeeze margin, and are then
        held there so the squeeze is actually applied: a face contact holds on friction
        alone, but a point or edge contact slips straight back out without it.

        The closing is a Cartesian goal on the distance between the two fingertip
        frames; Giskard finds the knuckle angle that realises it.

        :param robot: The robot whose gripper is driven.
        :param arm_side: Which arm's gripper to drive.
        :param target_body: The body to close around.
        :param squeeze_margin: How far past the half-width the fingers close;
            :attr:`squeeze_margin` if not given.
        :param half_width: Half the width, in metres, the pads are to meet the object
            across, for an object grasped across a known pair of faces. Defaults to
            half the object's own bounding-box width along the closing axis, which
            over-reads the width of a box standing a little turned relative to the
            gripper.
        """
        if squeeze_margin is None:
            squeeze_margin = self.squeeze_margin
        gripper = arm_of(robot, arm_side).end_effector
        if half_width is None:
            bounding_box = target_body.collision.as_bounding_box_collection_in_frame(
                gripper.root
            ).bounding_box()
            half_width = (bounding_box.max_x - bounding_box.min_x) / 2
        target_half_width = max(0.0, half_width - squeeze_margin)
        frame_distance = 2 * (target_half_width + self._pad_depth(gripper))

        simulator = self.simulation.simulator
        target_name = target_body.name.name
        thumb_tip_name = gripper.thumb.tip.name.name
        finger_tip_name = gripper.finger.tip.name.name

        def both_fingertips_touching() -> bool:
            """
            Whether both pads are in contact with the target right now.
            """
            thumb_contacts = simulator.get_contact_bodies(
                body_name=thumb_tip_name, including_children=False
            ).result
            finger_contacts = simulator.get_contact_bodies(
                body_name=finger_tip_name, including_children=False
            ).result
            return target_name in thumb_contacts and target_name in finger_contacts

        # the pads stay parallel, so one fingertip frame moves purely along the
        # other's closing axis and the distance between them is a 1-D goal
        self.run(
            CartesianPosition(
                root_link=gripper.finger.tip,
                tip_link=gripper.thumb.tip,
                goal_point=Point3(
                    frame_distance, 0.0, 0.0, reference_frame=gripper.finger.tip
                ),
                threshold=self.closing_threshold,
            ),
            avoid_collisions=False,
        )
        # the fingers stop on the object short of their set point, so the servos are
        # given a fixed squeeze time rather than waited for to settle
        self.hold(self.grasp_settle_time)
        logger.info(
            "%s: gripper closed to its own half-width-sized target (%.4fm); "
            "both fingertips %s the object.",
            target_body.name,
            target_half_width,
            "reached" if both_fingertips_touching() else "never reached",
        )
