"""
What a top-down grasp in this experiment needs beyond the plans coraplex already runs.

Coraplex drives a physically simulated robot itself, through
:class:`~coraplex.execution_environment.PhysicallySimulatedRobot`, and states its
motions as ordinary motion designators. What is left here is the part that is about
this gripper meeting this carton: where to aim so that the pads rather than the tool
frame land on the target, closing the pads to the carton's own width, and waiting for
the servos to catch up with a command once the controller is happy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import timedelta

import numpy
from coraplex.plans.executables import GiskardExecutable
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, List, Optional

from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    NoRunningSimulationError,
)

logger = logging.getLogger(__name__)


# %% where the fingers actually are


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
        :param x: Where the fingers should meet along the world root's x-axis.
        :param y: Where the fingers should meet along the world root's y-axis.
        :param z: Where the fingers should meet along the world root's z-axis.
        :return: The tool frame pose that puts the fingers there, top-down.
        """
        return self.arm.end_effector.tool_frame_goal_for_grasp(
            self.grasp_frame(x, y, z)
        )


# %% waiting for the simulated servos


@dataclass
class SimulatedServos:
    """
    The servos of the robot in the running simulation, which reach a command some time
    after the controller is satisfied with it.
    """

    simulation: MujocoSim
    """
    The running simulation whose physics is advanced while waiting.
    """

    control_period: timedelta = field(default=timedelta(seconds=1 / 100))
    """
    Simulated time the physics advances per step while waiting, matching the rate
    coraplex drives a physically simulated robot at.
    """

    convergence_threshold: float = 0.01
    """
    Largest per-joint distance, in radians, between a joint's simulated position and its
    set point for a motion to count as settled.
    """

    settle_timeout: timedelta = field(default=timedelta(seconds=10))
    """
    Simulated time to wait for the servos to settle once a motion has ended.
    """

    @classmethod
    def of_the_running_simulation(cls) -> SimulatedServos:
        """
        :return: The servos of the simulation the current plan is being performed
            against.
        :raises NoRunningSimulationError: If no plan is being performed against a
            physically simulated robot.
        """
        simulation = GiskardExecutable.simulation
        if simulation is None:
            raise NoRunningSimulationError()
        return cls(simulation=simulation)

    def hold(self, duration: timedelta) -> None:
        """
        Advance the physics with every set point held where it is.

        :param duration: Simulated time to hold.
        """
        self.simulation.step_simulation(duration)

    def settle(self, arm: Arm, timeout: Optional[timedelta] = None) -> None:
        """
        Advance the physics until every one of an arm's joints has reached its set
        point, or the timeout passes.

        :param arm: The arm whose joints to wait for.
        :param timeout: Simulated time to wait; :attr:`settle_timeout` if not given.
        """
        if timeout is None:
            timeout = self.settle_timeout
        world = self.simulation.world
        simulator = self.simulation.simulator
        joint_names = self._moving_joint_names(arm)
        set_points = {
            joint_name: world.state[
                world.get_connection_by_name(joint_name).raw_dof.id
            ].position
            for joint_name in joint_names
        }
        simulated_time = timedelta()
        errors: Dict[str, float] = {}
        while simulated_time < timeout:
            self.simulation.step_simulation(self.control_period)
            simulated_time += self.control_period
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

    def both_pads_touch(self, gripper: EndEffector, target_body: Body) -> bool:
        """
        :param gripper: The gripper whose pads are checked.
        :param target_body: The body the pads are meant to hold.
        :return: Whether both pads are in contact with it right now.
        """
        return all(
            target_body.name.name
            in self.simulation.simulator.get_contact_bodies(
                body_name=pad.name.name, including_children=False
            ).result
            for pad in (gripper.thumb.tip, gripper.finger.tip)
        )

    @staticmethod
    def _moving_joint_names(arm: Arm) -> List[str]:
        """
        :param arm: The arm to read.
        :return: The names of the joints a motion of it commands.
        """
        return [
            connection.raw_dof.name.name
            for connection in arm.active_connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
