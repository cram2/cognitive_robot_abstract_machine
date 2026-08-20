from __future__ import annotations

import cProfile
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Dict, List, Type

import numpy as np

from giskardpy.middleware.ros2 import rospy
from experiments.control_loop_experiments.control_loop_profiler import (
    CallTreeProfile,
    ControlLoopProfiler,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2StandaloneInterface,
    WorldWithPR2Config,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import GiskardTester
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph
from krrood.symbolic_math.symbolic_math import trinary_logic_and
from krrood.utils import recursive_subclasses
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% how much the measurement records


class PlotterMode(StrEnum):
    """
    Whether the post goal plotters record the motion while it is measured.

    Recording costs time in every cycle, so the same motion is measured under both modes
    to show what the plotters cost.
    """

    PLAIN = "plain"
    """
    The plotters are off, so only the control loop itself is measured.
    """

    DEBUG = "debug"
    """
    The plotters record the motion, as they do when a motion is debugged.
    """

    @property
    def records_trajectory(self) -> bool:
        """
        Whether Giskard has to keep the trajectory around for the plotters.

        :return: True if the trajectory has to be kept.
        """
        return self is PlotterMode.DEBUG


# %% the robot under measurement


@dataclass
class BenchmarkRobot(GiskardTester):
    """
    A PR2 that runs its motions in place, so the control loop is measured without a
    simulator or hardware in between.
    """

    plotter_mode: PlotterMode = PlotterMode.DEBUG
    """
    Whether the post goal plotters record the motion, which costs time in every cycle.
    """

    target_frequency: float = 20.0
    """
    Frequency the controller is discretized for, in hertz.
    """

    default_joint_state: ClassVar[Dict[str, float]] = {
        PR2Joint.RIGHT_ELBOW_FLEX: -0.15,
        PR2Joint.RIGHT_FOREARM_ROLL: 0,
        PR2Joint.RIGHT_SHOULDER_LIFT: 0,
        PR2Joint.RIGHT_SHOULDER_PAN: 0,
        PR2Joint.RIGHT_UPPER_ARM_ROLL: 0,
        PR2Joint.RIGHT_WRIST_FLEX: -0.10001,
        PR2Joint.RIGHT_WRIST_ROLL: 0,
        PR2Joint.LEFT_ELBOW_FLEX: -0.15,
        PR2Joint.LEFT_FOREARM_ROLL: 0,
        PR2Joint.LEFT_SHOULDER_LIFT: 0,
        PR2Joint.LEFT_SHOULDER_PAN: 0,
        PR2Joint.LEFT_UPPER_ARM_ROLL: 0,
        PR2Joint.LEFT_WRIST_FLEX: -0.10001,
        PR2Joint.LEFT_WRIST_ROLL: 0,
        PR2Joint.TORSO_LIFT: 0.2,
        PR2Joint.HEAD_PAN: 0,
        PR2Joint.HEAD_TILT: 0,
        PR2Joint.LEFT_GRIPPER_LEFT_FINGER: 0.55,
        PR2Joint.RIGHT_GRIPPER_LEFT_FINGER: 0.55,
    }
    """
    The neutral configuration the arms start from.
    """

    better_pose: ClassVar[Dict[str, float]] = {
        PR2Joint.RIGHT_SHOULDER_PAN: -1.7125,
        PR2Joint.RIGHT_SHOULDER_LIFT: -0.25672,
        PR2Joint.RIGHT_UPPER_ARM_ROLL: -1.46335,
        PR2Joint.RIGHT_ELBOW_FLEX: -2.12,
        PR2Joint.RIGHT_FOREARM_ROLL: 1.76632,
        PR2Joint.RIGHT_WRIST_FLEX: -0.10001,
        PR2Joint.RIGHT_WRIST_ROLL: 0.05106,
        PR2Joint.LEFT_SHOULDER_PAN: 1.9652,
        PR2Joint.LEFT_SHOULDER_LIFT: -0.26499,
        PR2Joint.LEFT_UPPER_ARM_ROLL: 1.3837,
        PR2Joint.LEFT_ELBOW_FLEX: -2.12,
        PR2Joint.LEFT_FOREARM_ROLL: 16.99,
        PR2Joint.LEFT_WRIST_FLEX: -0.10001,
        PR2Joint.LEFT_WRIST_ROLL: 0,
        PR2Joint.TORSO_LIFT: 0.2,
        PR2Joint.LEFT_GRIPPER_LEFT_FINGER: 0.55,
        PR2Joint.RIGHT_GRIPPER_LEFT_FINGER: 0.55,
        PR2Joint.HEAD_PAN: 0,
        PR2Joint.HEAD_TILT: 0,
    }
    """
    The configuration with the arms tucked in, used when the robot has to drive around.
    """

    pocky_pose: ClassVar[Dict[str, float]] = {
        PR2Joint.RIGHT_ELBOW_FLEX: -1.29610152504,
        PR2Joint.RIGHT_FOREARM_ROLL: -0.0301682323805,
        PR2Joint.RIGHT_SHOULDER_LIFT: 1.20324921318,
        PR2Joint.RIGHT_SHOULDER_PAN: -0.73456435706,
        PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.70790051778,
        PR2Joint.RIGHT_WRIST_FLEX: -0.10001,
        PR2Joint.RIGHT_WRIST_ROLL: 0.258268529825,
        PR2Joint.LEFT_ELBOW_FLEX: -1.29610152504,
        PR2Joint.LEFT_FOREARM_ROLL: 0.0301682323805,
        PR2Joint.LEFT_SHOULDER_LIFT: 1.20324921318,
        PR2Joint.LEFT_SHOULDER_PAN: 0.73456435706,
        PR2Joint.LEFT_UPPER_ARM_ROLL: 0.70790051778,
        PR2Joint.LEFT_WRIST_FLEX: -0.1001,
        PR2Joint.LEFT_WRIST_ROLL: -0.258268529825,
        PR2Joint.TORSO_LIFT: 0.2,
        PR2Joint.HEAD_PAN: 0,
        PR2Joint.HEAD_TILT: 0,
        PR2Joint.LEFT_GRIPPER_LEFT_FINGER: 0.55,
        PR2Joint.RIGHT_GRIPPER_LEFT_FINGER: 0.55,
    }
    """
    The configuration with both arms stretched out in front of the robot.
    """

    def setup_giskard(self) -> Giskard:
        records_trajectory = self.plotter_mode.records_trajectory
        return Giskard(
            world_config=WorldWithPR2Config(
                urdf=load_xacro(
                    "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
                )
            ),
            robot_interface_config=PR2StandaloneInterface(),
            server_config=GiskardServerConfig(
                execution_mode=ExecutionMode.STANDALONE,
                debug_mode=records_trajectory,
                plot_trajectory=records_trajectory,
            ),
            qp_controller_config=QPControllerConfig(
                target_frequency=self.target_frequency
            ),
        )

    @property
    def control_delta_time(self) -> float:
        """
        Seconds one control cycle may take before the robot is commanded too late.

        :return: The length of one control cycle in seconds.
        """
        return self.giskard.executor.context.qp_controller_config.control_dt

    def get_kinematic_structure_entity(self, name: str) -> KinematicStructureEntity:
        """
        The kinematic structure entity of the given name.

        :param name: Name of the kinematic structure entity in the world.
        :return: The entity registered under that name.
        """
        return self.api.world.get_kinematic_structure_entity_by_name(name)

    def teleport_to_configuration(self, joint_state: Dict[str, float]) -> None:
        """
        Teleport the robot into the given configuration and reset its odometry.

        This runs as a motion of its own, so the measured motion starts from a defined
        state.

        :param joint_state: Target position for each connection, by connection name.
        """
        connections = {
            self.api.world.get_connection_by_name(name): target
            for name, target in joint_state.items()
        }
        motion_statechart = MotionStatechart()
        seed_configuration = SetSeedConfiguration(
            name="initial configuration",
            seed_configuration=JointState.from_mapping(connections),
        )
        motion_statechart.add_node(seed_configuration)
        reached = seed_configuration.observation_variable
        if self.has_odometry_joint():
            odometry = SetOdometry(
                name="initial pose",
                base_pose=HomogeneousTransformationMatrix(
                    reference_frame=self.api.world.root
                ),
            )
            motion_statechart.add_node(odometry)
            reached = trinary_logic_and(reached, odometry.observation_variable)
        end = EndMotion(name="end")
        end.start_condition = reached
        motion_statechart.add_node(end)
        self.api.execute(motion_statechart)

    def add_environment(self, name: str, package_path: str, pose: Pose) -> None:
        """
        Load a whole environment into the world, which grows the cost of forward
        kinematics and collision checking.

        :param name: Name the environment is added under.
        :param package_path: Package path of the xacro or urdf describing the
            environment.
        :param pose: Where the environment is placed in the world.
        """
        self.default_env_name = name
        self.add_urdf_to_world(name=name, urdf=load_xacro(package_path), pose=pose)


# %% scenarios


@dataclass
class BenchmarkScenario(ABC):
    """
    One motion whose control loop is measured.
    """

    name: ClassVar[str]
    """
    Name the scenario is reported under.
    """

    @abstractmethod
    def seed_joint_state(self, robot: BenchmarkRobot) -> Dict[str, float]:
        """
        The configuration the robot is teleported into before the motion starts.

        :param robot: The robot the motion is measured on.
        :return: Target position for each connection, by connection name.
        """

    def prepare(self, robot: BenchmarkRobot) -> None:
        """
        Put everything the motion needs into the world before it is measured.

        :param robot: The robot the motion is measured on.
        """

    @abstractmethod
    def build_motion_statechart(self, robot: BenchmarkRobot) -> MotionStatechart:
        """
        Describe the motion that is measured.

        :param robot: The robot the motion is measured on.
        :return: The statechart describing the measured motion.
        """


@dataclass
class CartesianGoalScenario(BenchmarkScenario):
    """
    Moves one gripper backwards, with nothing else active.

    The cheapest motion Giskard can run: no collision checking and a world holding
    nothing but the robot.
    """

    name: ClassVar[str] = "cart_goal_1eef"

    def seed_joint_state(self, robot: BenchmarkRobot) -> Dict[str, float]:
        return robot.default_joint_state

    def build_motion_statechart(self, robot: BenchmarkRobot) -> MotionStatechart:
        tip = robot.get_kinematic_structure_entity("r_gripper_tool_frame")
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            cartesian_goal := CartesianPose(
                root_link=robot.get_kinematic_structure_entity("base_footprint"),
                tip_link=tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=-0.2, reference_frame=tip
                ),
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cartesian_goal))
        return motion_statechart


@dataclass
class CollisionAvoidanceScenario(BenchmarkScenario):
    """
    Reaches around a table sized box, so collisions are checked in every cycle.
    """

    name: ClassVar[str] = "avoid_collision_go_around_corner"

    def seed_joint_state(self, robot: BenchmarkRobot) -> Dict[str, float]:
        return robot.pocky_pose

    def prepare(self, robot: BenchmarkRobot) -> None:
        robot.add_box_to_world(
            name="box",
            size=(1.0, 1.0, 1.0),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.2,
                z=0.3,
                reference_frame=robot.get_kinematic_structure_entity("map"),
            ),
        )

    def build_motion_statechart(self, robot: BenchmarkRobot) -> MotionStatechart:
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            CartesianPose(
                root_link=robot.default_root,
                tip_link=robot.get_kinematic_structure_entity("r_gripper_tool_frame"),
                goal_pose=Pose.from_xyz_axis_angle(
                    x=0.8,
                    y=-0.38,
                    z=0.84,
                    axis=Vector3.Y(),
                    angle=np.pi / 2.0,
                    reference_frame=robot.default_root,
                ),
            )
        )
        motion_statechart.add_node(
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidExternalCollisions(
                        buffer_zone_distance=0.1,
                        violated_distance=0.0,
                        robot=robot.api.robot,
                    )
                ]
            )
        )
        motion_statechart.add_node(ExternalCollisionAvoidance(robot=robot.api.robot))
        motion_statechart.add_node(local_minimum := LocalMinimumReached())
        motion_statechart.add_node(EndMotion.when_true(local_minimum))
        return motion_statechart


@dataclass
class ApartmentDrivingScenario(BenchmarkScenario):
    """
    Drives the base two meters through a fully loaded apartment.
    """

    name: ClassVar[str] = "drive_into_apartment"

    def seed_joint_state(self, robot: BenchmarkRobot) -> Dict[str, float]:
        return robot.better_pose

    def prepare(self, robot: BenchmarkRobot) -> None:
        robot.add_environment(
            name="iai_apartment",
            package_path="package://iai_apartment/urdf/apartment.urdf",
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.5, y=1.4, yaw=np.pi, reference_frame=robot.api.world.root
            ),
        )

    def build_motion_statechart(self, robot: BenchmarkRobot) -> MotionStatechart:
        base = robot.get_kinematic_structure_entity("base_footprint")
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            cartesian_goal := CartesianPose(
                root_link=robot.get_kinematic_structure_entity("map"),
                tip_link=base,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.4, y=-2.0, reference_frame=base
                ),
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cartesian_goal))
        return motion_statechart


@dataclass
class KitchenPointingScenario(BenchmarkScenario):
    """
    Drives through a kitchen while keeping the camera on a fridge handle and folding the
    arms, so several tasks run at once in a large world.
    """

    name: ClassVar[str] = "pointing_kitchen"

    def seed_joint_state(self, robot: BenchmarkRobot) -> Dict[str, float]:
        return robot.better_pose

    def prepare(self, robot: BenchmarkRobot) -> None:
        robot.add_environment(
            name="iai_kitchen",
            package_path="package://iai_kitchen/urdf_obj/iai_kitchen_python.urdf.xacro",
            pose=HomogeneousTransformationMatrix(reference_frame=robot.api.world.root),
        )

    def build_motion_statechart(self, robot: BenchmarkRobot) -> MotionStatechart:
        camera = robot.get_kinematic_structure_entity("head_mount_kinect_rgb_link")
        map_frame = robot.get_kinematic_structure_entity("map")
        pointing_axis = Vector3.X(reference_frame=camera)
        handle_point = robot.api.world.compute_forward_kinematics(
            root=map_frame,
            tip=robot.get_kinematic_structure_entity("iai_fridge_door_handle"),
        ).to_position()
        arm_pose = {
            name: position
            for name, position in robot.better_pose.items()
            if name not in (PR2Joint.HEAD_PAN, PR2Joint.HEAD_TILT)
        }
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            sequence := Sequence(
                [
                    Pointing(
                        root_link=map_frame,
                        tip_link=camera,
                        goal_point=handle_point,
                        pointing_axis=pointing_axis,
                    ),
                    Parallel(
                        [
                            Pointing(
                                root_link=map_frame,
                                tip_link=camera,
                                goal_point=handle_point,
                                pointing_axis=pointing_axis,
                            ),
                            CartesianPose(
                                root_link=map_frame,
                                tip_link=robot.get_kinematic_structure_entity(
                                    "base_footprint"
                                ),
                                goal_pose=HomogeneousTransformationMatrix.from_xyz_axis_angle(
                                    y=2.0,
                                    axis=Vector3.Z(),
                                    angle=1,
                                    reference_frame=robot.get_kinematic_structure_entity(
                                        "base_footprint"
                                    ),
                                ),
                            ),
                            JointPositionList(
                                goal_state=JointState.from_str_dict(
                                    arm_pose, world=robot.api.world
                                )
                            ),
                        ]
                    ),
                ]
            )
        )
        motion_statechart.add_node(EndMotion.when_true(sequence))
        return motion_statechart


@dataclass
class LongSequenceScenario(BenchmarkScenario):
    """
    Reaches a chain of waypoints with both grippers while avoiding self collisions.

    The motion is long on purpose: a short goal cannot say whether the cycle time is
    stable, and this one runs long enough for the tail of the distribution to mean
    something.
    """

    name: ClassVar[str] = "long_sequence"

    def seed_joint_state(self, robot: BenchmarkRobot) -> Dict[str, float]:
        return robot.better_pose

    waypoint_offsets: ClassVar[List[float]] = [-0.15, 0.15, -0.2, 0.2, -0.1, 0.1]
    """
    How far along x each waypoint moves the gripper, relative to the one before.
    """

    def build_motion_statechart(self, robot: BenchmarkRobot) -> MotionStatechart:
        tips = [
            robot.get_kinematic_structure_entity("l_gripper_tool_frame"),
            robot.get_kinematic_structure_entity("r_gripper_tool_frame"),
        ]
        waypoints = [
            CartesianPose(
                root_link=robot.get_kinematic_structure_entity("base_footprint"),
                tip_link=tips[index % len(tips)],
                goal_pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=offset, reference_frame=tips[index % len(tips)]
                ),
            )
            for index, offset in enumerate(self.waypoint_offsets)
        ]
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(sequence := Sequence(waypoints))
        motion_statechart.add_node(SelfCollisionAvoidance(robot=robot.api.robot))
        motion_statechart.add_node(EndMotion.when_true(sequence))
        return motion_statechart


BENCHMARK_SCENARIOS: Dict[str, Type[BenchmarkScenario]] = {
    scenario.name: scenario
    for scenario in (
        CartesianGoalScenario,
        CollisionAvoidanceScenario,
        LongSequenceScenario,
        ApartmentDrivingScenario,
        KitchenPointingScenario,
    )
}
"""
Every scenario the control loop can be measured on, by name.
"""


# %% running a scenario


@dataclass
class IsolatedBenchmarkSession(AbstractContextManager):
    """
    Gives one measurement a ros node and a symbol graph of its own.

    A pytest run gets both from its fixtures; a script has to build them itself.
    """

    node_name: str = "giskard"
    """
    Name of the ros node the measured Giskard runs on.
    """

    _class_diagram: ClassDiagram | None = field(init=False, default=None)
    """
    The class diagram backing the symbol graph of this session.
    """

    def __enter__(self) -> IsolatedBenchmarkSession:
        SymbolGraph.clear()
        self._class_diagram = ClassDiagram(
            recursive_subclasses(Symbol) + [World],
            introspector=DescriptorAwareIntrospector(),
        )
        SymbolGraph(_class_diagram=self._class_diagram)
        rospy.init_node(self.node_name)
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        rospy.shutdown()
        SymbolGraph.clear()
        self._class_diagram.clear()


@dataclass
class ScenarioRunner:
    """
    Sets a scenario up, measures its motion and takes the robot down again.
    """

    plotter_mode: PlotterMode = PlotterMode.DEBUG
    """
    Whether the post goal plotters record the motion while it is measured.
    """

    target_frequency: float = 20.0
    """
    Frequency the controller is discretized for, in hertz.
    """

    python_profiler: cProfile.Profile | None = None
    """
    Records which python functions the measured motion spends its time in.

    It is only active while the motion runs, so that building the world and loading its
    meshes cannot drown out the control loop.
    """

    def run(self, scenario: BenchmarkScenario) -> CallTreeProfile:
        """
        Measure one motion of the given scenario on a freshly built robot.

        :param scenario: The scenario whose motion is measured.
        :return: The call tree recorded while the motion ran.
        """
        robot = BenchmarkRobot(
            plotter_mode=self.plotter_mode, target_frequency=self.target_frequency
        )
        try:
            robot.teleport_to_configuration(scenario.seed_joint_state(robot))
            scenario.prepare(robot)
            motion_statechart = scenario.build_motion_statechart(robot)
            profiler = ControlLoopProfiler(
                scenario_name=scenario.name, control_dt=robot.control_delta_time
            )
            with profiler:
                self._execute(robot, motion_statechart)
            return profiler.profile
        finally:
            robot.close()

    def _execute(
        self, robot: BenchmarkRobot, motion_statechart: MotionStatechart
    ) -> None:
        """
        Run the motion, with the python profiler active if one was given.

        :param robot: The robot the motion is executed on.
        :param motion_statechart: The motion that is executed.
        """
        if self.python_profiler is None:
            robot.api.execute(motion_statechart)
            return
        self.python_profiler.enable()
        try:
            robot.api.execute(motion_statechart)
        finally:
            self.python_profiler.disable()
