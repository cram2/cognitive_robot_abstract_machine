# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()


from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial

import numpy as np
import pytest

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor
from cramph.context import ContextExtension, StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.predefined_maps.building_floor import BuildingFloor
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

try:
    import rclpy
except ModuleNotFoundError:
    pass
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.utils import create_engine, drop_database

try:
    from coraplex.orm.ormatic_interface import Base
except ImportError:
    pass
try:
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ModuleNotFoundError:
    pass
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox

from .world_snapshot import WorldSnapshot
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm

from ..plan_running import expand, robot_extensions, statechart_of, world_of

# %% the arm a test runs with on any robot


def left_or_only_arm(robot: AbstractRobot) -> Arm:
    """
    :return: The left arm of a robot that names one, otherwise its first arm.
    """
    return robot.get_left_arm_if_specified() or robot.all_arms[0]


def right_or_only_arm(robot: AbstractRobot) -> Arm:
    """
    :return: The right arm of a robot that names one, otherwise its first arm.
    """
    return robot.get_right_arm_if_specified() or robot.all_arms[0]


# %% where a gripper grasps, which is what a goal is aimed at


def grasp_point_of(arm: Arm) -> np.ndarray:
    """
    :param arm: The arm whose gripper to read.
    :return: Where its fingers meet, in world coordinates. A goal puts this point on
        its target, not the tool frame, so this is what a reached goal is read at.
    """
    end_effector = arm.end_effector
    world = end_effector._world
    return (
        world.transform(end_effector.tool_P_grasp_when_closed, world.root)
        .to_np()
        .flatten()[:3]
    )


@pytest.fixture(scope="session")
def viz_marker_publisher():
    rclpy.init()
    node = rclpy.create_node("test_viz_marker_publisher")
    # VizMarkerPublisher(world, node)  # Initialize the publisher
    yield partial(VizMarkerPublisher, node=node)
    rclpy.shutdown()


# %% world rollback


@pytest.fixture(scope="function")
def pr2_apartment_context(pr2_apartment_world):
    """
    The shared PR2 apartment world, its robot and the context extensions of its plans,
    returned to its initial model and state after the test.
    """
    snapshot = WorldSnapshot.capture(pr2_apartment_world)
    pr2 = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
    yield pr2_apartment_world, pr2, robot_extensions(pr2)
    snapshot.restore()


@pytest.fixture(scope="function")
def simple_pr2_context(simple_pr2_world_setup):
    """
    The shared PR2 world in the simple apartment, its robot and the context extensions
    of its plans, returned to its initial model and state after the test.
    """
    world, robot_view, extensions = simple_pr2_world_setup
    snapshot = WorldSnapshot.capture(world)
    yield world, robot_view, extensions
    snapshot.restore()


@pytest.fixture(scope="function")
def stretch_apartment_context(stretch_apartment_world):
    """
    The shared Stretch apartment world, its robot and the context extensions of its
    plans, returned to its initial model and state after the test.
    """
    snapshot = WorldSnapshot.capture(stretch_apartment_world)
    robot = stretch_apartment_world.get_semantic_annotations_by_type(Stretch)[0]
    yield stretch_apartment_world, robot, robot_extensions(robot)
    snapshot.restore()


# %% database session


@pytest.fixture(scope="function")
def coraplex_testing_session():
    engine = create_engine("sqlite:///:memory:")
    session_maker = sessionmaker(engine)
    session = session_maker()
    Base.metadata.create_all(bind=session.bind)
    yield session
    drop_database(session.bind)
    session.close()
    engine.dispose()


# %% perception regions


@pytest.fixture
def whole_scene_region(pr2_apartment_context) -> VolumetricBoundingBox:
    """
    A region large enough to contain everything in the apartment fixture.

    Lets a perception test say "look everywhere" without restating the extents.
    """
    world, _, _ = pr2_apartment_context
    return VolumetricBoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        min_x=-10,
        min_y=-10,
        min_z=-10,
        max_x=10,
        max_y=10,
        max_z=10,
    )


@pytest.fixture
def empty_region(pr2_apartment_context) -> VolumetricBoundingBox:
    """
    A region well clear of everything in the apartment fixture.

    Lets a perception test say "look where nothing is" without restating the extents.
    """
    world, _, _ = pr2_apartment_context
    return VolumetricBoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        min_x=-10,
        min_y=-10,
        min_z=-10,
        max_x=-9,
        max_y=-9,
        max_z=-9,
    )


# %% building the giskard goals a test plan is made of


def tool_center_point_goal(
    robot: AbstractRobot,
    arm: Arm | None = None,
    target: Pose | None = None,
) -> CartesianPose:
    """
    Build the goal an action would build to move an arm's tool center point.

    Lets a test about plans and charts state which arm moves where without restating how
    an action assembles that goal.

    :param robot: The robot performing the plan, supplying the link the goal is
        expressed relative to.
    :param arm: Which arm's tool center point moves, the left or only arm of the robot
        by default.
    :param target: Where it should end up, the world's origin by default.
    :return: The goal moving that tool center point there.
    """
    return CartesianPose(
        root_link=RobotAccess(robot).controlled_root,
        tip_link=(
            left_or_only_arm(robot) if arm is None else arm
        ).end_effector.tool_frame,
        goal_pose=(
            Pose(reference_frame=robot._world.root) if target is None else target
        ),
    )


def motion_nodes_of(
    plan: StatechartNode, extensions: list[ContextExtension]
) -> list[StatechartNode]:
    """
    :param plan: The plan whose motions to read.
    :param extensions: The context extensions the plan is expanded with.
    :return: Every node the plan's steps expand into, so a test can look for a task
        without knowing whether the action wrapped it alongside speed caps or collision
        rules.
    """
    root = expand(plan, extensions)
    return [root, *root.descendants]


def nodes_of_type(
    plan: StatechartNode, node_type: type[StatechartNode]
) -> list[StatechartNode]:
    """
    :param plan: An expanded plan.
    :param node_type: The type of the nodes to collect.
    :return: Every node of `node_type` the plan runs, in tree order.
    """
    return [node for node in plan.descendants if isinstance(node, node_type)]


# %% a physically simulated world without the physics


@dataclass
class SimulationThatCountsItsSteps:
    """
    Stands in for a started simulation, counting the physics steps asked of it.
    """

    steps: list[timedelta] = field(default_factory=list)
    """
    Every period the physics was asked to advance by.
    """

    world: World = field(default_factory=World)
    """
    The world it simulates, without a servo, so nothing is waited for to settle.
    """

    def step_simulation(self, duration: timedelta) -> None:
        """
        :param duration: Simulated time to advance.
        """
        self.steps.append(duration)


def physically_simulated_executor(
    extensions: list[ContextExtension], **options
) -> PhysicallySimulatedPlanExecutor:
    """
    :param extensions: The context extensions of a plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess`.
    :param options: Further arguments of the executor.
    :return: An executor performing plans in physics, against a simulation that only
        counts its steps while the controller moves the world.
    """
    return PhysicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(world_of(extensions), extensions),
        simulation=SimulationThatCountsItsSteps(),
        **options,
    )


def motion_nodes_in_physics_of(
    plan: StatechartNode, extensions: list[ContextExtension]
) -> list[StatechartNode]:
    """
    :param plan: The plan whose motions to read.
    :param extensions: The context extensions the plan is expanded with.
    :return: Every node the plan's steps expand into when it is performed in physics.
    """
    statechart_of(physically_simulated_executor(extensions), plan)
    return [plan, *plan.descendants]
