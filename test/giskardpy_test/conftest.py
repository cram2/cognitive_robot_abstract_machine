# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()


import pytest

from giskardpy.middleware.ros2.utils.utils_for_tests import GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from cramph.statechart import Statechart
from krrood.symbolic_math.symbolic_math import logic_and
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
)
from semantic_digital_twin.robots.daisy import DAiSyJoint
from semantic_digital_twin.robots.hsrb import HSRBJoint
from semantic_digital_twin.robots.pr2 import PR2Joint


@pytest.fixture()
def better_pr2_pose():
    return {
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


@pytest.fixture()
def default_pr2_pose() -> dict[str, float]:
    """
    :return: Joint positions of a PR2 with both arms stretched out forwards.
    """
    return {
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


@pytest.fixture()
def pocky_pr2_pose() -> dict[str, float]:
    """
    :return: Joint positions of a PR2 with both arms bent and the grippers in front of it.
    """
    return {
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


@pytest.fixture()
def default_hsr_pose() -> dict[str, float]:
    """
    :return: Joint positions of an HSR with its arm lift at the bottom and every other
        arm and head joint near zero.
    """
    return {
        HSRBJoint.ARM_FLEX: -0.03,
        HSRBJoint.ARM_LIFT: 0.01,
        HSRBJoint.ARM_ROLL: 0.0,
        HSRBJoint.HEAD_PAN: 0.0,
        HSRBJoint.HEAD_TILT: 0.0,
        HSRBJoint.WRIST_FLEX: 0.0,
        HSRBJoint.WRIST_ROLL: 0.0,
    }


@pytest.fixture()
def default_daisy_pose() -> dict[str, float]:
    """
    :return: Joint positions of a DAiSy with both arms in the same configuration.
    """
    return {
        DAiSyJoint.LEFT_SHOULDER_PAN: 0,
        DAiSyJoint.LEFT_SHOULDER_LIFT: -1.57,
        DAiSyJoint.LEFT_ELBOW: 1,
        DAiSyJoint.LEFT_WRIST_1: 0,
        DAiSyJoint.LEFT_WRIST_2: 0,
        DAiSyJoint.LEFT_WRIST_3: 0,
        DAiSyJoint.RIGHT_SHOULDER_PAN: 0,
        DAiSyJoint.RIGHT_SHOULDER_LIFT: -1.57,
        DAiSyJoint.RIGHT_ELBOW: 1,
        DAiSyJoint.RIGHT_WRIST_1: 0,
        DAiSyJoint.RIGHT_WRIST_2: 0,
        DAiSyJoint.RIGHT_WRIST_3: 0,
    }


@pytest.fixture()
def orientation_speed_ratio_tolerance() -> float:
    """
    :return: How much faster a far away orientation goal may turn than a close one.
        Both turn at the speed the reference velocity sets once they are under way; the
        margin covers that the closer goal starts braking sooner.
    """
    return 1.15


@pytest.fixture(scope="function")
def pr2_with_box(pr2_world_copy) -> World:
    with pr2_world_copy.modify_world():
        box = Body(
            name=PrefixedName("box"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(1, 1, 1))]),
            collision=ShapeCollection(shapes=[Box(scale=Scale(1, 1, 1))]),
        )
        root_C_box = FixedConnection(
            parent=pr2_world_copy.root,
            child=box,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.2, z=0.3, reference_frame=pr2_world_copy.root
            ),
        )
        pr2_world_copy.add_connection(root_C_box)
    return pr2_world_copy


@pytest.fixture()
def giskard_factory(init_rospy, robot: GiskardTester):
    def _create_giskard(seed_joint_state: dict[str, float]) -> GiskardTester:
        parse_seed_joint_state = {
            robot.api.world.get_connection_by_name(name): target
            for name, target in seed_joint_state.items()
        }
        statechart = Statechart(context=robot.api.context)

        initial_config = SetSeedConfiguration(
            name="initial configuration",
            seed_configuration=JointState.from_mapping(parse_seed_joint_state),
        )
        statechart.add_node(initial_config)

        if robot.has_odometry_joint():
            base_goal = HomogeneousTransformationMatrix(
                reference_frame=robot.api.world.root
            )
            base_pose_reached = SetOdometry(name="initial pose", base_pose=base_goal)
            statechart.add_node(base_pose_reached)
            done = logic_and(
                initial_config.observes_true,
                base_pose_reached.observes_true,
            )
        else:
            done = initial_config.observes_true
        end = EndMotion(name="end")
        statechart.add_node(end)
        end.start_condition = done
        robot.api.compile(statechart)
        robot.api.execute()
        return robot

    return _create_giskard


@pytest.fixture()
def giskard(giskard_factory, default_joint_state):
    return giskard_factory(default_joint_state)


def _symmetric_prismatic_limits(
    position: float | None,
    velocity: float,
    jerk: float | None = None,
    acceleration: float | None = None,
) -> DegreeOfFreedomLimits:
    """
    Builds symmetric prismatic degree-of-freedom limits.

    :param position: Position limit, ``None`` for a joint without position limits.
    :param velocity: Velocity limit.
    :param jerk: Jerk limit, ``None`` for a joint without one.
    :param acceleration: Acceleration limit, ``None`` for a joint without one.
    :return: Limits mirrored around zero.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(
            position=None if position is None else -position,
            velocity=-velocity,
            acceleration=None if acceleration is None else -acceleration,
            jerk=None if jerk is None else -jerk,
        ),
        upper=DerivativeMap(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
        ),
    )


def _make_prismatic_world(dof_limits: list[DegreeOfFreedomLimits]) -> World:
    """
    Builds a world with one Z-axis prismatic connection per given degree-of-freedom
    limit set.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        for index, limits in enumerate(dof_limits):
            child = Body(
                name=PrefixedName("robot" if index == 0 else f"robot{index + 1}")
            )
            dof = DegreeOfFreedom(limits=limits, has_hardware_interface=True)
            world.add_degree_of_freedom(dof)
            world.add_connection(
                PrismaticConnection(
                    parent=map_body, child=child, raw_dof=dof, axis=Vector3.Z()
                )
            )
    MinimalRobot.from_world(world)
    return world


@pytest.fixture()
def prismatic_bot():
    return _make_prismatic_world([_symmetric_prismatic_limits(1, 1)])


@pytest.fixture()
def prismatic_bot2():
    return _make_prismatic_world(
        [_symmetric_prismatic_limits(1, 1), _symmetric_prismatic_limits(0.5, 0.5)]
    )


@pytest.fixture()
def prismatic_bot_with_jerk_limit():
    return _make_prismatic_world([_symmetric_prismatic_limits(1, 1, jerk=40)])


@pytest.fixture()
def prismatic_world_no_position_limits():
    return _make_prismatic_world([_symmetric_prismatic_limits(None, 1)])


JERK_LIMIT_TOO_LOW_FOR_SHORT_HORIZONS = 1.0
"""
A jerk limit with which a joint needs dozens of time steps at 20 Hz to brake from a
velocity of 1 to rest.
"""

ACCELERATION_LIMIT = 2.0
"""
An acceleration limit well below what the default braking time implies for a velocity
limit of 1.
"""


@pytest.fixture(
    params=[2, None], ids=["with_position_limits", "without_position_limits"]
)
def prismatic_world_with_acceleration_limit(request):
    """
    A prismatic joint with an acceleration limit of its own, with and without position
    limits.
    """
    return _make_prismatic_world(
        [_symmetric_prismatic_limits(request.param, 1, acceleration=ACCELERATION_LIMIT)]
    )


@pytest.fixture(
    params=[1, None], ids=["with_position_limits", "without_position_limits"]
)
def prismatic_world_with_low_jerk_limit(request):
    """
    A prismatic joint whose own jerk limit is too low to brake from its velocity limit
    within a short prediction horizon, with and without position limits.
    """
    return _make_prismatic_world(
        [
            _symmetric_prismatic_limits(
                request.param, 1, jerk=JERK_LIMIT_TOO_LOW_FOR_SHORT_HORIZONS
            )
        ]
    )
